from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import os
import uuid
import logging
import shutil
import sys
from pymongo import MongoClient
from typing import List, Optional, Dict, Any
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import jwt
from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pydantic import BaseModel, EmailStr, Field
from dotenv import load_dotenv

# JWT settings
SECRET_KEY = "documorph_secret_key"  # In production, use a secure key stored in environment
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

# Create directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Import from the wrapper module
try:
    from documorph_wrapper import (
        process_document, 
        apply_template_to_document,
        initialize_db_templates,
        get_templates_for_user,
        UserTier
    )
except ImportError:
    # Fallback to direct import if wrapper not available
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from backend.documorph_ai import (
        process_document, 
        apply_template_to_document,
        initialize_db_templates,
        get_templates_for_user,
        UserTier
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="DocuMorph AI API")

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,  # Allow cookies for cross-origin requests
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],  # Needed for downloads
)

# Import dotenv for environment variables
load_dotenv()

# Store environment variables for easy access
ENV_CONFIG = {
    "MONGO_URI": os.environ.get("MONGO_URI", "mongodb://localhost:27017"),
    "LANGCHAIN_PROJECT": os.environ.get("LANGCHAIN_PROJECT", ""),
    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    "GROQ_API_KEY": os.environ.get("groq_api_key", ""),
    "SERPAPI_KEY": os.environ.get("serpapi", ""),
    "GOOGLE_CLIENT_ID": os.environ.get("GOOGLE_CLIENT_ID", ""),
    "CLIENT_SECRET": os.environ.get("Client_secret", ""),
    "GOOGLE_API_KEY": os.environ.get("goog_api_key", ""),
    "GOOGLE_REDIRECT_URI": os.environ.get("GOOGLE_REDIRECT_URI", ""),
    "APP_ENV": os.environ.get("APP_ENV", "development"),
    "JWT_SECRET": os.environ.get("JWT_SECRET", "supersecretkey"),
    "JWT_ALGORITHM": os.environ.get("JWT_ALGORITHM", "HS256"),
    "JWT_EXPIRATION_MINUTES": int(os.environ.get("JWT_EXPIRATION_MINUTES", 60 * 24))
}

# Initialize MongoDB connection
@app.on_event("startup")
async def startup_db_client():
    try:
        # Get MongoDB URI from environment config
        mongo_uri = ENV_CONFIG["MONGO_URI"]
        app.mongodb_client = AsyncIOMotorClient(mongo_uri)
        app.db = app.mongodb_client.documorph
        
        # Check connection by attempting to get server info
        await app.mongodb_client.admin.command('ping')
        logger.info("Connected to MongoDB database")
        
        # Create admin user if not exists
        admin_user = await app.db["users"].find_one({"email": "mandarak123@gmail.com"})
        if not admin_user:
            hashed_password = get_password_hash("Mak@1944")
            admin_user_data = {
                "user_id": str(uuid.uuid4()),
                "email": "mandarak123@gmail.com",
                "password": hashed_password,
                "name": "Admin User",
                "isAdmin": True,
                "isDevMode": True,
                "subscription": "premium",
                "created_at": datetime.now().isoformat()
            }
            await app.db["users"].insert_one(admin_user_data)
            logger.info("Admin user created with email: mandarak123@gmail.com")
        else:
            # Ensure admin user has dev mode enabled
            if not admin_user.get("isDevMode", False) or not admin_user.get("isAdmin", False):
                await app.db["users"].update_one(
                    {"email": "mandarak123@gmail.com"},
                    {"$set": {"isDevMode": True, "isAdmin": True, "subscription": "premium"}}
                )
                logger.info("Updated admin user with dev mode capabilities")
        
        # Initialize document, template, and user collections
        app.docs_collection = app.db["documents"]
        app.templates_collection = app.db["templates"]
        app.users_collection = app.db["users"]
        
        # Initialize templates for admin user
        admin_user = await app.db["users"].find_one({"email": "mandarak123@gmail.com"})
        if admin_user:
            try:
                # Try to import from modules, otherwise define a simple placeholder
                try:
                    # Attempt to import from modules
                    from modules.documorph_ai import initialize_db_templates
                except ImportError:
                    try:
                        # Try an alternate import path
                        from server.modules.documorph_ai import initialize_db_templates
                    except ImportError:
                        # Define a placeholder function if import fails
                        async def initialize_db_templates(user_id):
                            logger.info(f"Templates already exist for user {user_id}")
                            return
                
                # Initialize templates
                await initialize_db_templates(admin_user["user_id"])
                logger.info("Templates initialized for admin user")
            except Exception as e:
                logger.error(f"Failed to initialize templates: {str(e)}")
                
    except Exception as e:
        logger.error(f"MongoDB connection error: {str(e)}")
        # Provide fallback for testing
        app.mongodb_client = None
        app.db = None
        app.docs_collection = None
        app.templates_collection = None
        app.users_collection = None
        raise

@app.on_event("shutdown")
async def shutdown_db_client():
    if app.mongodb_client:
        app.mongodb_client.close()

# In-memory storage (fallback when MongoDB is not available)
in_memory_documents = {}
in_memory_templates = {}
in_memory_users = {}

# User functions
def create_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(email: str):
    if app.users_collection:
        user = app.users_collection.find_one({"email": email})
        return user
    else:
        return in_memory_users.get(email)

def authenticate_user(email: str, password: str):
    user = get_user(email)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user

# Authentication dependency
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        user = get_user(email)
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return {
            "user_id": str(user["_id"]) if "_id" in user else user["id"],
            "email": user["email"],
            "tier": user.get("tier", UserTier.FREE)
        }
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# Optional auth for endpoints that can work without authentication
async def get_optional_user(authorization: Optional[str] = Header(None)):
    if not authorization:
        return {"user_id": "anonymous", "tier": UserTier.FREE}
        
    try:
        scheme, token = authorization.split()
        if scheme.lower() != 'bearer':
            return {"user_id": "anonymous", "tier": UserTier.FREE}
            
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return {"user_id": "anonymous", "tier": UserTier.FREE}
            
        user = get_user(email)
        if user is None:
            return {"user_id": "anonymous", "tier": UserTier.FREE}
            
        return {
            "user_id": str(user["_id"]) if "_id" in user else user["id"],
            "email": user["email"],
            "tier": user.get("tier", UserTier.FREE)
        }
    except:
        return {"user_id": "anonymous", "tier": UserTier.FREE}

# Authentication API endpoints
@app.post("/api/auth/register")
async def register_user(form_data: Dict[str, Any]):
    try:
        email = form_data.get("email")
        password = form_data.get("password")
        name = form_data.get("name", "")
        
        if not email or not password:
            raise HTTPException(status_code=400, detail="Email and password are required")
        
        # Check if user already exists
        existing_user = get_user(email)
        if existing_user:
            raise HTTPException(status_code=400, detail="User already registered")
        
        # Create new user
        user_id = str(uuid.uuid4())
        user_data = {
            "id": user_id,
            "email": email,
            "name": name,
            "hashed_password": get_password_hash(password),
            "tier": UserTier.FREE,
            "created_at": datetime.now()
        }
        
        # Store user
        if app.users_collection:
            app.users_collection.insert_one(user_data)
        else:
            in_memory_users[email] = user_data
        
        # Initialize templates for the new user
        if app.templates_collection:
            initialize_db_templates(user_id)
        
        return {"id": user_id, "email": email, "name": name}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Registration error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/api/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        user = authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(status_code=401, detail="Incorrect email or password")
            
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_token(
            data={"sub": user["email"]}, 
            expires_delta=access_token_expires
        )
        
        user_id = str(user["_id"]) if "_id" in user else user["id"]
        
        return {
            "access_token": access_token, 
            "token_type": "bearer",
            "user": {
                "id": user_id,
                "email": user["email"],
                "name": user.get("name", ""),
                "tier": user.get("tier", UserTier.FREE)
            }
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Login error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@app.get("/api/auth/me")
async def get_user_profile(current_user = Depends(get_current_user)):
    try:
        return {
            "id": current_user["user_id"],
            "email": current_user["email"],
            "tier": current_user["tier"]
        }
    except Exception as e:
        logger.error(f"Profile error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not retrieve profile: {str(e)}")

# API Endpoints
@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    current_user = Depends(get_optional_user)
):
    try:
        # Create a unique ID for this document
        document_id = str(uuid.uuid4())
        
        # Determine file extension
        file_extension = os.path.splitext(file.filename)[1]
        
        # Save the uploaded file
        upload_dir = os.path.join("uploads", document_id)
        os.makedirs(upload_dir, exist_ok=True)
        upload_path = os.path.join(upload_dir, f"original{file_extension}")
        
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Store document info
        document_info = {
            "id": document_id,
            "original_name": file.filename,
            "upload_path": upload_path,
            "status": "uploaded",
            "upload_date": datetime.now(),
            "file_type": file_extension.lower().replace(".", ""),
            "user_id": current_user["user_id"]
        }
        
        # Store in MongoDB or in-memory
        if app.docs_collection:
            app.docs_collection.insert_one(document_info)
        else:
            in_memory_documents[document_id] = document_info
        
        # Process document in background
        if background_tasks:
            background_tasks.add_task(
                process_document_task, 
                document_id, 
                upload_path, 
                document_info["file_type"],
                current_user["tier"]
            )
        
        return {"documentId": document_id, "message": "Document uploaded successfully"}
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_document_task(document_id, file_path, file_type, user_tier):
    try:
        # Process the document (extract structure, tables, figures, etc.)
        processed_data = process_document(file_path, file_type, user_tier)
        
        # Update document status
        if app.docs_collection:
            app.docs_collection.update_one(
                {"id": document_id},
                {"$set": {
                    "status": "processed",
                    "processed_data": processed_data
                }}
            )
        else:
            in_memory_documents[document_id]["status"] = "processed"
            in_memory_documents[document_id]["processed_data"] = processed_data
            
    except Exception as e:
        logger.error(f"Processing error for document {document_id}: {str(e)}", exc_info=True)
        # Update document with error status
        if app.docs_collection:
            app.docs_collection.update_one(
                {"id": document_id},
                {"$set": {"status": "error", "error": str(e)}}
            )
        else:
            in_memory_documents[document_id]["status"] = "error"
            in_memory_documents[document_id]["error"] = str(e)

@app.get("/api/templates")
async def get_templates(user = Depends(get_optional_user)):
    try:
        user_id = user["user_id"]
        user_tier = user["tier"]
        
        # Initialize templates if needed
        if app.templates_collection:
            initialize_db_templates(user_id)
            templates = get_templates_for_user(user_id, app.templates_collection, user_tier)
        else:
            # Fallback to in-memory templates
            if user_id not in in_memory_templates:
                # Create default templates
                in_memory_templates[user_id] = [
                    {
                        "id": str(uuid.uuid4()),
                        "name": "Student Report",
                        "description": "Academic formatting for student papers",
                        "category": "Student"
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "name": "Content Blog",
                        "description": "Modern blog article format",
                        "category": "Content Creator"
                    }
                ]
            templates = in_memory_templates[user_id]
            
            # Filter templates based on user tier
            if user_tier == UserTier.FREE:
                allowed_categories = ["Student", "Content Creator"]
                templates = [t for t in templates if t.get("category") in allowed_categories]
                
        return templates
    
    except Exception as e:
        logger.error(f"Error fetching templates: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch templates: {str(e)}")

@app.post("/api/format")
async def format_document(
    background_tasks: BackgroundTasks,
    document_id: str = Form(...),
    template_id: str = Form(...),
    current_user = Depends(get_optional_user)
):
    try:
        # Get document info
        if app.docs_collection:
            document = app.docs_collection.find_one({"id": document_id})
        else:
            document = in_memory_documents.get(document_id)
            
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
            
        # Check user has access to this document
        if document.get("user_id") not in [current_user["user_id"], "anonymous"]:
            raise HTTPException(status_code=403, detail="You don't have access to this document")
            
        # Get template info
        if app.templates_collection:
            template = app.templates_collection.find_one({"id": template_id})
        else:
            # Search through all user templates
            template = None
            for user_templates in in_memory_templates.values():
                for t in user_templates:
                    if t.get("id") == template_id:
                        template = t
                        break
                if template:
                    break
                    
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
            
        # Format document in background
        formatted_doc_id = str(uuid.uuid4())
        background_tasks.add_task(
            format_document_task,
            document_id,
            template_id,
            formatted_doc_id,
            current_user["user_id"]
        )
        
        return {
            "formattedDocId": formatted_doc_id,
            "message": "Document formatting started",
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Format error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Formatting failed: {str(e)}")

async def format_document_task(document_id, template_id, formatted_doc_id, user_id):
    try:
        # Get document info
        if app.docs_collection:
            document = app.docs_collection.find_one({"id": document_id})
        else:
            document = in_memory_documents.get(document_id)
            
        # Get template info
        if app.templates_collection:
            template = app.templates_collection.find_one({"id": template_id})
        else:
            # Search through all user templates
            template = None
            for user_templates in in_memory_templates.values():
                for t in user_templates:
                    if t.get("id") == template_id:
                        template = t
                        break
                if template:
                    break
        
        # Define output path
        output_dir = os.path.join("outputs", formatted_doc_id)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"formatted.docx")
        
        # Apply template to document
        apply_template_to_document(
            document["upload_path"],
            template,
            output_path,
            tables=document.get("processed_data", {}).get("tables", []),
            figures=document.get("processed_data", {}).get("figures", []),
            chapters=document.get("processed_data", {}).get("chapters", [])
        )
        
        # Store formatted document info
        formatted_doc_info = {
            "id": formatted_doc_id,
            "original_document_id": document_id,
            "template_id": template_id,
            "output_path": output_path,
            "status": "completed",
            "creation_date": datetime.now(),
            "user_id": user_id
        }
        
        if app.docs_collection:
            app.docs_collection.insert_one(formatted_doc_info)
        else:
            in_memory_documents[formatted_doc_id] = formatted_doc_info
            
    except Exception as e:
        logger.error(f"Formatting error for document {document_id}: {str(e)}", exc_info=True)
        # Store error info
        formatted_doc_info = {
            "id": formatted_doc_id,
            "original_document_id": document_id,
            "template_id": template_id,
            "status": "error",
            "error": str(e),
            "creation_date": datetime.now(),
            "user_id": user_id
        }
        
        if app.docs_collection:
            app.docs_collection.insert_one(formatted_doc_info)
        else:
            in_memory_documents[formatted_doc_id] = formatted_doc_info

@app.get("/api/preview/{document_id}")
async def get_preview(document_id: str, current_user = Depends(get_optional_user)):
    try:
        # Check if document exists
        if app.docs_collection:
            document = app.docs_collection.find_one({"id": document_id})
        else:
            document = in_memory_documents.get(document_id)
            
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
            
        # Check user has access to this document
        if document.get("user_id") not in [current_user["user_id"], "anonymous"]:
            raise HTTPException(status_code=403, detail="You don't have access to this document")
            
        # Get file path
        file_path = document.get("output_path") or document.get("upload_path")
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document file not found")
            
        # For now, we'll assume HTML preview isn't supported directly
        # and just return document metadata
        return {
            "id": document_id,
            "status": document.get("status", "unknown"),
            "name": document.get("original_name", "Document"),
            "type": os.path.splitext(file_path)[1].replace(".", ""),
            "previewAvailable": document.get("status") == "completed"
        }
        
    except Exception as e:
        logger.error(f"Preview error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")

@app.get("/api/download/{document_id}")
async def download_document(document_id: str, current_user = Depends(get_optional_user)):
    try:
        # Check if document exists
        if app.docs_collection:
            document = app.docs_collection.find_one({"id": document_id})
        else:
            document = in_memory_documents.get(document_id)
            
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
            
        # Check user has access to this document
        if document.get("user_id") not in [current_user["user_id"], "anonymous"]:
            raise HTTPException(status_code=403, detail="You don't have access to this document")
            
        # Get file path
        file_path = document.get("output_path") or document.get("upload_path")
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document file not found")
        
        # Determine filename
        filename = document.get("original_name", "document")
        if document.get("status") == "completed":
            # For formatted documents, add a suffix
            name, ext = os.path.splitext(filename)
            filename = f"{name}_formatted{ext}"
            
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.delete("/api/document/{document_id}")
async def delete_document(document_id: str, current_user = Depends(get_current_user)):
    try:
        # Check if document exists
        if app.docs_collection:
            document = await app.docs_collection.find_one({"id": document_id})
        else:
            document = in_memory_documents.get(document_id)
            
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
            
        # Check user has access to this document
        if document.get("user_id") != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="You don't have access to this document")
            
        # Get file paths
        file_paths = []
        if document.get("upload_path"):
            file_paths.append(document.get("upload_path"))
        if document.get("output_path"):
            file_paths.append(document.get("output_path"))
        
        # Delete the files
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    # If file was in a directory with its ID, remove that too
                    dir_path = os.path.dirname(file_path)
                    if os.path.basename(dir_path) == document_id and not os.listdir(dir_path):
                        os.rmdir(dir_path)
            except Exception as e:
                logger.warning(f"Error deleting file {file_path}: {str(e)}")
        
        # Delete the document from the database
        if app.docs_collection:
            result = await app.docs_collection.delete_one({"id": document_id})
            if result.deleted_count == 0:
                raise HTTPException(status_code=404, detail="Document not found in database")
        else:
            if document_id in in_memory_documents:
                del in_memory_documents[document_id]
            else:
                raise HTTPException(status_code=404, detail="Document not found in memory")
            
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

# Modify the static files mounting to handle development mode better
import os

# Check if the dist directory exists, otherwise serve from a default location
frontend_dist = "../frontend/dist"
if not os.path.exists(os.path.join(os.path.dirname(__file__), frontend_dist)):
    # In development, we'll skip the static files mounting to avoid errors
    if os.environ.get("APP_ENV") != "production":
        # Just a placeholder route for non-production environments
        @app.get("/")
        async def read_root():
            return {"message": "API running. Frontend should be served separately in development mode."}
    else:
        # In production, we'll create the directory to avoid errors
        os.makedirs(os.path.join(os.path.dirname(__file__), frontend_dist), exist_ok=True)
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

# Define a Pydantic model for the subscription update request
class SubscriptionUpdateRequest(BaseModel):
    newTier: str

@app.put("/api/user/subscription")
async def update_subscription_status(
    request: SubscriptionUpdateRequest,
    current_user: dict = Depends(get_current_user) # get_current_user already fetches user from DB
):
    users_collection = app.db["users"]

    if request.newTier not in [UserTier.FREE, UserTier.PREMIUM]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid tier specified")

    user_email = current_user.get("email")

    if not user_email:
        # This should ideally be caught by get_current_user if token is invalid
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not authenticated or email not found")

    # Ensure the user exists (though get_current_user should handle this)
    user_in_db = await users_collection.find_one({"email": user_email})
    if not user_in_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    update_result = await users_collection.update_one(
        {"email": user_email},
        {"$set": {"subscription": request.newTier}}
    )

    if update_result.matched_count == 0:
        # This case should ideally not be reached if user_in_db check passed
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found during update")
    
    # If modified_count is 0, it means the subscription was already set to the newTier. This is not an error.
    # Optionally, you could add a check here: if user_in_db.get("subscription") == request.newTier: return {"message": "Subscription already set to this tier"}

    return {"message": "Subscription updated successfully", "newTier": request.newTier}

@app.get("/api/document/{document_id}/images")
async def extract_document_images(document_id: str, current_user = Depends(get_optional_user)):
    """Extract images from a document and return their information"""
    try:
        # Get document info from database
        document = await app.docs_collection.find_one({"_id": document_id})
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if user has access to this document
        if current_user and document.get("user_id") and document.get("user_id") != current_user.get("user_id"):
            raise HTTPException(status_code=403, detail="Not authorized to access this document")
        
        # Get file path
        file_path = document.get("file_path")
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document file not found")
        
        # Create output folder
        output_folder = os.path.join(os.path.dirname(file_path), f"{document_id}_images")
        
        try:
            # Import function from documorph_fixes
            try:
                from .modules.documorph_fixes import extract_images_from_pdf
            except ImportError:
                try:
                    from modules.documorph_fixes import extract_images_from_pdf
                except ImportError:
                    raise HTTPException(status_code=500, detail="Image extraction module not available")
            
            # Extract images
            images = extract_images_from_pdf(file_path, output_folder)
            
            # Update document with image info
            await app.docs_collection.update_one(
                {"_id": document_id},
                {"$set": {"images": images}}
            )
            
            return {"images": images}
            
        except Exception as e:
            logger.error(f"Error extracting images: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error extracting images: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.get("/api/document/{document_id}/tables")
async def extract_document_tables(document_id: str, current_user = Depends(get_optional_user)):
    """Extract tables from a document and return their information"""
    try:
        # Get document info from database
        document = await app.docs_collection.find_one({"_id": document_id})
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if user has access to this document
        if current_user and document.get("user_id") and document.get("user_id") != current_user.get("user_id"):
            raise HTTPException(status_code=403, detail="Not authorized to access this document")
        
        # Get file path
        file_path = document.get("file_path")
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document file not found")
        
        # Get file type
        file_type = os.path.splitext(file_path)[1].lower().replace(".", "")
        
        # Create output folder
        output_folder = os.path.join(os.path.dirname(file_path), f"{document_id}_tables")
        
        try:
            # Import functions based on file type
            if file_type == "pdf":
                try:
                    from modules.table_extraction import extract_tables
                except ImportError:
                    try:
                        from .modules.table_extraction import extract_tables
                    except ImportError:
                        raise HTTPException(status_code=500, detail="Table extraction module not available")
                
                # Extract tables from PDF
                tables = []
                for page_num in range(document.get("page_count", 1)):
                    page_tables = extract_tables(None, page_num)
                    tables.extend(page_tables)
            else:
                raise HTTPException(status_code=400, detail=f"Table extraction not supported for {file_type} files")
            
            # Process extracted tables
            try:
                from modules.documorph_fixes import process_extracted_tables
            except ImportError:
                try:
                    from .modules.documorph_fixes import process_extracted_tables
                except ImportError:
                    # Simple fallback if process_extracted_tables isn't available
                    processed_tables = [{"table_index": i, "data": str(table)} for i, table in enumerate(tables)]
            else:
                processed_tables = process_extracted_tables(tables, output_folder=output_folder)
            
            # Update document with table info
            await app.docs_collection.update_one(
                {"_id": document_id},
                {"$set": {"tables": processed_tables}}
            )
            
            return {"tables": processed_tables}
            
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error extracting tables: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document tables: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 