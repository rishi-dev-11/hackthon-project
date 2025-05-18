import os
import sys
import logging
import uvicorn
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

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

# Load environment variables
load_dotenv()

# MongoDB connection settings with fallback
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/documorph")
MONGO_TIMEOUT_MS = 5000  # 5 seconds timeout

async def test_mongodb_connection():
    """Test MongoDB connection and return True if successful, False otherwise"""
    try:
        client = AsyncIOMotorClient(
            MONGO_URI, 
            serverSelectionTimeoutMS=MONGO_TIMEOUT_MS
        )
        # Ping the database
        await client.admin.command('ping')
        logger.info("MongoDB connection test successful")
        client.close()
        return True
    except Exception as e:
        logger.error(f"MongoDB connection test failed: {str(e)}")
        return False

def modify_main_for_fallback():
    """Modify main.py to work without MongoDB if connection fails"""
    try:
        # Path to main.py
        main_path = os.path.join(os.path.dirname(__file__), "main.py")
        
        # Read main.py content
        with open(main_path, "r") as f:
            content = f.read()
        
        # Check if already modified
        if "# FALLBACK: MODIFIED FOR NO DB MODE" in content:
            logger.info("main.py already modified for fallback mode")
            return
            
        # Backup original file
        backup_path = os.path.join(os.path.dirname(__file__), "main.py.bak")
        with open(backup_path, "w") as f:
            f.write(content)
            
        # Modify the startup_db_client function to handle connection errors gracefully
        fallback_code = """
@app.on_event("startup")
async def startup_db_client():
    # FALLBACK: MODIFIED FOR NO DB MODE
    try:
        # Get MongoDB URI from environment config
        mongo_uri = ENV_CONFIG["MONGO_URI"]
        app.mongodb_client = AsyncIOMotorClient(mongo_uri, serverSelectionTimeoutMS=5000)
        app.db = app.mongodb_client.documorph
        
        # Check connection by attempting to get server info
        try:
            await app.mongodb_client.admin.command('ping')
            logger.info("Connected to MongoDB database")
            
            # Setup collections
            app.docs_collection = app.db["documents"]
            app.templates_collection = app.db["templates"]
            app.users_collection = app.db["users"]
            
            # Create admin user if needed and database is available
            try:
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
            except Exception as e:
                logger.warning(f"Could not create admin user: {str(e)}")
                
        except Exception as e:
            logger.error(f"MongoDB server ping failed: {str(e)}")
            raise RuntimeError("MongoDB connection error - falling back to in-memory mode")
            
    except Exception as e:
        logger.warning(f"MongoDB connection error: {str(e)} - Using in-memory storage fallback")
        # Set up in-memory fallback
        app.mongodb_client = None
        app.db = None
        app.docs_collection = None
        app.templates_collection = None
        app.users_collection = None
        
        # Log the fallback mode
        logger.info("Running in FALLBACK MODE with in-memory storage")
"""

        # Replace the original startup_db_client function
        import re
        modified_content = re.sub(
            r'@app.on_event\("startup"\)\nasync def startup_db_client\(\):.*?(?=@app.on_event\("shutdown"|$)',
            fallback_code,
            content,
            flags=re.DOTALL
        )
        
        # Write modified content back
        with open(main_path, "w") as f:
            f.write(modified_content)
            
        logger.info("Modified main.py for fallback mode")
        
    except Exception as e:
        logger.error(f"Failed to modify main.py: {str(e)}")

async def main():
    """Main function to test MongoDB connection and start the server"""
    # Test MongoDB connection
    mongo_available = await test_mongodb_connection()
    
    if not mongo_available:
        logger.warning("MongoDB connection failed - will modify main.py for fallback mode")
        modify_main_for_fallback()
    
    # Start the server
    config = uvicorn.Config(
        "main:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main()) 