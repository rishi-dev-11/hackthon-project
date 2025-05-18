from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get MongoDB URI
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    print("Error: MONGO_URI not found in environment variables")
    exit(1)

print(f"Testing connection to MongoDB...")
print(f"URI loaded (first part): {mongo_uri.split('@')[0]}...")

try:
    # Create client with longer timeouts
    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=30000,
        socketTimeoutMS=30000,
        appname="documorph-test"
    )
    
    # Test connection
    client.admin.command('ping')
    print("✅ MongoDB connection successful!")
    
    # Test database access
    db = client.get_database("documorph_db")
    print("✅ Database access successful!")
    
    # Test collections
    templates = db.get_collection("templates")
    documents = db.get_collection("documents")
    print("✅ Collections accessible!")
    
except Exception as e:
    print(f"❌ Connection failed: {str(e)}")
