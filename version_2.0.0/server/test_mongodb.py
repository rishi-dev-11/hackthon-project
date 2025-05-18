#!/usr/bin/env python
"""
Test MongoDB connection and create admin user
"""
import os
import sys
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from passlib.context import CryptContext
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def test_connection():
    """Test MongoDB connection"""
    print("Testing MongoDB connection...")
    
    # Get MongoDB URI from environment with fallback
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    print(f"Using connection URI: {mongo_uri}")
    
    try:
        client = AsyncIOMotorClient(mongo_uri)
        await client.admin.command('ping')
        print("✅ Connected to MongoDB successfully!")
        return client
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {str(e)}")
        return None

async def create_admin_user(client):
    """Create admin user if not exists"""
    if not client:
        print("Cannot create admin user: MongoDB connection failed")
        return False
    
    print("\nChecking if admin user exists...")
    db = client.documorph
    
    # Check if admin user exists
    admin_user = await db["users"].find_one({"email": "mandarak123@gmail.com"})
    
    if admin_user:
        print(f"Admin user already exists: {admin_user['email']}")
        
        # Update admin privileges if needed
        if not admin_user.get("isDevMode", False) or not admin_user.get("isAdmin", False):
            print("Updating admin privileges...")
            await db["users"].update_one(
                {"email": "mandarak123@gmail.com"},
                {"$set": {"isDevMode": True, "isAdmin": True, "subscription": "premium"}}
            )
            print("✅ Admin privileges updated")
        return True
    
    # Create new admin user
    print("Creating new admin user...")
    
    # Hash password
    hashed_password = pwd_context.hash("Mak@1944")
    
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
    
    try:
        result = await db["users"].insert_one(admin_user_data)
        print("✅ Admin user created successfully!")
        print(f"User ID: {admin_user_data['user_id']}")
        print(f"Email: {admin_user_data['email']}")
        print(f"Password: Mak@1944")
        return True
    except Exception as e:
        print(f"❌ Failed to create admin user: {str(e)}")
        return False

async def list_collections(client):
    """List all collections in the database"""
    if not client:
        print("Cannot list collections: MongoDB connection failed")
        return
    
    print("\nListing collections:")
    db = client.documorph
    
    try:
        collections = await db.list_collection_names()
        if collections:
            for collection in collections:
                count = await db[collection].count_documents({})
                print(f"- {collection}: {count} documents")
        else:
            print("No collections found in the database")
    except Exception as e:
        print(f"❌ Failed to list collections: {str(e)}")

async def main():
    """Main function"""
    print("\n=== MongoDB Connection Test ===\n")
    
    # Test connection
    client = await test_connection()
    if not client:
        sys.exit(1)
    
    # Create admin user
    success = await create_admin_user(client)
    
    # List collections
    await list_collections(client)
    
    print("\n=== Test Completed ===\n")
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 