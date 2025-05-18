# DocuMorph AI

A powerful AI-powered document transformation and formatting application.

## Features

- **Document Upload**: Upload PDF, DOCX, and TXT files
- **Template Selection**: Choose from various document templates
- **AI-Powered Formatting**: Apply intelligent formatting to your documents
- **Premium Features**: Access advanced features like style analysis, content analysis, and table extraction
- **Developer Mode**: Test different subscription tiers during development

## Technology Stack

- **Frontend**: React (Vite), Material-UI, GSAP for animations
- **Backend**: FastAPI, Python
- **Database**: MongoDB
- **AI**: Various AI libraries for document processing

## Setup Instructions

### Prerequisites

- Node.js (v16+)
- Python (3.9+)
- MongoDB (local or cloud instance)

### Environment Variables

Create a `.env` file in the project root with the following variables:

```
# MongoDB Connection
MONGO_URI=mongodb+srv://your-mongodb-connection-string

# App Settings
APP_ENV=development
JWT_SECRET=your-jwt-secret
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=1440

# API Keys (optional based on features you want to use)
LANGCHAIN_PROJECT=
HF_TOKEN=
groq_api_key=
serpapi=
GOOGLE_CLIENT_ID=
Client_secret=
goog_api_key=
GOOGLE_REDIRECT_URI=
```

### Installation

1. Clone the repository
2. Install frontend dependencies:

```bash
cd frontend
npm install
```

3. Install backend dependencies:

```bash
cd server
pip install -r requirements.txt
```

### Development

#### Start the Backend

```bash
cd server
uvicorn main:app --reload
```

#### Start the Frontend

```bash
cd frontend
npm run dev
```

### Running with MongoDB Fallback Mode

If you're experiencing issues with MongoDB connection, you can use the MongoDB fallback mode:

```bash
# On Windows
run_with_mongodb_fallback.bat

# On Unix/Linux/Mac
bash -c "cd server && python run_server.py" &
bash -c "cd frontend && npm install && npm run dev" &
```

This mode will start the server with in-memory storage if MongoDB is unavailable, allowing you to test and use the application without a database.

### Document Analysis Tools

The project now includes specialized tools for document analysis in the `workflow` directory:

- **PDF Table Extraction**: Extract tables from PDF documents and export to Excel or CSV
- **Figure Extraction**: Extract images and figures from PDF documents
- **Document Analysis UI**: Interactive Streamlit application for document content analysis

To use these tools:

```bash
cd workflow
streamlit run document_analyzer.py
```

### Production Build

To build for production:

```bash
# Build the frontend
cd frontend
npm run build

# Start the server
cd ../server
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Deployment to Render

This application is configured for deployment on Render. Follow these steps:

1. Fork this repository to your GitHub account
2. Connect your Render account with GitHub
3. Create a new Web Service on Render pointing to your forked repo
4. Use the following settings:
   - Build Command: `./setup.sh`
   - Start Command: `cd server && uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add the required environment variables in the Render dashboard
6. Deploy the service

## Admin User

An admin user will be automatically created during the first server startup:

- **Email**: mandarak123@gmail.com
- **Password**: Mak@1944
- **Features**: Has developer mode enabled to switch between subscription tiers

## Troubleshooting

### Common Issues

#### MongoDB Connection Failures
- Ensure your MongoDB URI is correct
- Check if your IP is allowed in MongoDB Atlas settings (if using Atlas)
- Use the MongoDB fallback mode with `run_with_mongodb_fallback.bat` if needed

#### Frontend Build Issues
- Clear `node_modules` and reinstall dependencies
- Update npm to the latest version

#### Backend Server Not Starting
- Check for any missing Python dependencies
- Verify your PYTHONPATH includes the project root

## Integration Guide

### Frontend-Backend Integration

The application follows a standard client-server architecture:

1. **Backend (FastAPI)**: Provides REST API endpoints that handle document processing, user authentication, and template management.

2. **Frontend (React)**: Communicates with the backend through API calls defined in `frontend/src/services/api.js`.

3. **Integration Layer**: The `server/documorph_wrapper.py` module serves as a bridge between the API and the core document processing functionality in `backend/documorph_ai.py`.

### Adding New Features

When adding new features that require both frontend and backend components:

1. First, implement the backend functionality in `documorph_ai.py` or a suitable module in `server/modules/`.

2. Update `documorph_wrapper.py` to expose the new functionality.

3. Add an API endpoint in `server/main.py` to access the feature.

4. Finally, create the frontend components and use the API service in `frontend/src/services/api.js` to connect to the backend.

### Authentication Flow

The authentication flow uses JWT tokens:

1. User logs in via the `/api/auth/login` endpoint using username/password.
2. Server returns a JWT token.
3. Frontend stores the token in `localStorage`.
4. The token is included in the `Authorization` header for subsequent API requests.
5. Protected endpoints verify the token before providing access.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 