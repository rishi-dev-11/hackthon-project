import os
import re
import logging
import requests
import json
import tempfile
from config import SERPAPI_KEY, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, UserTier, logger, GOOGLE_API_AVAILABLE
if GOOGLE_API_AVAILABLE:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials # For checking existing token
else: # Create dummy classes/functions if Google API is not available to prevent import errors
    InstalledAppFlow = None
    build = None
    MediaFileUpload = None
    Request = None
    Credentials = None


def check_plagiarism_serpapi(text_to_check, user_tier=UserTier.FREE):
    """Check text for plagiarism using SerpAPI."""
    if user_tier == UserTier.FREE or not UserTier.get_tier_features()[user_tier].get("plagiarism_check", False):
        return {"error": "Plagiarism checking is a Premium feature."}
    if not SERPAPI_KEY:
        return {"error": "SerpAPI key not configured."}

    try:
        # Split text into manageable chunks (e.g., sentences or short paragraphs)
        # For SerpAPI, searching very long exact strings is often not effective.
        # Better to search for key phrases or representative sentences.
        # This is a simplified approach: taking first few sentences as queries.
        sentences = re.split(r'(?<=[.!?])\s+', text_to_check.strip())
        queries_to_check = sentences[:5] # Check up to 5 representative sentences/phrases
        if not queries_to_check and text_to_check: # If no sentences, use the whole text if short
            queries_to_check = [text_to_check[:300]]


        all_results_summary = []
        plagiarized_segments_count = 0
        total_segments_checked = len(queries_to_check)

        for i, query_text in enumerate(queries_to_check):
            if not query_text.strip(): continue

            params = {
                "engine": "google",
                "q": f'"{query_text}"',  # Exact match for the segment
                "api_key": SERPAPI_KEY
            }
            response = requests.get("https://serpapi.com/search", params=params, timeout=10)
            response.raise_for_status() # Will raise an HTTPError for bad responses (4XX or 5XX)
            data = response.json()
            
            organic_results = data.get("organic_results", [])
            segment_matches = []
            has_exact_match_in_snippet = False

            # Process results - look for very similar snippets
            # This is a heuristic. True plagiarism detection is more complex.
            for res in organic_results[:3]: # Check top 3 results
                snippet = res.get("snippet", "").lower()
                # A very basic check: if a significant portion of our query appears in the snippet
                # This is not robust plagiarism detection.
                # More advanced would be fuzzy matching, comparing document structures etc.
                # For now, if any organic result is found for an exact query, we flag it.
                segment_matches.append({
                    "title": res.get("title", "No Title"),
                    "link": res.get("link", "#"),
                    "snippet": res.get("snippet", "No Snippet")
                })
                if query_text.lower() in snippet: # Simple check
                    has_exact_match_in_snippet = True
            
            if segment_matches: # If any results were found for the exact query
                plagiarized_segments_count += 1
            
            all_results_summary.append({
                "chunk_index": i,
                "text": query_text,
                "matches": segment_matches,
                "has_matches": bool(segment_matches), # True if any search result found
            })
        
        plagiarism_percentage = (plagiarized_segments_count / total_segments_checked * 100) if total_segments_checked > 0 else 0
        
        return {
            "has_plagiarism": plagiarized_segments_count > 0,
            "plagiarism_percentage": round(plagiarism_percentage, 2),
            "chunks_checked": total_segments_checked, # Number of queries sent
            "total_chunks": len(sentences), # Total sentences (approx segments)
            "results": all_results_summary
        }

    except requests.exceptions.RequestException as e_req:
        logger.error(f"SerpAPI request failed: {e_req}")
        return {"error": f"Plagiarism check API request failed: {e_req}"}
    except Exception as e:
        logger.error(f"Error checking plagiarism: {e}", exc_info=True)
        return {"error": f"Error during plagiarism check: {str(e)}"}


def export_to_google_drive(file_path, file_name, user_tier=UserTier.FREE):
    if not GOOGLE_API_AVAILABLE:
         return {"success": False, "error": "Google API libraries not installed."}
    if user_tier == UserTier.FREE or not UserTier.get_tier_features()[user_tier].get("google_drive_export", False):
        return {"success": False, "error": "Google Drive export is a Premium feature."}
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return {"success": False, "error": "Google Client ID or Secret not configured in .env file."}

    try:
        creds_dict = {
            "installed": {
                "client_id": GOOGLE_CLIENT_ID,
                "project_id": "documorph-ai", # You might need to set this in your Google Cloud project
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost:8501"] # Add Streamlit default port
            }
        }
        # Using a temporary file for client_config is more robust for InstalledAppFlow
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_cred_file:
            json.dump(creds_dict, temp_cred_file)
            credentials_path_temp = temp_cred_file.name
        
        flow = InstalledAppFlow.from_client_secrets_file(
            credentials_path_temp,
            scopes=['https://www.googleapis.com/auth/drive.file'],
            redirect_uri='urn:ietf:wg:oauth:2.0:oob' # For copy-paste code
        )
        auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline') # offline for refresh token
        
        # Store credentials_path_temp to be used in complete_google_drive_export
        return {
            "success": False, "auth_url": auth_url,
            "message": "Please authorize DocuMorph AI to access your Google Drive.",
            "credentials_path": credentials_path_temp, # Pass path to temp secrets file
            "file_path": file_path, "file_name": file_name
        }
    except Exception as e:
        logger.error(f"Error setting up Google Drive export: {e}", exc_info=True)
        if 'credentials_path_temp' in locals() and os.path.exists(credentials_path_temp):
             os.unlink(credentials_path_temp)
        return {"success": False, "error": f"Google Drive setup error: {str(e)}"}


def complete_google_drive_export(auth_code, credentials_path_temp, file_path, file_name):
    if not GOOGLE_API_AVAILABLE:
         return {"success": False, "error": "Google API libraries not installed."}
    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            credentials_path_temp, # Use the path to the temporary secrets file
            scopes=['https://www.googleapis.com/auth/drive.file'],
            redirect_uri='urn:ietf:wg:oauth:2.0:oob'
        )
        flow.fetch_token(code=auth_code)
        creds = flow.credentials

        # Optional: Save credentials for future use (e.g., in session or secure storage)
        # For Streamlit, storing in st.session_state (after serializing) is an option,
        # but be mindful of security if deploying. A token.json file is common for local scripts.
        # For this app, we are re-authenticating each time for simplicity.

        drive_service = build('drive', 'v3', credentials=creds)
        
        # Determine MIME type
        if file_path.endswith('.docx'):
            mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        elif file_path.endswith('.pdf'):
            mime_type = 'application/pdf'
        elif file_path.endswith('.txt'):
            mime_type = 'text/plain'
        else:
            mime_type = 'application/octet-stream'

        file_metadata = {'name': file_name}
        media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
        
        gfile = drive_service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute()
        
        if os.path.exists(credentials_path_temp): # Clean up the temporary secrets file
            os.unlink(credentials_path_temp)
            
        return {"success": True, "file_id": gfile.get('id'), "web_link": gfile.get('webViewLink')}
    except Exception as e:
        logger.error(f"Error completing Google Drive export: {e}", exc_info=True)
        if 'credentials_path_temp' in locals() and os.path.exists(credentials_path_temp):
             os.unlink(credentials_path_temp)
        return {"success": False, "error": f"Google Drive export completion error: {str(e)}"}