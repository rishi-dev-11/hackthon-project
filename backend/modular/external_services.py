import os
import logging
import requests
import json
import tempfile
from config import SERPAPI_KEY, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, UserTier, logger, GOOGLE_API_AVAILABLE, BASE_DIR # Added BASE_DIR

if GOOGLE_API_AVAILABLE:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    # from google.auth.transport.requests import Request # Not strictly needed for this flow
    # from google.oauth2.credentials import Credentials # Not strictly needed for this flow
else:
    InstalledAppFlow = None
    build = None
    MediaFileUpload = None

# Plagiarism Check (using SerpAPI)
def check_plagiarism_serpapi(text_to_check, user_tier_val=UserTier.FREE):
    # (Ensure this function uses user_tier_val correctly as passed)
    if not UserTier.get_tier_features()[user_tier_val].get("plagiarism_check", False): # More robust check
        return {"error": "Plagiarism checking is a Premium feature."}
    if not SERPAPI_KEY:
        return {"error": "SerpAPI key not configured."}
    # (Rest of the plagiarism check logic from your previous version)
    try:
        sentences = re.split(r'(?<=[.!?])\s+', text_to_check.strip())
        queries_to_check = [s for s in sentences if len(s.strip()) > 15][:5] # Meaningful sentences
        if not queries_to_check and text_to_check:
            queries_to_check = [text_to_check[:300]]

        all_results_summary = []
        plagiarized_segments_count = 0
        total_segments_checked = len(queries_to_check)
        if total_segments_checked == 0:
            return {"error": "No suitable text segments found for plagiarism check."}


        for i, query_text in enumerate(queries_to_check):
            if not query_text.strip(): continue
            params = {"engine": "google", "q": f'"{query_text}"', "api_key": SERPAPI_KEY}
            response = requests.get("https://serpapi.com/search", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            organic_results = data.get("organic_results", [])
            segment_matches = []
            
            for res in organic_results[:3]:
                segment_matches.append({
                    "title": res.get("title", "N/A"), "link": res.get("link", "#"),
                    "snippet": res.get("snippet", "N/A")
                })
            if segment_matches: plagiarized_segments_count += 1
            all_results_summary.append({
                "chunk_index": i, "text": query_text, "matches": segment_matches,
                "has_matches": bool(segment_matches),
            })
        
        plagiarism_percentage = (plagiarized_segments_count / total_segments_checked * 100) if total_segments_checked > 0 else 0
        return {
            "has_plagiarism": plagiarized_segments_count > 0,
            "plagiarism_percentage": round(plagiarism_percentage, 2),
            "chunks_checked": total_segments_checked,
            "total_chunks": len(sentences),
            "results": all_results_summary
        }
    except requests.exceptions.RequestException as e_req:
        logger.error(f"SerpAPI request failed: {e_req}")
        return {"error": f"Plagiarism check API request failed: {e_req}"}
    except Exception as e:
        logger.error(f"Error checking plagiarism: {e}", exc_info=True)
        return {"error": f"Error during plagiarism check: {str(e)}"}


# Google Drive Export
def export_to_google_drive(file_to_export_path, export_as_filename, user_tier_val=UserTier.FREE):
    if not GOOGLE_API_AVAILABLE:
         return {"success": False, "error": "Google API libraries not installed."}
    if not UserTier.get_tier_features()[user_tier_val].get("google_drive_export", False):
        return {"success": False, "error": "Google Drive export is a Premium feature."}
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return {"success": False, "error": "Google Client ID or Secret not configured."}

    try:
        # Construct client_config directly
        client_config = {
            "installed": {
                "client_id": GOOGLE_CLIENT_ID,
                "project_id": "documorph-ai", # This should match your GCP project ID
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost:8501"]
            }
        }
        
        # No need to save client_config to a temp file if InstalledAppFlow.from_client_config is used
        flow = InstalledAppFlow.from_client_config(
            client_config=client_config, # Pass the dict directly
            scopes=['https://www.googleapis.com/auth/drive.file'],
            # redirect_uri='urn:ietf:wg:oauth:2.0:oob' # This is used if you want the code displayed
        )
        # For Streamlit, running a local server for redirect is better for user experience
        # However, this requires http://localhost:PORT to be in redirect_uris in GCP console
        # and your app to listen on that port.
        # For simplicity with InstalledAppFlow, 'oob' is easier if user can copy-paste.
        # If you want to try the local server redirect:
        # creds = flow.run_local_server(port=0) # This will open browser, user auths, redirects to temp local server
        # This approach is better than making user copy-paste a code if it works reliably.
        # For "oob" (copy-paste code):
        auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
        
        return {
            "success": False, "auth_url": auth_url,
            "message": "Please authorize DocuMorph AI to access your Google Drive.",
            "flow": flow, # Pass the flow object to use for fetching token later
            "file_path": file_to_export_path, "file_name": export_as_filename
        }

    except Exception as e:
        logger.error(f"Error setting up Google Drive export: {e}", exc_info=True)
        return {"success": False, "error": f"Google Drive setup error: {str(e)}"}


def complete_google_drive_export(flow_obj, auth_code_from_user, file_to_export_path, export_as_filename):
    if not GOOGLE_API_AVAILABLE:
         return {"success": False, "error": "Google API libraries not installed."}
    try:
        flow_obj.fetch_token(code=auth_code_from_user)
        creds = flow_obj.credentials

        drive_service = build('drive', 'v3', credentials=creds)
        
        mime_type_map = {
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
        }
        file_ext = os.path.splitext(file_to_export_path)[1].lower()
        mime_type = mime_type_map.get(file_ext, 'application/octet-stream')

        file_metadata = {'name': export_as_filename} # Use the desired export filename
        media = MediaFileUpload(file_to_export_path, mimetype=mime_type, resumable=True)
        
        gfile = drive_service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute()
            
        return {"success": True, "file_id": gfile.get('id'), "web_link": gfile.get('webViewLink')}
    except Exception as e:
        logger.error(f"Error completing Google Drive export: {e}", exc_info=True)
        return {"success": False, "error": f"Google Drive export completion error: {str(e)}"}