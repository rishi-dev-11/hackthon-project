import streamlit as st
import os
import uuid
import logging
import tempfile
import json
import re

# Import from our modularized files
from config import (
    UserTier, PERSONA_CATEGORIES, SERPAPI_KEY, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET,
    logger, GOOGLE_API_AVAILABLE, UNSTRUCTURED_AVAILABLE, REPORTLAB_AVAILABLE, WEASYPRINT_AVAILABLE,
    DOCUMENTS_DIR, OUTPUT_DIR, TEMP_DIR
)
from database import init_mongodb, initialize_default_formatting_templates, get_available_formatting_templates
# from vector_store_utils import init_vector_store # Not used in current feature set

from nlp_utils import init_nlp_models, detect_language, translate_text
from llm_services import (
    init_llm, extract_document_structure_llm, generate_abstract_llm,
    suggest_section_titles_llm, enhance_text_style_llm, get_word_suggestions_llm,
    analyze_document_chapters_llm,
    rule_based_structure_extraction, rule_based_chapter_detection,
    rule_based_abstract_generation, rule_based_word_suggestions
)
from document_processing import (
    process_document_master,
    detect_tables_from_pdf, detect_tables_from_docx, # These call the updated logic
    detect_figures_from_pdf, detect_figures_from_docx, # These call the updated logic
    extract_and_zip_images_from_pdf # New image export utility
)
from template_application import (
    apply_template_to_document, # For DOCX
    assign_figure_table_numbers,
    generate_formatted_pdf_reportlab, # For PDF
    generate_formatted_pdf_weasyprint # Alternative PDF
)
from external_apis import check_plagiarism_serpapi, export_to_google_drive, complete_google_drive_export
from ui_helpers import display_document_preview, get_persona_specific_prompt, can_use_template_role


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="DocuMorph AI", layout="wide", initial_sidebar_state="expanded")

# --- Helper: Check Feature Access ---
def can_use_llm_for_persona(user_tier_val):
    return UserTier.get_tier_features()[user_tier_val]["llm_persona_guidance"]

def can_use_advanced_llm_features(user_tier_val):
     return UserTier.get_tier_features()[user_tier_val]["llm_advanced_features"]

def can_use_premium_feature(user_tier_val, feature_key):
    return UserTier.get_tier_features()[user_tier_val].get(feature_key, False)


# --- Main Application ---
def main():
    st.title("✨ DocuMorph AI - Intelligent Document Transformation Engine")
    st.markdown("Transform your documents with AI-powered formatting and content enhancement.")

    # --- Initialize Session State ---
    if 'user_id' not in st.session_state: st.session_state.user_id = str(uuid.uuid4())
    if 'user_tier' not in st.session_state: st.session_state.user_tier = UserTier.FREE
    if 'current_persona' not in st.session_state: st.session_state.current_persona = "General/Others"
    
    # Document related
    if 'active_doc_info' not in st.session_state: st.session_state.active_doc_info = None # {doc_id, original_name, processed_temp_path, file_type, ...}
    if 'active_doc_content_chunks' not in st.session_state: st.session_state.active_doc_content_chunks = [] # Langchain Document objects
    if 'active_doc_live_text' not in st.session_state: st.session_state.active_doc_live_text = "" # For editor

    # Extracted elements
    if 'extracted_structure_text' not in st.session_state: st.session_state.extracted_structure_text = ""
    if 'extracted_tables_list' not in st.session_state: st.session_state.extracted_tables_list = []
    if 'extracted_figures_list' not in st.session_state: st.session_state.extracted_figures_list = []
    if 'analyzed_chapters_list' not in st.session_state: st.session_state.analyzed_chapters_list = []
    
    # Formatting Template related
    if 'available_formatting_templates' not in st.session_state: st.session_state.available_formatting_templates = []
    if 'active_formatting_template' not in st.session_state: st.session_state.active_formatting_template = None # The chosen dict

    # Output state
    if 'formatted_docx_path' not in st.session_state: st.session_state.formatted_docx_path = None
    if 'formatted_pdf_path' not in st.session_state: st.session_state.formatted_pdf_path = None
    
    # Misc state
    if 'word_suggestions' not in st.session_state: st.session_state.word_suggestions = []
    if 'google_auth_flow_details' not in st.session_state: st.session_state.google_auth_flow_details = None


    # --- Initializations ---
    # MongoDB
    db_docs, db_templates, db_users = init_mongodb()
    if db_templates is not None: # Only if DB connection is successful
        initialize_default_formatting_templates(db_templates, user_id_to_check="system_default")
    
    # Load available formatting templates (system defaults + user's custom ones)
    # This should be refreshed if templates are added/deleted
    if not st.session_state.available_formatting_templates or 'refresh_templates' in st.session_state:
        st.session_state.available_formatting_templates = get_available_formatting_templates(
            db_templates, st.session_state.user_id, st.session_state.user_tier
        )
        if 'refresh_templates' in st.session_state:
            del st.session_state['refresh_templates']


    # NLP Models (spaCy, and prepares for translator)
    nlp_models = init_nlp_models()
    # LLM (Groq)
    llm = init_llm() if can_use_llm_for_persona(st.session_state.user_tier) or can_use_advanced_llm_features(st.session_state.user_tier) else None
    if (can_use_llm_for_persona(st.session_state.user_tier) or can_use_advanced_llm_features(st.session_state.user_tier)) and not llm:
        st.sidebar.error("LLM (Groq) could not be initialized. AI features will be limited.")


    # --- Sidebar ---
    with st.sidebar:
        st.image("logo.png", width=100) # Add a placeholder for your logo
        st.header("👤 User Settings")
        
        # Tier Selection
        current_tier_index = [UserTier.FREE, UserTier.PREMIUM].index(st.session_state.user_tier)
        tier_selection = st.radio("Select Tier:", (UserTier.FREE, UserTier.PREMIUM),
                                  format_func=lambda x: x.capitalize(), index=current_tier_index, key="tier_selector")
        if tier_selection != st.session_state.user_tier:
            st.session_state.user_tier = tier_selection
            st.success(f"Switched to {tier_selection.capitalize()} tier.")
            st.rerun()

        # Persona Selection (for LLM guidance)
        if can_use_llm_for_persona(st.session_state.user_tier):
            st.header("🤖 AI Persona")
            current_persona_index = PERSONA_CATEGORIES.index(st.session_state.current_persona) \
                                    if st.session_state.current_persona in PERSONA_CATEGORIES else \
                                    PERSONA_CATEGORIES.index("General/Others")
            selected_persona = st.selectbox("Guide AI with Persona:", PERSONA_CATEGORIES, index=current_persona_index, key="persona_selector")
            if selected_persona != st.session_state.current_persona:
                st.session_state.current_persona = selected_persona
                st.success(f"AI Persona set to: {selected_persona}")
        else:
            st.markdown("Select **Premium Tier** to unlock AI Personas for content enhancement.")
            st.session_state.current_persona = "General/Others" # Default for free tier

        st.markdown("---")
        st.header("🛠️ Application Features")
        
        feature_list = [
            "📄 Document Processing Hub",
            "🎨 Formatting Templates",
            # "✨ Custom Formatting Templates", # Combined into Formatting Templates
            "💡 AI Content Enhancement",
            "📊 Tables & Figures Manager",
            "✍️ Live Text Editor",
            "🚀 Preview & Export",
            "🖼️ Export All Images from PDF"
        ]
        if can_use_premium_feature(st.session_state.user_tier, "plagiarism_check"):
            feature_list.append("🔍 Plagiarism Check (Premium)")
        if can_use_premium_feature(st.session_state.user_tier, "google_drive_export"):
            feature_list.append("🔗 Export to Google Drive (Premium)")

        active_feature = st.radio("Navigate:", feature_list, key="feature_selector")

    # --- Main Page Content Based on Feature Selection ---

    if active_feature == "📄 Document Processing Hub":
        st.header("📄 Document Processing Hub")
        st.markdown("Upload your document to begin. We'll extract content, tables, and figures.")

        # Document Upload
        # Add logic for document limits based on tier
        # docs_count_for_user = db_docs.count_documents({"user_id": st.session_state.user_id}) if db_docs else 0
        # max_docs_allowed = UserTier.get_tier_features()[st.session_state.user_tier]["max_documents_processed"]
        # if docs_count_for_user >= max_docs_allowed:
        #     st.warning(f"You have reached the document limit ({max_docs_allowed}) for the {st.session_state.user_tier} tier.")
        #     uploaded_doc_file = None
        # else:
        uploaded_doc_file = st.file_uploader("Upload Document (DOCX, PDF, TXT, Images)",
                                             type=["docx", "pdf", "txt", "png", "jpg", "jpeg", "tiff"],
                                             key="main_doc_uploader")
        
        # Information about handwritten text support
        if st.session_state.user_tier == UserTier.PREMIUM:
            st.info("✨ **Premium Feature**: Enhanced OCR with multi-language handwritten text recognition for images. Supports rotated text, poor lighting conditions, and multiple languages.", icon="✍️")
            with st.expander("Tips for handwritten content"):
                st.markdown("""
                **Tips for better handwritten text recognition:**
                - Ensure good lighting when taking photos of handwritten notes
                - Keep the document flat to avoid distortion
                - For best results, use dark ink on light background
                - Multiple languages are supported (10+ languages for Premium users)
                - The system can handle different handwriting styles and some rotation
                """)
        else:
            st.info("Basic handwritten text recognition available for images. Upgrade to Premium for multi-language support and enhanced accuracy.", icon="✍️")
        
        if uploaded_doc_file:
            if st.button("Process Uploaded Document", key="process_doc_btn"):
                with st.spinner("Analyzing document... Please wait."):
                    file_ext = os.path.splitext(uploaded_doc_file.name)[1].lower().strip('.')
                    
                    # Save to a temporary file for robust processing by different libraries
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}", dir=TEMP_DIR) as tmp_f:
                        tmp_f.write(uploaded_doc_file.getvalue())
                        temp_doc_path = tmp_f.name
                    
                    st.session_state.active_doc_info = {
                        "doc_id": str(uuid.uuid4()), "user_id": st.session_state.user_id,
                        "original_name": uploaded_doc_file.name,
                        "processed_temp_path": temp_doc_path, # Store path to temp file
                        "file_type": file_ext, "upload_time": str(uuid.uuid4()) # Placeholder timestamp
                    }
                    logger.info(f"Processing document: {uploaded_doc_file.name} from temp path {temp_doc_path}")

                    # Master processing function
                    chunks, tables, figures, error = process_document_master(
                        temp_doc_path, file_ext, st.session_state.user_tier
                    )

                    if error:
                        st.error(f"Error processing document: {error}")
                        st.session_state.active_doc_info = None # Clear if processing failed
                    elif not chunks:
                        st.warning("No content could be extracted from the document.")
                        st.session_state.active_doc_info = None
                    else:
                        st.success(f"'{uploaded_doc_file.name}' processed successfully!")
                        st.session_state.active_doc_content_chunks = chunks
                        st.session_state.extracted_tables_list = tables if tables else []
                        st.session_state.extracted_figures_list = figures if figures else []
                        
                        # Derive live editor text from chunks
                        st.session_state.active_doc_live_text = "\n\n".join([c.page_content for c in chunks])

                        # Structure and Chapter Analysis (LLM or Rule-based)
                        if llm and can_use_llm_for_persona(st.session_state.user_tier):
                            st.session_state.extracted_structure_text = extract_document_structure_llm(chunks, llm, st.session_state.current_persona)
                            st.session_state.analyzed_chapters_list = analyze_document_chapters_llm(st.session_state.extracted_structure_text, llm, st.session_state.current_persona)
                        else:
                            st.session_state.extracted_structure_text = rule_based_structure_extraction(chunks)
                            st.session_state.analyzed_chapters_list = rule_based_chapter_detection(chunks)
                        
                        st.subheader("Initial Analysis:")
                        with st.expander("View Extracted Document Structure", expanded=False):
                            st.text_area("Structure", st.session_state.extracted_structure_text, height=200, disabled=True, key="struct_view")
                        st.info(f"Found: {len(st.session_state.extracted_tables_list)} Tables, {len(st.session_state.extracted_figures_list)} Figures/Charts, {len(st.session_state.analyzed_chapters_list)} Chapters (approx).")
                        
                        # Clean up temp file if it's no longer the active source, or manage its lifecycle
                        # For now, `processed_temp_path` is kept for other features to use.
                        # Proper temp file management is important for long-running apps.
        
        # Display active document info
        if st.session_state.active_doc_info:
            st.markdown("---")
            st.subheader(f"Active Document: `{st.session_state.active_doc_info['original_name']}`")
            if st.button("Clear Active Document", key="clear_active_doc"):
                # Clean up the temporary file associated with the active document
                if st.session_state.active_doc_info.get("processed_temp_path") and \
                   os.path.exists(st.session_state.active_doc_info["processed_temp_path"]):
                    try:
                        os.unlink(st.session_state.active_doc_info["processed_temp_path"])
                        logger.info(f"Deleted temp file: {st.session_state.active_doc_info['processed_temp_path']}")
                    except Exception as e_del:
                        logger.error(f"Error deleting temp file {st.session_state.active_doc_info['processed_temp_path']}: {e_del}")
                
                st.session_state.active_doc_info = None
                st.session_state.active_doc_content_chunks = []
                st.session_state.active_doc_live_text = ""
                st.session_state.extracted_structure_text = ""
                st.session_state.extracted_tables_list = []
                st.session_state.extracted_figures_list = []
                st.session_state.analyzed_chapters_list = []
                st.session_state.formatted_docx_path = None
                st.session_state.formatted_pdf_path = None
                st.success("Active document cleared.")
                st.rerun()


    elif active_feature == "🎨 Formatting Templates":
        st.header("🎨 Formatting Templates")
        st.markdown("Select a pre-defined formatting template or create your own custom styles.")

        # Display available templates (system + user's custom)
        if not st.session_state.available_formatting_templates:
            st.info("No formatting templates found. System defaults might be initializing or DB is unavailable.")
        else:
            template_names = [t.get("name", "Unnamed Template") for t in st.session_state.available_formatting_templates]
            
            # Determine current selection index
            current_selection_index = 0
            if st.session_state.active_formatting_template:
                try:
                    current_selection_index = template_names.index(st.session_state.active_formatting_template.get("name"))
                except ValueError: # If active template name not in current list
                    st.session_state.active_formatting_template = None # Reset

            selected_template_name = st.selectbox(
                "Choose a Formatting Template:",
                options=template_names,
                index=current_selection_index,
                key="template_selector_main"
            )

            if selected_template_name:
                chosen_template = next((t for t in st.session_state.available_formatting_templates if t.get("name") == selected_template_name), None)
                if chosen_template and chosen_template != st.session_state.active_formatting_template:
                    st.session_state.active_formatting_template = chosen_template
                    st.success(f"Formatting Template '{chosen_template.get('name')}' selected.")
                    # st.rerun() # Rerun can be disruptive, update UI elements as needed directly

            if st.session_state.active_formatting_template:
                st.subheader(f"Selected Template: `{st.session_state.active_formatting_template.get('name')}`")
                with st.expander("View Template Details", expanded=False):
                    # Filter out MongoDB's _id for cleaner display if it exists
                    display_template = {k: v for k, v in st.session_state.active_formatting_template.items() if k != '_id'}
                    st.json(display_template)
            else:
                st.info("No formatting template currently selected.")

        # Link to create custom templates
        st.markdown("---")
        st.markdown("Want more control? You can also **create your own custom formatting templates!**")
        if st.button("Go to Custom Template Creator", key="goto_custom_templates_btn"):
            # This ideally would switch the radio button in the sidebar.
            # For simplicity, you might instruct user or use more complex state to switch active_feature.
            st.info("Please select 'Custom Templates' from the sidebar navigation to create your own.")

    # ... (Implementation for "Custom Templates" feature - involves forms for detailed template creation) ...
    # ... (Implementation for "AI Content Enhancement" - using persona and LLM) ...
    # ... (Implementation for "Tables & Figures Manager" - display, edit captions, numbering) ...
    # ... (Implementation for "Live Text Editor" - with persona-based word suggestions) ...

    elif active_feature == "🚀 Preview & Export":
        st.header("🚀 Preview & Export Document")
        if not st.session_state.active_doc_info:
            st.warning("Please upload and process a document first.")
            return
        if not st.session_state.active_formatting_template:
            st.warning("Please select a Formatting Template first.")
            return

        st.info(f"Document: **{st.session_state.active_doc_info['original_name']}** | Persona for AI: **{st.session_state.current_persona}** | Formatting Template: **{st.session_state.active_formatting_template.get('name')}**")

        # Option to use AI to suggest a document title based on content and persona
        doc_title_final = st.session_state.active_doc_info['original_name']
        if llm and can_use_advanced_llm_features(st.session_state.user_tier):
            if st.checkbox("Use AI to suggest a document title?", key="ai_title_suggest_cb"):
                with st.spinner("AI is thinking of a title..."):
                    # Simplified: Use abstract generation prompt but ask for a title
                    title_prompt = f"Based on the following content and the persona of a '{st.session_state.current_persona}', suggest a suitable document title. Content: {' '.join([c.page_content for c in st.session_state.active_doc_content_chunks[:2]])[:1000]}"
                    title_suggestion = llm.invoke(title_prompt).content.strip().split('\n')[0] # Take first line
                    doc_title_final = st.text_input("Suggested Document Title:", value=title_suggestion, key="ai_doc_title")
        
        # Option to generate/include an abstract
        generated_abstract = None
        if st.checkbox("Include an Abstract?", key="include_abstract_cb"):
            if llm and can_use_advanced_llm_features(st.session_state.user_tier):
                if st.button("Generate Abstract with AI", key="gen_abs_btn_export"):
                    with st.spinner("AI generating abstract..."):
                        generated_abstract = generate_abstract_llm(st.session_state.active_doc_content_chunks, llm, st.session_state.current_persona)
                        st.session_state.generated_abstract_for_export = generated_abstract # Store for export
                if "generated_abstract_for_export" in st.session_state and st.session_state.generated_abstract_for_export:
                    generated_abstract = st.session_state.generated_abstract_for_export
                    st.text_area("Generated Abstract (will be included):", generated_abstract, height=150, disabled=True, key="abs_view_export")
            else: # Rule-based for free tier
                generated_abstract = rule_based_abstract_generation(st.session_state.active_doc_content_chunks)
                st.text_area("Basic Abstract (will be included):", generated_abstract, height=150, disabled=True, key="abs_view_export_free")


        export_format = st.radio("Select Export Format:", ("DOCX", "PDF"), key="export_format_radio")

        # "Auto-Format" button (Premium)
        if st.session_state.user_tier == UserTier.PREMIUM:
            if st.button("✨ Auto-Format with Persona & Preview", key="auto_format_persona_btn"):
                with st.spinner(f"Auto-formatting based on '{st.session_state.current_persona}' persona..."):
                    # Find a default template associated with the current persona
                    # This requires default templates to have a 'persona_category' field.
                    default_template_for_persona = next(
                        (t for t in st.session_state.available_formatting_templates 
                         if not t.get("is_custom") and t.get("persona_category") == st.session_state.current_persona),
                        st.session_state.available_formatting_templates[0] if st.session_state.available_formatting_templates else None # Fallback
                    )
                    if default_template_for_persona:
                        st.session_state.active_formatting_template = default_template_for_persona
                        st.info(f"Applied default template '{default_template_for_persona.get('name')}' for '{st.session_state.current_persona}' persona.")
                        # Now, trigger the standard apply and preview logic (which will run below if button is clicked)
                    else:
                        st.warning(f"No default formatting template found for '{st.session_state.current_persona}' persona. Please select a template manually.")
                        # Do not proceed with formatting if no template is found


        if st.button(f"Apply Template & Generate {export_format}", key="apply_and_export_btn"):
            if not st.session_state.active_formatting_template:
                st.error("No formatting template selected. Please choose one from 'Formatting Templates'.")
            else:
                with st.spinner(f"Applying template and generating {export_format}..."):
                    # Use live editor text if available, otherwise fall back to processed chunks
                    source_text_for_processing = st.session_state.active_doc_live_text \
                        if st.session_state.active_doc_live_text.strip() \
                        else "\n\n".join([c.page_content for c in st.session_state.active_doc_content_chunks])

                    # Create a source docx.Document object from the text
                    source_docx_for_template = docx.Document()
                    for para_text in source_text_for_processing.split('\n\n'): # Simple paragraph split
                        source_docx_for_template.add_paragraph(para_text.strip())
                    
                    # Prepare output paths
                    base_output_filename = os.path.splitext(st.session_state.active_doc_info['original_name'])[0]
                    user_output_dir = os.path.join(OUTPUT_DIR, st.session_state.user_id)
                    os.makedirs(user_output_dir, exist_ok=True)

                    final_docx_path = os.path.join(user_output_dir, f"formatted_{base_output_filename}_{uuid.uuid4().hex[:4]}.docx")
                    
                    # Always generate DOCX first, as it can be a source for PDF generation
                    success_docx, docx_path_or_msg = apply_template_to_document(
                        source_doc_obj=source_docx_for_template,
                        template_config=st.session_state.active_formatting_template,
                        output_docx_path=final_docx_path,
                        tables_data=st.session_state.extracted_tables_list,
                        figures_data=st.session_state.extracted_figures_list,
                        chapters_data=st.session_state.analyzed_chapters_list,
                        document_title_override=doc_title_final,
                        abstract_text=generated_abstract
                    )

                    if success_docx:
                        st.session_state.formatted_docx_path = docx_path_or_msg
                        st.success(f"DOCX generated: {os.path.basename(docx_path_or_msg)}")
                        display_document_preview(docx_path_or_msg) # Show DOCX preview

                        if export_format == "PDF":
                            final_pdf_path = os.path.join(user_output_dir, f"formatted_{base_output_filename}_{uuid.uuid4().hex[:4]}.pdf")
                            pdf_quality = UserTier.get_tier_features()[st.session_state.user_tier]["pdf_export_quality"]
                            
                            st.info(f"Generating PDF ({pdf_quality} quality)...")
                            if pdf_quality == "enhanced" and WEASYPRINT_AVAILABLE:
                                # Convert DOCX to HTML (this is the hard part, needs a good library or complex logic)
                                # For now, we'll use a placeholder or a very simple HTML conversion
                                # Option: use pandoc if available `os.system(f"pandoc {docx_path_or_msg} -o temp.html")`
                                # Or, build HTML string from the python-docx object (complex for good styling)
                                # For this example, we'll generate PDF from the source_docx_for_template using ReportLab
                                # which will re-apply styles. True high-fidelity DOCX -> Styled PDF is hard.
                                st.warning("Enhanced PDF (WeasyPrint from styled HTML) generation from DOCX is complex and not fully implemented here. Using ReportLab.")
                                success_pdf, pdf_path_or_msg_rl = generate_formatted_pdf_reportlab(
                                    source_doc_obj=docx.Document(docx_path_or_msg), # Read the generated DOCX
                                    template_config=st.session_state.active_formatting_template,
                                    output_pdf_path=final_pdf_path,
                                    tables_data=st.session_state.extracted_tables_list, # Pass data again
                                    figures_data=st.session_state.extracted_figures_list,
                                    chapters_data=st.session_state.analyzed_chapters_list,
                                    document_title_override=doc_title_final,
                                    abstract_text=generated_abstract
                                )

                            else: # Basic PDF with ReportLab
                                 success_pdf, pdf_path_or_msg_rl = generate_formatted_pdf_reportlab(
                                    source_doc_obj=docx.Document(docx_path_or_msg), # Read the generated DOCX
                                    template_config=st.session_state.active_formatting_template,
                                    output_pdf_path=final_pdf_path,
                                    tables_data=st.session_state.extracted_tables_list,
                                    figures_data=st.session_state.extracted_figures_list,
                                    chapters_data=st.session_state.analyzed_chapters_list,
                                    document_title_override=doc_title_final,
                                    abstract_text=generated_abstract
                                )
                            
                            if success_pdf:
                                st.session_state.formatted_pdf_path = pdf_path_or_msg_rl
                                st.success(f"PDF generated: {os.path.basename(pdf_path_or_msg_rl)}")
                                display_document_preview(pdf_path_or_msg_rl)
                            else:
                                st.error(f"Failed to generate PDF: {pdf_path_or_msg_rl}")
                    else:
                        st.error(f"Failed to apply template for DOCX: {docx_path_or_msg}")

        # Display download buttons if files were generated
        if st.session_state.formatted_docx_path and os.path.exists(st.session_state.formatted_docx_path):
            st.markdown("---") # Separator
            # display_document_preview(st.session_state.formatted_docx_path) # Already displayed above

        if st.session_state.formatted_pdf_path and os.path.exists(st.session_state.formatted_pdf_path) and export_format == "PDF":
            st.markdown("---")
            # display_document_preview(st.session_state.formatted_pdf_path) # Already displayed above


    elif active_feature == "🖼️ Export All Images from PDF":
        st.header("🖼️ Export All Images from PDF")
        st.markdown("Upload a PDF file to extract all unique images into a downloadable ZIP archive.")
        # (Logic from previous response, ensure extract_and_zip_images_from_pdf is imported from document_processing)
        pdf_file_for_img_export = st.file_uploader("Upload PDF for Image Extraction", type=["pdf"], key="img_export_uploader")
        if pdf_file_for_img_export:
            if st.button("Extract & Create ZIP of Images", key="extract_zip_btn"):
                with st.spinner("Extracting images..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=TEMP_DIR) as tmp_pdf_img:
                        tmp_pdf_img.write(pdf_file_for_img_export.getvalue())
                        tmp_pdf_img_path = tmp_pdf_img.name
                    
                    zip_name = f"images_{os.path.splitext(pdf_file_for_img_export.name)[0]}.zip"
                    # The extract_and_zip_images_from_pdf function now saves the zip and returns its path
                    zip_output_path = extract_and_zip_images_from_pdf(tmp_pdf_img_path, output_zip_name=zip_name)

                    if os.path.exists(tmp_pdf_img_path): os.unlink(tmp_pdf_img_path) # Clean up temp input PDF

                    if zip_output_path:
                        st.success("Images extracted and zipped!")
                        with open(zip_output_path, "rb") as fp_zip:
                            st.download_button("Download Images ZIP", fp_zip.read(), os.path.basename(zip_output_path), "application/zip")
                        # os.unlink(zip_output_path) # Clean up the generated zip from temp after download offered
                    elif zip_output_path is None: # Explicitly None if no images found
                        st.info("No unique images found in the PDF to extract.")
                    else: # Error occurred
                        st.error("Could not extract images. Please check the logs.")

    # ... (Other feature elif blocks: Plagiarism Check, Export to Google Drive)
    # Ensure they use can_use_premium_feature and the functions from external_apis.py

    # --- Footer ---
    st.markdown("---")
    st.caption(f"DocuMorph AI v0.4 | User: {st.session_state.user_id[:8]} | Tier: {st.session_state.user_tier.capitalize()}")
    # Display warnings for missing optional dependencies
    if not UNSTRUCTURED_AVAILABLE: st.sidebar.warning("Unstructured.io module unavailable. Advanced parsing limited.")
    if not REPORTLAB_AVAILABLE: st.sidebar.warning("ReportLab unavailable. PDF generation quality may be basic.")
    if not WEASYPRINT_AVAILABLE: st.sidebar.warning("WeasyPrint unavailable. Enhanced HTML-to-PDF quality limited.")


if __name__ == "__main__":
    # Ensure base storage directories exist on startup
    for dir_path_init in [VECTOR_STORE_DIR, DOCUMENTS_DIR, OUTPUT_DIR, TEMP_DIR]:
        os.makedirs(dir_path_init, exist_ok=True)
    main()