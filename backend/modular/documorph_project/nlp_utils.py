import streamlit as st
import logging
import spacy
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langdetect import detect as langdetect_detect, LangDetectException

logger = logging.getLogger("DocuMorphAI")

@st.cache_resource
def init_nlp_models():
    nlp_models = {
        "nlp_spacy": None, # Renamed for clarity
        "translation_model": None,
        "translation_tokenizer": None,
        "spacy_available": False,
        "translator_available": False # Will be set to True if/when loaded
    }
    try:
        nlp_models["nlp_spacy"] = spacy.load("en_core_web_sm")
        nlp_models["spacy_available"] = True
        logger.info("spaCy 'en_core_web_sm' model loaded for NLP tasks.")
    except IOError: # More specific error for model not found
        logger.warning("spaCy 'en_core_web_sm' model not found. Download with 'python -m spacy download en_core_web_sm'. Some NLP features (like advanced section detection) will be limited.")
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
    return nlp_models

@st.cache_resource # Cache the translation model and tokenizer together
def _load_translation_resources(): # Underscore indicates an internal helper
    try:
        model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        logger.info("M2M100 translation model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load M2M100 translation resources: {e}")
        st.warning("Translation features might be unavailable due to model loading issues.")
        return None, None

def get_translation_model_tokenizer(nlp_models_dict):
    """Lazily loads or retrieves translation model and tokenizer."""
    if not nlp_models_dict.get("translator_available", False) or \
       not nlp_models_dict["translation_model"] or \
       not nlp_models_dict["translation_tokenizer"]:
        
        model, tokenizer = _load_translation_resources()
        if model and tokenizer:
            nlp_models_dict["translation_model"] = model
            nlp_models_dict["translation_tokenizer"] = tokenizer
            nlp_models_dict["translator_available"] = True
        else: # Failed to load
            nlp_models_dict["translator_available"] = False
            return None, None # Explicitly return None if loading fails
            
    return nlp_models_dict["translation_model"], nlp_models_dict["translation_tokenizer"]


def detect_language(text_sample: str) -> str:
    """Detects language of a given text sample."""
    if not text_sample or not text_sample.strip():
        return 'en' # Default if no text
    try:
        # Ensure text_sample is not too short for langdetect
        if len(text_sample) < 20: # langdetect might struggle with very short texts
            return 'en' # Default for very short strings
        lang = langdetect_detect(text_sample)
        return lang
    except LangDetectException: # Specifically catch langdetect errors
        logger.warning(f"Language detection failed for sample '{text_sample[:50]}...'. Defaulting to 'en'.")
        return 'en'
    except Exception as e:
        logger.error(f"Unexpected error in language detection: {e}")
        return 'en'

def translate_text(text_to_translate: str, target_language_code: str, nlp_models_dict, source_language_code: str = 'en') -> str:
    """Translates text to the target language using M2M100 model."""
    if not text_to_translate or source_language_code == target_language_code:
        return text_to_translate

    model, tokenizer = get_translation_model_tokenizer(nlp_models_dict)

    if not model or not tokenizer:
        logger.warning(f"Translation model not available. Cannot translate '{text_to_translate[:50]}...' to {target_language_code}.")
        return text_to_translate # Return original text if translation isn't possible

    try:
        tokenizer.src_lang = source_language_code
        # Ensure text is not excessively long for a single translation call
        # Truncation might be too aggressive; consider splitting for long texts if quality is paramount.
        encoded_input = tokenizer(text_to_translate, return_tensors="pt", truncation=True, max_length=512)
        
        # Ensure target language ID is valid
        try:
            target_lang_id = tokenizer.get_lang_id(target_language_code)
        except KeyError:
            logger.error(f"Invalid target language code '{target_language_code}' for M2M100 tokenizer.")
            return text_to_translate # Fallback if language code is not supported

        generated_tokens = model.generate(**encoded_input, forced_bos_token_id=target_lang_id)
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        logger.info(f"Translated '{text_to_translate[:30]}...' from {source_language_code} to {target_language_code}: '{translated_text[:30]}...'")
        return translated_text
    except Exception as e:
        logger.error(f"Translation error from {source_language_code} to {target_language_code} for text '{text_to_translate[:50]}...': {e}", exc_info=True)
        return text_to_translate # Fallback to original text