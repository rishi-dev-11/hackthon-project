# ai_processing.py
import streamlit as st
import logging
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import spacy
# from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer # Lazy loaded

logger = logging.getLogger(__name__)

@st.cache_resource
def init_llm():
    try:
        return ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct", # Consider making model name a config
            temperature=0.2,
            api_key=st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "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")) # Prefer secrets or env
        )
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}", exc_info=True)
        st.error(f"Error initializing LLM: {str(e)}")
        return None

@st.cache_resource
def init_nlp_models():
    try:
        nlp = spacy.load("en_core_web_sm")
        translation_model = None # Lazy load M2M100
        translation_tokenizer = None
        logger.info("spaCy model loaded. Translation models will be lazy-loaded if needed.")
        return {
            "nlp": nlp,
            "translation_model": translation_model,
            "translation_tokenizer": translation_tokenizer
        }
    except Exception as e:
        logger.error(f"Error initializing NLP models: {e}", exc_info=True)
        st.warning(f"Could not load all NLP models: {e}")
        return {"nlp": None, "translation_model": None, "translation_tokenizer": None}


def extract_document_structure(chunks, llm):
    if not llm: return "LLM not available for structure extraction."
    try:
        full_text = "\n\n".join([chunk.page_content for chunk in chunks])
        prompt = PromptTemplate(
            template="""Analyze the following document and identify its logical structure. 
            Extract the main sections, subsections, and their hierarchical relationships.
            For each section, provide:
            1. The heading/title
            2. The heading level (H1, H2, H3, etc.)
            3. A brief description of the content
            Document Content:
            {content}
            Output Format:
            - H1: [Section Title]
              Description: [Brief description]
              - H2: [Subsection Title]
                Description: [Brief description]
                - H3: [Sub-subsection Title]
                  Description: [Brief description]
            """,
            input_variables=["content"],
        )
        chain = prompt | llm | StrOutputParser()
        structure = chain.invoke({"content": full_text[:10000]}) # Limit input size
        return structure
    except Exception as e:
        logger.error(f"Error extracting document structure: {e}", exc_info=True)
        return f"Error extracting document structure: {str(e)}"

def generate_abstract(chunks, llm, instructions=""):
    if not llm: return "LLM not available for abstract generation."
    try:
        content = "\n\n".join([chunk.page_content for chunk in chunks[:5]]) # Use first 5 chunks
        prompt = PromptTemplate(
            template="""Generate a concise, professional abstract for the following document content.
            Document Content:
            {content}
            User Instructions: {instructions}
            Generate an abstract that accurately summarizes the main points, purpose, and findings of the document.
            The abstract should be well-structured, clear, and approximately 150-250 words.
            """,
            input_variables=["content", "instructions"],
        )
        chain = prompt | llm | StrOutputParser()
        abstract = chain.invoke({"content": content, "instructions": instructions})
        return abstract
    except Exception as e:
        logger.error(f"Error generating abstract: {e}", exc_info=True)
        return f"Error generating abstract: {str(e)}"

def suggest_section_titles(section_content, current_title, llm, instructions=""):
    if not llm: return "LLM not available for title suggestions."
    try:
        prompt = PromptTemplate(
            template="""Given the following content from a document section and its current title (if any),
            suggest 2-3 alternative title options that accurately reflect the content and are professional and engaging.
            Current section title: {current_title}
            Section content:
            {content}
            User Instructions: {instructions}
            Provide 2-3 alternative title suggestions in this format:
            1. [Title suggestion 1]
            2. [Title suggestion 2]
            3. [Title suggestion 3]
            Each title should be concise, specific, and reflective of the section's content.
            """,
            input_variables=["content", "current_title", "instructions"],
        )
        chain = prompt | llm | StrOutputParser()
        title_suggestions = chain.invoke({
            "content": section_content[:2000], # Limit input size
            "current_title": current_title,
            "instructions": instructions
        })
        return title_suggestions
    except Exception as e:
        logger.error(f"Error suggesting section titles: {e}", exc_info=True)
        return f"Error suggesting section titles: {str(e)}"

def enhance_text_style(text, target_style, llm):
    if not llm: return "LLM not available for style enhancement."
    try:
        prompt = PromptTemplate(
            template="""Enhance the following text by improving word choice, tone, and style according to the target style.
            Original text:
            {text}
            Target style: {target_style}
            Rewrite the text to match the target style while preserving all information.
            Focus on replacing generic terms with more precise and contextually appropriate words,
            improving sentence structure, and enhancing the overall professionalism of the text.
            """,
            input_variables=["text", "target_style"],
        )
        chain = prompt | llm | StrOutputParser()
        enhanced_text = chain.invoke({"text": text, "target_style": target_style})
        return enhanced_text
    except Exception as e:
        logger.error(f"Error enhancing text style: {e}", exc_info=True)
        return f"Error enhancing text style: {str(e)}"

def get_word_suggestions(current_text, llm, num_suggestions=3):
    if not llm: return ["LLM", "not", "available"]
    try:
        words = current_text.split()
        context = " ".join(words[-10:]) if len(words) > 10 else current_text
        prompt = PromptTemplate(
            template="""Based on the following text, suggest {num_suggestions} words or phrases that might 
            come next in a professional document. The suggestions should be contextually relevant and 
            help the user complete their thought.
            Current text:
            {context}
            Provide exactly {num_suggestions} suggestions in this format:
            1. [suggestion 1]
            2. [suggestion 2]
            3. [suggestion 3]
            The suggestions should be brief (1-3 words) and professional.
            """,
            input_variables=["context", "num_suggestions"],
        )
        chain = prompt | llm | StrOutputParser()
        suggestions_text = chain.invoke({
            "context": context,
            "num_suggestions": num_suggestions
        })
        suggestions = []
        for line in suggestions_text.split('\n'):
            if re.match(r'^\d+\.', line.strip()):
                suggestion = line.split('.', 1)[1].strip()
                if suggestion.startswith('[') and suggestion.endswith(']'):
                    suggestion = suggestion[1:-1]
                suggestions.append(suggestion)
        return suggestions[:num_suggestions] if suggestions else ["Try", "typing", "more"]
    except Exception as e:
        logger.error(f"Error getting word suggestions: {e}", exc_info=True)
        return ["Error", "getting", "suggestions"]

def analyze_document_chapters(doc_structure, llm):
    if not llm: return [] # LLM not available
    try:
        prompt = PromptTemplate(
            template="""Analyze the following document structure and identify the main chapters or sections
            that should be used for numbering figures and tables.
            Document Structure:
            {structure}
            For each major section (chapter), provide:
            1. The chapter number (starting with 1)
            2. The chapter title
            3. Any subsections that belong to this chapter
            Output Format:
            Chapter 1: [Title]
            - Subsection: [Title] 
            - Subsection: [Title]
            Chapter 2: [Title]
            - Subsection: [Title]
            ...
            """,
            input_variables=["structure"],
        )
        chain = prompt | llm | StrOutputParser()
        chapter_analysis = chain.invoke({"structure": doc_structure})
        chapters = []
        current_chapter = None
        for line in chapter_analysis.split('\n'):
            line = line.strip()
            if line.startswith('Chapter '):
                chapter_match = re.match(r'Chapter (\d+): (.+)', line)
                if chapter_match:
                    current_chapter = {
                        "number": int(chapter_match.group(1)),
                        "title": chapter_match.group(2),
                        "subsections": []
                    }
                    chapters.append(current_chapter)
            elif line.startswith('- Subsection: ') and current_chapter:
                subsection_title = line[13:].strip()
                current_chapter["subsections"].append(subsection_title)
        return chapters
    except Exception as e:
        logger.error(f"Error analyzing document chapters: {e}", exc_info=True)
        return []

# --- Rule-based alternatives ---
def rule_based_structure_extraction(chunks):
    try:
        full_text = "\n\n".join([chunk.page_content for chunk in chunks])
        heading_patterns = [
            (r'^#+\s+(.+)$', 'H1'), (r'^(.+)\n=+$', 'H1'), (r'^(.+)\n-+$', 'H2'),
            (r'^[A-Z][A-Z\s]+[A-Z]$', 'H1'), (r'^[0-9]+\.\s+(.+)$', 'H2'), 
            (r'^[A-Z][a-z].{10,60}$', 'H3')
        ]
        structure_output = []
        lines = full_text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            for pattern, level in heading_patterns:
                match = re.match(pattern, line)
                if match:
                    heading_text = match.group(1) if match.groups() else line
                    description = ""
                    if i + 1 < len(lines):
                        for j in range(i + 1, min(i + 5, len(lines))):
                            if lines[j].strip() and not any(re.match(p, lines[j].strip()) for p, _ in heading_patterns):
                                description += lines[j].strip() + " "
                    structure_output.append(f"- {level}: {heading_text}")
                    structure_output.append(f"  Description: {description[:200].strip()}...")
                    break # Found a heading pattern for this line
        if not structure_output:
            return "- H1: Document Title\n  Description: Main document content..."
        return "\n".join(structure_output)
    except Exception as e:
        logger.error(f"Error in rule-based structure extraction: {e}", exc_info=True)
        return "Error extracting document structure using rule-based approach"

def rule_based_chapter_detection(chunks):
    try:
        chapters = [
            {"number": 1, "title": "Introduction", "subsections": ["Overview", "Background"]},
            {"number": 2, "title": "Content", "subsections": ["Main Points", "Details"]},
            {"number": 3, "title": "Conclusion", "subsections": ["Summary", "Next Steps"]}
        ]
        full_text = "\n\n".join([chunk.page_content for chunk in chunks])
        lines = full_text.split('\n')
        chapter_patterns = [
            r'Chapter\s+(\d+)[:\s]+(.+)', r'Section\s+(\d+)[:\s]+(.+)', r'^(\d+)\.\s+(.+)'
        ]
        detected_chapters = []
        for line in lines:
            for pattern in chapter_patterns:
                match = re.search(pattern, line)
                if match:
                    try:
                        chapter_num = int(match.group(1))
                        chapter_title = match.group(2).strip()
                        # Avoid adding duplicate chapter numbers
                        if not any(ch['number'] == chapter_num for ch in detected_chapters):
                           detected_chapters.append({"number": chapter_num, "title": chapter_title, "subsections": []})
                    except ValueError: # Handle cases where group(1) might not be a number
                        continue
        if detected_chapters:
            chapters = sorted(detected_chapters, key=lambda x: x['number']) # Sort by chapter number
            # Simple subsection detection (can be improved)
            for i, ch in enumerate(chapters):
                # Look for lines that might be subsections following this chapter's title
                # This is a very basic heuristic
                start_index = 0
                if ch['title'] in full_text:
                    start_index = full_text.find(ch['title'])
                
                next_chapter_start_index = len(full_text)
                if i + 1 < len(chapters) and chapters[i+1]['title'] in full_text:
                    next_chapter_start_index = full_text.find(chapters[i+1]['title'])

                chapter_content = full_text[start_index:next_chapter_start_index]
                for sub_line in chapter_content.split('\n'):
                    sub_line = sub_line.strip()
                    # Example: "1.1 Subsection Title" or "A. Subsection Title"
                    if re.match(r"^\d+\.\d+\s+.+", sub_line) or re.match(r"^[A-Za-z]\.\s+.+", sub_line):
                        if len(sub_line) > 5 and len(sub_line) < 100: # Basic length check
                             # Remove the numbering part
                            subsection_title = re.sub(r"^\d+\.\d+\s+", "", sub_line)
                            subsection_title = re.sub(r"^[A-Za-z]\.\s+", "", subsection_title)
                            if subsection_title not in ch['subsections']:
                                ch['subsections'].append(subsection_title.strip())
        return chapters if chapters else [{"number": 1, "title": "Document", "subsections": []}]
    except Exception as e:
        logger.error(f"Error in rule-based chapter detection: {e}", exc_info=True)
        return [{"number": 1, "title": "Document", "subsections": []}]


def rule_based_abstract_generation(chunks):
    try:
        content = "\n\n".join([chunk.page_content for chunk in chunks[:3]])
        sentences = re.split(r'(?<=[.!?])\s+', content)
        selected_sentences = []
        if sentences:
            if len(sentences[0]) > 20: selected_sentences.append(sentences[0])
            middle_start = max(1, len(sentences) // 3)
            middle_end = min(len(sentences) - 1, middle_start + 3) # Take up to 3 from middle
            for i in range(middle_start, middle_end):
                if i < len(sentences) and len(sentences[i]) > 30:
                    selected_sentences.append(sentences[i])
            if len(sentences) > 1 and len(sentences[-1]) > 20 and sentences[-1] not in selected_sentences:
                 selected_sentences.append(sentences[-1])
        
        abstract = " ".join(list(dict.fromkeys(selected_sentences))) # Remove duplicates, preserve order
        if len(abstract) < 100 and abstract:
            abstract += " This document provides information on the subject matter covered within its contents."
        elif not abstract:
            abstract = "This document contains information relevant to the subject matter. For a more detailed abstract, upgrade to premium tier."
        return abstract[:500] # Limit abstract length
    except Exception as e:
        logger.error(f"Error in rule-based abstract generation: {e}", exc_info=True)
        return "Error in rule-based abstract generation."

def rule_based_word_suggestions(current_text):
    try:
        suggestion_dict = {
            "the": ["following", "most", "best"], "a": ["significant", "major", "detailed"],
            "this": ["approach", "method", "result"], "in": ["addition", "conclusion", "summary"],
            "for": ["example", "instance", "reference"], "to": ["summarize", "conclude", "illustrate"],
            "of": ["course", "importance", "significance"], "with": ["respect", "regard", "reference"],
            "and": ["therefore", "thus", "furthermore"], "is": ["important", "necessary", "critical"]
        }
        default_suggestions = ["the", "and", "therefore"]
        words = current_text.lower().split()
        last_word = words[-1].strip(".,!?;:") if words else ""
        
        if last_word in suggestion_dict:
            return suggestion_dict[last_word][:3]
        if not words or current_text.endswith(('.', '!', '?','\n')):
            return ["The", "This", "In"][:3]
        return default_suggestions[:3]
    except Exception as e:
        logger.error(f"Error in rule-based word suggestions: {e}", exc_info=True)
        return ["and", "the", "therefore"]
