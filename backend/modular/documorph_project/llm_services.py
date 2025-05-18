import streamlit as st
import logging
import re
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, logger # Assuming GROQ_API_KEY and logger are in config.py
from ui_helpers import get_persona_specific_prompt # To fetch persona-based instructions

@st.cache_resource
def init_llm():
    try:
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY not found. LLM features will be disabled.")
            # st.error("Groq API key not configured.") # Avoid direct st.error in non-UI module
            return None
        return ChatGroq(
            model_name=os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192"), # Use env var or default
            temperature=0.2,
            groq_api_key=GROQ_API_KEY
        )
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}", exc_info=True)
        # st.error(f"Error initializing LLM: {str(e)}")
        return None

# --- LLM-Powered Functions (now with persona awareness) ---

def extract_document_structure_llm(chunks, llm, persona="General/Others"):
    if not llm: return "LLM not available for structure extraction."
    try:
        full_text = "\n\n".join([chunk.page_content for chunk in chunks])
        persona_instructions = get_persona_specific_prompt(persona, "structure")

        prompt_template_str = """Analyze the following document and identify its logical structure based on the persona context provided.
        Persona Context for Analysis: {persona_instructions}

        Extract the main sections, subsections, and their hierarchical relationships.
        For each section, provide:
        1. The heading/title
        2. The heading level (e.g., H1, H2, H3)
        3. A brief description of the content relative to the persona's interest.

        Document Content:
        {content}

        Output Format (example):
        - H1: [Section Title]
          Description: [Brief description aligned with persona]
          - H2: [Subsection Title]
            Description: [Brief description]
        """
        prompt = PromptTemplate(
            template=prompt_template_str,
            input_variables=["content", "persona_instructions"],
        )
        chain = prompt | llm | StrOutputParser()
        structure = chain.invoke({
            "content": full_text[:15000], # Limit context
            "persona_instructions": persona_instructions
        })
        return structure
    except Exception as e:
        logger.error(f"Error extracting document structure with LLM (persona: {persona}): {e}", exc_info=True)
        return f"Error extracting document structure: {str(e)}"


def generate_abstract_llm(chunks, llm, persona="General/Others", user_custom_instructions=""):
    if not llm: return "LLM not available for abstract generation."
    try:
        content = "\n\n".join([chunk.page_content for chunk in chunks[:5]]) # Use first 5 chunks
        base_persona_instructions = get_persona_specific_prompt(persona, "abstract")
        
        combined_instructions = base_persona_instructions
        if user_custom_instructions:
            combined_instructions += f"\n\nSpecific User Instructions: {user_custom_instructions}"

        prompt_template_str = """Generate a concise, professional abstract for the following document content.
        Adhere to the provided persona and user instructions.

        Persona Context & Base Instructions:
        {combined_instructions}

        Document Content:
        {content}

        Generate an abstract that accurately summarizes the main points, purpose, and findings.
        The abstract should be well-structured, clear, and approximately 150-250 words unless specified otherwise by instructions.
        """
        prompt = PromptTemplate(
            template=prompt_template_str,
            input_variables=["content", "combined_instructions"],
        )
        chain = prompt | llm | StrOutputParser()
        abstract = chain.invoke({"content": content[:15000], "combined_instructions": combined_instructions})
        return abstract
    except Exception as e:
        logger.error(f"Error generating abstract with LLM (persona: {persona}): {e}", exc_info=True)
        return f"Error generating abstract: {str(e)}"

def suggest_section_titles_llm(section_content, current_title, llm, persona="General/Others", user_custom_instructions=""):
    if not llm: return "LLM not available for title suggestions."
    try:
        base_persona_instructions = get_persona_specific_prompt(persona, "section_titles")
        combined_instructions = base_persona_instructions
        if user_custom_instructions:
            combined_instructions += f"\n\nSpecific User Instructions: {user_custom_instructions}"

        prompt_template_str = """Given the content from a document section and its current title (if any),
        suggest 2-3 alternative title options that accurately reflect the content.
        Consider the persona context and user instructions provided.

        Persona Context & Base Instructions for Titles:
        {combined_instructions}

        Current section title: {current_title}
        Section content:
        {content}

        Provide 2-3 alternative title suggestions in this format:
        1. [Title suggestion 1]
        2. [Title suggestion 2]
        3. [Title suggestion 3]
        Each title should be concise, specific, and reflective of the section's content and persona.
        """
        prompt = PromptTemplate(
            template=prompt_template_str,
            input_variables=["content", "current_title", "combined_instructions"],
        )
        chain = prompt | llm | StrOutputParser()
        title_suggestions = chain.invoke({
            "content": section_content[:4000],
            "current_title": current_title,
            "combined_instructions": combined_instructions
        })
        return title_suggestions
    except Exception as e:
        logger.error(f"Error suggesting section titles with LLM (persona: {persona}): {e}", exc_info=True)
        return f"Error suggesting section titles: {str(e)}"

def enhance_text_style_llm(text_to_enhance, llm, persona="General/Others", user_target_style=""):
    if not llm: return "LLM not available for style enhancement."
    try:
        # The 'target_style' for the LLM can be a combination of persona and user input
        base_persona_style = get_persona_specific_prompt(persona, "style") # This should return a descriptive style like "Academic"
        
        final_target_style = base_persona_style
        if user_target_style and user_target_style.lower() != base_persona_style.lower():
            final_target_style = f"{base_persona_style}, with an emphasis on '{user_target_style}'"
        elif user_target_style: # If user provided style is same as persona's default, just use it
             final_target_style = user_target_style


        prompt_template_str = """Enhance the following text by improving word choice, tone, and overall style
        to match the specified target style.

        Target Style: {final_target_style}

        Original text:
        {text_to_enhance}

        Rewrite the text to align with the target style, ensuring all original information and meaning are preserved.
        Focus on:
        - Replacing generic or weak terms with more precise, impactful, and contextually appropriate vocabulary.
        - Refining sentence structure for better clarity, flow, and readability.
        - Adjusting the tone to be consistent with the target style.
        - Enhancing the overall professionalism and polish of the text.
        Return only the rewritten text.
        """
        prompt = PromptTemplate(
            template=prompt_template_str,
            input_variables=["text_to_enhance", "final_target_style"],
        )
        chain = prompt | llm | StrOutputParser()
        enhanced_text = chain.invoke({"text_to_enhance": text_to_enhance[:4000], "final_target_style": final_target_style})
        return enhanced_text
    except Exception as e:
        logger.error(f"Error enhancing text style with LLM (persona: {persona}, style: {user_target_style}): {e}", exc_info=True)
        return f"Error enhancing text style: {str(e)}. Original text: {text_to_enhance}"


def get_word_suggestions_llm(current_text, llm, persona="General/Others", num_suggestions=3):
    if not llm: return ["LLM", "not", "available"]
    # (The rest of this function is mostly the same as before, persona doesn't heavily influence next-word prediction directly, but can be added to context if desired)
    try:
        words = current_text.split()
        context = " ".join(words[-20:]) if len(words) > 20 else current_text # Slightly more context
        # Persona hint for LLM context
        persona_hint = f"(Writing style context: {persona})"

        prompt = PromptTemplate(
            template="""Based on the following text {persona_hint}, suggest {num_suggestions} words or short phrases (1-3 words)
            that might contextually come next in a professional document.
            Current text:
            {context}
            Provide exactly {num_suggestions} suggestions in this format, each on a new line, starting with a number and a period:
            1. [suggestion 1]
            2. [suggestion 2]
            3. [suggestion 3]
            """,
            input_variables=["context", "num_suggestions", "persona_hint"],
        )
        chain = prompt | llm | StrOutputParser()
        suggestions_text = chain.invoke({
            "context": context,
            "num_suggestions": num_suggestions,
            "persona_hint": persona_hint
        })
        suggestions = []
        for line in suggestions_text.split('\n'):
            match = re.match(r'^\d+\.\s*(.+)', line.strip())
            if match:
                suggestion = match.group(1).strip().strip('[]') # Remove brackets if LLM adds them
                suggestions.append(suggestion)
        return suggestions[:num_suggestions] if suggestions else ["No", "suggestions", "found"]
    except Exception as e:
        logger.error(f"Error getting word suggestions with LLM: {e}", exc_info=True)
        return ["Error", "getting", "suggestions"]


def analyze_document_chapters_llm(doc_structure_text, llm, persona="General/Others"):
    if not llm: return [{"number": 1, "title": "Main Document (LLM N/A)", "subsections": []}]
    try:
        persona_instructions = get_persona_specific_prompt(persona, "structure") # Use structure prompt for chapter context
        
        prompt_template_str = """Analyze the following document structure, keeping in mind the persona context,
        and identify the main chapters or top-level sections suitable for numbering items like figures and tables.

        Persona Context for Analysis: {persona_instructions}

        Document Structure Outline:
        {doc_structure_text}

        Output a list of chapters. For each chapter, provide:
        1. The chapter number (integer, starting from 1).
        2. The chapter title (extracted or inferred from the structure).
        Optionally, list key subsections if clearly identifiable.

        Format:
        Chapter 1: [Chapter 1 Title]
        - Subsection: [Subsection Title 1A]
        - Subsection: [Subsection Title 1B]
        Chapter 2: [Chapter 2 Title]
        ...

        If the structure is very flat or no clear chapters are identifiable, provide a single entry like:
        Chapter 1: Main Document Content
        """
        prompt = PromptTemplate(
            template=prompt_template_str,
            input_variables=["doc_structure_text", "persona_instructions"],
        )
        chain = prompt | llm | StrOutputParser()
        chapter_analysis_text = chain.invoke({
            "doc_structure_text": doc_structure_text,
            "persona_instructions": persona_instructions
            })
        
        chapters = []
        current_chapter_dict = None
        for line in chapter_analysis_text.split('\n'):
            line = line.strip()
            chapter_match = re.match(r'Chapter\s+(\d+)[:\s]+(.+)', line, re.IGNORECASE)
            subsection_match = re.match(r'-\s*Subsection[:\s]+(.+)', line, re.IGNORECASE)

            if chapter_match:
                if current_chapter_dict: # Save previous chapter if exists
                    chapters.append(current_chapter_dict)
                current_chapter_dict = {
                    "number": int(chapter_match.group(1)),
                    "title": chapter_match.group(2).strip(),
                    "subsections": []
                }
            elif subsection_match and current_chapter_dict:
                current_chapter_dict["subsections"].append(subsection_match.group(1).strip())
        
        if current_chapter_dict and current_chapter_dict not in chapters: # Add the last chapter
            chapters.append(current_chapter_dict)
            
        if not chapters: # Fallback
             chapters = [{"number": 1, "title": "Main Document Content", "subsections": []}]
        logger.info(f"LLM analyzed chapters (persona {persona}): {chapters}")
        return chapters
    except Exception as e:
        logger.error(f"Error analyzing document chapters with LLM (persona: {persona}): {e}", exc_info=True)
        return [{"number": 1, "title": "Main Document (LLM analysis error)", "subsections": []}]


# --- Rule-Based Fallback Functions (content from previous llm_services.py) ---
def rule_based_structure_extraction(chunks):
    # ... (implementation from previous version) ...
    try:
        full_text = "\n\n".join([chunk.page_content for chunk in chunks])
        heading_patterns = [
            (r'^#+\s+(.+)$', 'H1'), (r'^(.+)\n=+$', 'H1'), (r'^(.+)\n-+$', 'H2'),
            (r'^[A-Z][A-Z\s()&-]{3,}[A-Z]$', 'H1'), # Improved ALL CAPS, allows some chars
            (r'^\d+(\.\d+)*\s+(.+)', 'H2'),    # Numbered headings 1. / 1.1.
            (r'^[A-Z][a-zA-Z\s,]{10,80}:?$', 'H3')  # Sentence-like, ends with : or not
        ]
        structure = []
        lines = full_text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 3 : continue # Skip very short lines
            
            is_heading = False
            for pattern, level in heading_patterns:
                match = re.match(pattern, line)
                if match:
                    heading_text = match.groups()[-1] if match.groups() else line # Get the last group (title)
                    
                    description = ""
                    # Look ahead for description, stop if another heading or too many lines
                    for j in range(i + 1, min(i + 4, len(lines))):
                        next_line_strip = lines[j].strip()
                        if not next_line_strip: continue # Skip empty
                        if any(re.match(p_next, next_line_strip) for p_next, _ in heading_patterns): # Stop if next heading
                            break
                        description += next_line_strip + " "
                        if len(description) > 200: break
                    
                    structure.append(f"- {level}: {heading_text.strip()}")
                    if description.strip():
                        structure.append(f"  Description: {description.strip()[:200]}...")
                    is_heading = True
                    break 
            # if not is_heading and len(line) > 50: # Treat longer non-heading lines as potential paragraph starts for context
            #     structure.append(f"  Paragraph Start: {line[:80]}...")


        if not structure:
            structure = ["- H1: Document Title\n  Description: Main document content..."]
        return "\n".join(structure)
    except Exception as e:
        logger.error(f"Error in rule-based structure extraction: {e}", exc_info=True)
        return "Error extracting document structure (rule-based)"

def rule_based_chapter_detection(chunks):
    # ... (implementation from previous version, can be simplified) ...
    chapters = []
    full_text = "\n\n".join([chunk.page_content for chunk in chunks])
    lines = full_text.split('\n')
    # More specific patterns for chapters
    chapter_patterns = [
        r'^(?:CHAPTER|Chapter|SECTION|Section)\s+([IVXLCDM\d]+)[:\s\.\-]+(.+)', # Chapter X: Title or Section X Title
        r'^([IVXLCDM\d]+)\.\s+([A-Z][A-Za-z\s]{5,})' #  1. Title like this
    ]
    
    chapter_count = 0
    for line in lines:
        line_strip = line.strip()
        if not line_strip or len(line_strip) < 5: continue

        for pattern in chapter_patterns:
            match = re.match(pattern, line_strip)
            if match:
                chapter_count += 1
                num_part = match.group(1).strip()
                title_part = match.group(2).strip()
                # Try to make chapter number an int if it's a digit, otherwise use the Roman numeral/string
                try: chapter_num_display = int(num_part)
                except ValueError: chapter_num_display = num_part

                chapters.append({
                    "number": chapter_count, # Sequential internal number
                    "title": f"{num_part}. {title_part}" if not title_part.lower().startswith(num_part.lower()) else title_part,
                    "subsections": [] # Subsections can be populated by more advanced logic if needed
                })
                break # Found chapter on this line

    if not chapters:
        chapters = [{"number": 1, "title": "Main Document", "subsections": []}]
    return chapters


def rule_based_abstract_generation(chunks):
    # ... (implementation from previous version) ...
    try:
        content = "\n\n".join([chunk.page_content for chunk in chunks[:4]]) # First 4 chunks
        # Remove bracketed placeholders like [TABLE: ...], [FIGURE: ...] before sentence splitting
        content_cleaned = re.sub(r'\[(TABLE|FIGURE|TITLE):.*?\]', '', content, flags=re.IGNORECASE | re.DOTALL)
        
        sentences = re.split(r'(?<=[.!?])\s+', content_cleaned.strip())
        
        # Filter out very short sentences or non-alphanumeric ones
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 25 and re.search(r'[a-zA-Z]', s)]

        if not meaningful_sentences:
            return "Insufficient content for a meaningful rule-based abstract. Document appears to be very short or non-textual."

        selected_sentences = []
        if meaningful_sentences:
            selected_sentences.append(meaningful_sentences[0]) # First meaningful sentence
            
            if len(meaningful_sentences) > 5: # If enough content, take some from middle
                mid_idx1 = len(meaningful_sentences) // 3
                mid_idx2 = 2 * len(meaningful_sentences) // 3
                if mid_idx1 > 0: selected_sentences.append(meaningful_sentences[mid_idx1])
                if mid_idx2 > mid_idx1 and mid_idx2 < len(meaningful_sentences) -1 : selected_sentences.append(meaningful_sentences[mid_idx2])
            
            if len(meaningful_sentences) > 1 and meaningful_sentences[-1] not in selected_sentences:
                 selected_sentences.append(meaningful_sentences[-1]) # Last meaningful sentence
        
        # Remove duplicates while preserving order (Python 3.7+)
        selected_sentences = list(dict.fromkeys(selected_sentences))

        abstract = " ".join(selected_sentences)
        
        if len(abstract) < 70: # Too short
            return f"This document covers material related to its title. (Content length: {len(content_cleaned)} chars)"
        
        return abstract[:800] # Limit length
    except Exception as e:
        logger.error(f"Error in rule-based abstract generation: {e}", exc_info=True)
        return "This document contains information relevant to the subject matter."


def rule_based_word_suggestions(current_text):
    # ... (implementation from previous version, keep it simple) ...
    default_suggestions = ["the", "and", "therefore", "however", "furthermore"]
    words = current_text.lower().split()
    if not words: return ["The", "This", "In"] # Start of document/sentence
    last_word = words[-1].strip(".,!?;:")
    
    # Simple common followers (can be expanded greatly)
    if last_word == "is": return ["a", "the", "important"]
    if last_word == "the": return ["next", "following", "main"]
    if last_word == "in": return ["this", "summary", "conclusion"]
    if current_text.endswith((". ", "! ", "? ")): return ["Furthermore", "However", "Therefore"]

    return default_suggestions[:3]