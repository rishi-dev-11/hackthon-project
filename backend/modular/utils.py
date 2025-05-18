# utils.py
import logging
import fitz # PyMuPDF
import re
from config import UserTier # Import UserTier from config.py

logger = logging.getLogger(__name__)

# PyMuPDF compatibility wrapper (If not using the one from documorph_fixes exclusively)
# If documorph_fixes.open_pdf is preferred, this can be removed or adapted.
def open_pdf_internal(pdf_path):
    """Open a PDF with PyMuPDF using the most compatible method."""
    try:
        return fitz.open(pdf_path)
    except AttributeError:
        try:
            return fitz.Document(pdf_path)
        except AttributeError:
            # This part depends on whether you're using the 'pymupdf' or 'fitz' import style
            # For 'import fitz', 'fitz.Document' might be the old way.
            # If you installed PyMuPDF recently, 'fitz.open()' is standard.
            raise ImportError("Could not initialize PyMuPDF with common methods.")


def parse_word_suggestions(text):
    """Parse the word suggestions output from LLM into a structured format."""
    suggestions = {}
    current_phrase = None
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("Original phrase"):
            try:
                phrase_parts = line.split(":", 1)
                if len(phrase_parts) > 1:
                    current_phrase = phrase_parts[1].strip().strip('"')
                    suggestions[current_phrase] = []
            except:
                continue
        
        elif line.startswith("-") and current_phrase:
            try:
                alt_parts = line.split(":", 1)
                if len(alt_parts) > 1:
                    alternative = alt_parts[1].strip().strip('"')
                    suggestions[current_phrase].append(alternative)
            except:
                continue
    return suggestions

def can_use_feature(user_tier, feature_name):
    """Check if the user's tier has access to a specific feature."""
    tier_features = UserTier.get_tier_features()
    return tier_features[user_tier].get(feature_name, False)

def can_use_template_role(user_tier, role):
    """Check if the user's tier allows using templates with this role."""
    tier_features = UserTier.get_tier_features()
    allowed_categories = tier_features[user_tier]["template_categories"]
    return role in allowed_categories

def get_role_specific_prompt(role, task_type):
    """Generate role-specific prompts for different document processing tasks."""
    prompts = {
        "Student": {
            "abstract": """Generate a concise, academic abstract for this student document.
            Focus on clarity, proper academic terminology, and highlighting the main arguments or findings.
            The abstract should be well-structured and approximately 150-250 words.
            Use formal language appropriate for academic submission.
            """,
            "section_titles": """Suggest academic section titles that would be appropriate for a student paper.
            Titles should be clear, descriptive, and follow academic conventions.
            Consider standard sections like Introduction, Literature Review, Methodology, Results, Discussion, and Conclusion.
            """,
            "style": "Academic",
            "structure": """Analyze this student document and suggest an academic structure.
            Focus on logical flow, proper academic sections, and clear organization of ideas.
            Identify where the document could benefit from better section organization or additional headings.
            """
        },
        "Content Creator": {
            "abstract": """Generate an engaging, reader-friendly summary for this content piece.
            Focus on hooking the reader's interest, highlighting key points, and creating a compelling preview.
            The summary should be approximately 100-200 words and use engaging, conversational language.
            """,
            "section_titles": """Suggest engaging, attention-grabbing section titles for this content.
            Titles should be catchy, memorable, and encourage readers to continue reading.
            Consider creative headings that spark curiosity while clearly indicating section content.
            """,
            "style": "Engaging and Conversational",
            "structure": """Analyze this content and suggest a structure optimized for reader engagement.
            Focus on creating a narrative flow, using hooks throughout the content, and organizing information
            in a way that maintains reader interest from beginning to end.
            """
        },
        "Researcher": {
            "abstract": """Generate a comprehensive, scientific abstract for this research document.
            Follow standard scientific abstract structure: background, methods, results, and conclusions.
            The abstract should be precise, data-focused, and approximately 200-300 words.
            Use formal scientific language and emphasize research significance and findings.
            """,
            "section_titles": """Suggest formal research section titles following scientific conventions.
            Titles should be precise, descriptive, and follow standard research paper organization.
            Consider sections like Abstract, Introduction, Methods, Results, Discussion, Conclusion, and References.
            """,
            "style": "Scientific and Formal",
            "structure": """Analyze this research document and suggest a structure following scientific conventions.
            Focus on logical progression of ideas, proper organization of research components, and clear
            separation between different elements of the research (methods, results, discussion, etc.).
            """
        },
        "Business Professional": {
            "abstract": """Generate a concise executive summary for this business document.
            Focus on key business insights, actionable information, and bottom-line impact.
            The summary should be approximately 150-250 words and use clear, professional language.
            Emphasize business value, recommendations, and strategic implications.
            """,
            "section_titles": """Suggest professional business section titles that convey authority and clarity.
            Titles should be direct, action-oriented, and clearly communicate section purpose.
            Consider sections like Executive Summary, Market Analysis, Strategic Recommendations, Implementation Plan, etc.
            """,
            "style": "Professional and Concise",
            "structure": """Analyze this business document and suggest a structure optimized for business decision-makers.
            Focus on presenting information in a hierarchy of importance, with executive summary first,
            followed by supporting details, analysis, and recommendations or next steps.
            """
        },
        "Multilingual User": {
            "abstract": """Generate a clear, straightforward abstract that would be easily understood across languages.
            Focus on simple sentence structures, common vocabulary, and universal concepts.
            The abstract should be approximately 150-250 words and avoid idioms or culturally specific references.
            """,
            "section_titles": """Suggest clear, universally understandable section titles.
            Titles should be straightforward, descriptive, and avoid language-specific idioms or complex terms.
            Consider simple, direct headings that would translate well across multiple languages.
            """,
            "style": "Clear and Universally Accessible",
            "structure": """Analyze this document and suggest a structure that would work well across languages and cultures.
            Focus on universal organizational patterns, clear progression of ideas, and avoiding
            culturally specific organizational structures that might not translate well.
            """
        },
        "Author": {
            "abstract": """Generate an engaging book chapter summary or overview.
            Focus on narrative elements, themes, key arguments, and reader takeaways.
            The summary should be approximately 150-250 words and reflect the author's voice and style.
            Create something that would entice readers to continue reading the full chapter.
            """,
            "section_titles": """Suggest creative yet descriptive section titles appropriate for a book chapter.
            Titles should be balance creativity with clarity and maintain the author's voice.
            Consider how these headings guide the reader through the narrative or argument.
            """,
            "style": "Literary and Narrative-Focused",
            "structure": """Analyze this document and suggest a structure appropriate for a book chapter.
            Focus on narrative flow, thematic development, and reader engagement.
            Consider how to organize content to build reader interest throughout the chapter.
            """
        },
        "Collaborator": {
            "abstract": """Generate a clear team document summary that highlights key points for all collaborators.
            Focus on shared goals, action items, responsibilities, and next steps.
            The summary should be approximately 150-200 words and use clear, inclusive language.
            Emphasize collaborative elements and information relevant to all team members.
            """,
            "section_titles": """Suggest practical section titles that facilitate team collaboration.
            Titles should be clear, action-oriented, and help team members quickly find relevant information.
            Consider sections like Project Overview, Team Responsibilities, Timeline, Action Items, etc.
            """,
            "style": "Clear, Practical, and Action-Oriented",
            "structure": """Analyze this collaborative document and suggest a structure that facilitates team work.
            Focus on organizing information for quick reference, clear responsibility assignment,
            and effective tracking of project elements and progress.
            """
        },
        "Project Manager": {
            "abstract": """Generate a comprehensive project summary highlighting key objectives, status, and outcomes.
            Focus on project metrics, milestones, resource allocation, and critical path elements.
            The summary should be approximately 150-250 words and use precise project management terminology.
            Emphasize timeline, deliverables, and current status relative to project goals.
            """,
            "section_titles": """Suggest project management section titles that facilitate project tracking and reporting.
            Titles should be specific, metric-oriented, and aligned with project management methodologies.
            Consider sections like Project Scope, Timeline, Resource Allocation, Risk Assessment, etc.
            """,
            "style": "Structured and Metrics-Focused",
            "structure": """Analyze this project document and suggest a structure optimized for project management.
            Focus on organizing information to track project progress, highlight dependencies,
            and clearly communicate status, risks, and next steps to stakeholders.
            """
        }
    }
    default_prompts = {
        "abstract": """Generate a concise, professional abstract summarizing the main points of this document.
        The abstract should be well-structured, clear, and approximately 150-250 words.
        """,
        "section_titles": """Suggest professional section titles that would improve the organization of this document.
        Titles should be clear, descriptive, and help guide the reader through the content.
        """,
        "style": "Professional",
        "structure": """Analyze this document and suggest an improved structure.
        Focus on logical organization, clear progression of ideas, and appropriate section divisions.
        """
    }
    role_prompts = prompts.get(role, default_prompts)
    return role_prompts.get(task_type, default_prompts.get(task_type, ""))
