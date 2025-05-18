# DocuMorph AI backend package initialization
# This file ensures Python treats this directory as a package

# Import main components for convenience
try:
    from .table_extraction import extract_tables
    from .chart_extraction import identify_and_extract_charts
except ImportError:
    pass 