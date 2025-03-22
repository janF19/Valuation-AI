import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.src.financials.extractor import FinancialExtractor
from backend.config.settings import settings  # Import settings object




# Process a single report
extractor = FinancialExtractor()

# HTML-based extraction
with open('financial_report.html') as f:
    html_data = f.read()
html_results = extractor.extract_from_html(html_data)