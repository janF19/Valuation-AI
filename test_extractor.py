from backend.src.financials.extractor import FinancialExtractor
from backend.config.settings import settings
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_extraction(file_path: str):
    try:
        # Initialize extractor
        extractor = FinancialExtractor(openai_api_key=settings.OPENAI_API_KEY)
        
        # Read the HTML file
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Extract data
        financial_data = extractor.extract_from_html(html_content)
        
        # Save the results
        output_file = 'extraction_results2.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(financial_data, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Results saved to {output_file}")
        logger.info("Extracted data summary:")
        logger.info(f"Income Statement items: {len(financial_data['income_statement'])}")
        logger.info(f"Balance Sheet items: {len(financial_data['balance_sheet'])}")
        logger.info(f"Cash Flow items: {len(financial_data['cash_flow'])}")
        logger.info(f"Company Information items: {len(financial_data['information'])}")
        
    except Exception as e:
        logger.error(f"Error during extraction: {e}", exc_info=True)

if __name__ == "__main__":
    # Use Windows-style paths with raw string to avoid escape issues
    
    file_path = r"backend\temp_cli_results\html_content.html"
    # Or use os.path for cross-platform compatibility
    # from os import path
    # file_path = path.join("backend", "src", "temp_cli_results", "output_vz_2023_isotra.html")
    test_extraction(file_path)