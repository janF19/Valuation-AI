import logging
from pathlib import Path
from backend.src.ocr.processing import OCRProcessor
#from backend.src.financials.extractor import FinancialExtractor
from backend.src.reporting.generator import ReportGenerator
from backend.config.settings import settings
from backend.src.valuation.valuator import CompanyValuator
from backend.src.financials.extractor import FinancialExtractor

# Configure logging here
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ValuationWorkflow:
    def execute(self, file_path: str, output_path: str = None):
        logger.info("Starting valuation workflow")
        
        # Create temp_results directory
        temp_dir = Path(__file__).parent / "temp_cli_results"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing document: {file_path}")
            
        try:
            # OCR Processing
            logger.info("Initializing OCR processing")
            # Verify API key is available
            if not settings.MISTRAL_API_KEY:
                logger.error("MISTRAL_API_KEY is not set or empty")
                raise ValueError("MISTRAL_API_KEY is missing. Please check your environment variables or settings.")
            
            # Add more detailed logging for API key
            logger.info(f"Using Mistral API key (first 5 chars): {settings.MISTRAL_API_KEY[:5]}...")
            logger.info(f"API key length: {len(settings.MISTRAL_API_KEY)}")
            
            processor = OCRProcessor(api_key=settings.MISTRAL_API_KEY)
            
            logger.info("Processing document with OCR")
            try:
                # Add a retry mechanism for API calls
                max_retries = 3
                retry_count = 0
                last_error = None
                
                while retry_count < max_retries:
                    try:
                        logger.info(f"OCR attempt {retry_count + 1}/{max_retries}")
                        result = processor.process_document(
                            str(file_path), format="html"
                        )
                        
                        html_content = result[0] if isinstance(result, tuple) else result
                        logger.info("OCR processing completed successfully")
                        
                        # Save html_content
                        with open(temp_dir / "html_content.html", "w", encoding="utf-8") as f:
                            f.write(html_content)
                        logger.info("Saved HTML content to temp_results")
                        break  # Success, exit the retry loop
                    except Exception as retry_error:
                        retry_count += 1
                        last_error = retry_error
                        logger.warning(f"OCR attempt {retry_count} failed: {str(retry_error)}")
                        if retry_count < max_retries:
                            import time
                            wait_time = 2 ** retry_count  # Exponential backoff
                            logger.info(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"All {max_retries} OCR attempts failed")
                            raise last_error
            except Exception as ocr_error:
                logger.error(f"OCR processing failed: {str(ocr_error)}")
                
                # Check if it's an authentication error
                if "401 Unauthorized" in str(ocr_error) or "authentication" in str(ocr_error).lower():
                    logger.error("Authentication error detected. Please verify your Mistral API key is valid and not expired.")
                    raise ValueError("Mistral API authentication failed. Please check your API key.") from ocr_error
                
                raise
           
            
            
            logger.info("Extracting financial data")
            extractor = FinancialExtractor()
            financial_data = extractor.extract_from_html(html_content)
            logger.info("Financial data extraction completed")
            # Save financial_data with proper encoding for Czech characters
            import json
            with open(temp_dir / "financial_data.json", "w", encoding="utf-8") as f:
                json.dump(financial_data, f, indent=4, ensure_ascii=False)
            logger.info("Saved financial data to temp_results")
            
 
            
            logger.info("Calculating valuation multiples")
            valuation_multiple = CompanyValuator(financial_data)
            result_valuation = valuation_multiple.calculate_multiples()
            logger.info("Valuation calculations completed")
            
            # Save result_valuation with proper encoding for Czech characters
            with open(temp_dir / "result_valuation.json", "w", encoding="utf-8") as f:
                json.dump(result_valuation, f, indent=4, ensure_ascii=False)
            logger.info("Saved valuation results to temp_results")
            
            # Generate Report
            logger.info("Generating final report")
            report = ReportGenerator.generate({
                "financial_data": financial_data,
                "result_valuation": result_valuation
            })
            
            # Determine where to save the report
            if output_path:
                save_path = Path(output_path)
            else:
                # Default behavior: save to backend/data
                output_dir = Path(__file__).parent / "data"
                output_dir.mkdir(exist_ok=True)
                save_path = output_dir / "report_output.docx"
            
            # Create parent directories if they don't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the report
            logger.info(f"Saving report to {save_path}")
            report.save(str(save_path))
            logger.info("Report saved successfully")
            
            return report
            
        except Exception as e:
            logger.error(f"Error in valuation workflow: {str(e)}", exc_info=True)
            raise