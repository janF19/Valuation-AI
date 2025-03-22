import logging
from pathlib import Path
from backend.src.ocr.processing import OCRProcessor
from backend.src.financials.extractor import FinancialExtractor
from backend.src.reporting.generator import ReportGenerator
from backend.config.settings import settings
from backend.src.valuation.valuator import CompanyValuator

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
            processor = OCRProcessor(api_key=settings.MISTRAL_API_KEY)
            
            logger.info("Processing document with OCR")
            result = processor.process_document(
                str(file_path), format="html"
            )
            
            html_content = result[0] if isinstance(result, tuple) else result
            logger.info("OCR processing completed successfully")
            
            # Save html_content
            with open(temp_dir / "html_content.html", "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info("Saved HTML content to temp_results")
            
            logger.info("Extracting financial data")
            extractor = FinancialExtractor()
            financial_data = extractor.extract_from_html(html_content)
            logger.info("Financial data extraction completed")
            # Save financial_data
            import json
            with open(temp_dir / "financial_data.json", "w", encoding="utf-8") as f:
                json.dump(financial_data, f, indent=4)
            logger.info("Saved financial data to temp_results")
            
            logger.info("Calculating valuation multiples")
            valuation_multiple = CompanyValuator(financial_data)
            result_valuation = valuation_multiple.calculate_multiples()
            logger.info("Valuation calculations completed")
            
            # Save result_valuation
            with open(temp_dir / "result_valuation.json", "w", encoding="utf-8") as f:
                json.dump(result_valuation, f, indent=4)
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