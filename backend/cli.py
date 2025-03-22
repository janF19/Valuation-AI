import argparse
from pathlib import Path
import logging
from backend.main import ValuationWorkflow

def setup_logging():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler('valuation.log')  # File output
        ]
    )
    return logging.getLogger(__name__)

def validate_file_path(file_path: str) -> Path:
    path = Path(file_path)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File {file_path} does not exist")
    if not path.suffix.lower() == '.pdf':
        raise argparse.ArgumentTypeError(f"File {file_path} is not a PDF file")
    return path

def main():
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(
        description='Financial Valuation CLI - Process PDF financial statements and generate valuation reports'
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to the input PDF file containing financial statements'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Path where to save the output report (default: input_file_report.docx)'
    )
    
    args = parser.parse_args()
    
    try:
        input_path = validate_file_path(args.input)
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_report.docx"
        
        logger.info(f"Processing file: {input_path}")
        
        # Run the workflow with the specified output path
        workflow = ValuationWorkflow()
        report = workflow.execute(str(input_path), str(output_path))
        logger.info("Workflow completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()