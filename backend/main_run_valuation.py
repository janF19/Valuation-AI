from backend.main import ValuationWorkflow
from pathlib import Path
import os

if __name__ == "__main__":
    # Get the backend directory path
    backend_dir = Path(__file__).parent
    data_dir = backend_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Define the input file path
    input_file = data_dir / "vz_2023_isotra.pdf"
    
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file.absolute()}")
        print("Please place your PDF file in the backend/data directory")
        exit(1)
        
    workflow = ValuationWorkflow()
    try:
        print("Starting valuation workflow...")
        report = workflow.execute(str(input_file))
        print("Workflow completed successfully!")
    except Exception as e:
        print(f"Workflow failed: {str(e)}")