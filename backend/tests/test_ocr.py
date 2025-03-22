import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.src.ocr.processing import OCRProcessor
from backend.config.settings import settings  # Import settings object

def test_ocr():
    # Initialize the processor
    processor = OCRProcessor(api_key=settings.MISTRAL_API_KEY)

    # Use project_root to create absolute path to the PDF
    pdf_path = project_root / "backend" / "tests" / "testFinancials" / "vz_2023_vitkovice.pdf"
    
    result = processor.process_document(
        str(pdf_path),
        format="html"
    )

    # Unpack the result (it's a tuple of content and image_map)
    html_content = result[0] if isinstance(result, tuple) else result

    # Create results directory if it doesn't exist
    results_dir_path = project_root / "backend" / "tests" / "results"
    
    
    results_dir_path.mkdir(parents=True, exist_ok=True)

    # Save the HTML file (with inline images)
    output_file = results_dir_path / "output2.html"
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(html_content)

def test_multiple_ocr():
    # Initialize the processor
    processor = OCRProcessor(api_key=settings.MISTRAL_API_KEY)

    # Define multiple PDF paths
    pdf_paths = [
        str(project_root / "backend" / "tests" / "testFinancials" / "vz_2023_isotra.pdf"),
        str(project_root / "backend" / "tests" / "testFinancials" / "vz_2022_isotra.pdf"),
        str(project_root / "backend" / "tests" / "testFinancials" / "vz_2021_isotra.pdf"),
        # Add more test PDFs as needed
    ]
    
    # Process multiple documents
    #Process without output_dir since by default when i specify it wants to save images
    results = processor.process_documents(
        file_paths=pdf_paths,
        format="html",
        max_workers=2
    )

    # Create results directory
    results_dir = project_root / "backend" / "tests" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save results manually
    for pdf_path, result in results.items():
        # Skip if there was an error
        if isinstance(result, str) and result.startswith("Error"):
            print(f"Error processing {pdf_path}: {result}")
            continue
            
        # Unpack the result
        html_content = result[0] if isinstance(result, tuple) else result
        
        # Save the result with a unique name based on the PDF filename
        output_filename = f"output_{Path(pdf_path).stem}.html"
        output_file = results_dir / output_filename
        
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(html_content)
            print(f"Saved results for {pdf_path} to {output_file}")

if __name__ == "__main__":
    test_ocr()
