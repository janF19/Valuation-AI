import requests
import os
from backend.src.ocr.mistral_processor import MistralOCRProcessor
from typing import Optional, Union, Tuple, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class OCRProcessor:
    def __init__(self, api_key: str):
        self.mistral_processor = MistralOCRProcessor(api_key)

    def process_document(
        self, 
        file_path: str, 
        output_dir: Optional[str] = None,
        format: str = "html",  # Can be "html", "markdown", or "json"
        include_images_int_text: bool = False,  # New parameter to control image handling
        save_images_separately: bool = False
    ) -> Union[str, Tuple[str, Dict]]:
        """
        Central OCR processing function that coordinates different OCR processors.
        Currently supports Mistral OCR, but can be extended for other providers.
        
        Args:
            file_path: Path to the document to process
            output_dir: Optional directory to save results and images
            format: Output format ("html", "markdown", or "json")
            include_images_int_text: If True, includes images inline in the output content
            save_images_separately: If True and output_dir is provided, saves images as separate files
            
        Returns:
            If images are included: Tuple of (content, image_map)
            Otherwise: Content string
        """
        # Convert format parameter to Mistral processor options
        json_output = format == "json"
        html_output = format == "html"
        
        try:
            # Process with Mistral OCR
            result = self.mistral_processor.process_document(
                file_path=file_path,
                output_dir=output_dir,  # Pass through for image extraction
                json_output=json_output,
                html_output=html_output,
                inline_images= include_images_int_text,
                extract_images= True if output_dir and save_images_separately else False,
                silent=True
            )
            
            # Save the result if output_dir is provided
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                if isinstance(result, tuple):
                    content, _ = result
                else:
                    content = result
                
                # Determine output filename
                if html_output:
                    output_file = output_path / "index.html"
                elif json_output:
                    output_file = output_path / "result.json"
                else:
                    output_file = output_path / "README.md"
                
                # Save the file
                output_file.write_text(content, encoding='utf-8')
            
            return result
            
        except Exception as e:
            raise OCRException(f"OCR processing failed: {str(e)}")

    def process_documents(
        self,
        file_paths: list[str],
        output_dir: Optional[str] = None,
        format: str = "html",
        include_images_int_text: bool = False,
        save_images_separately: bool = False,
        max_workers: int = 3
    ) -> Dict[str, Union[str, Tuple[str, Dict]]]:
        """
        Process multiple documents in parallel using ThreadPoolExecutor.
        
        Args:
            file_paths: List of paths to documents to process
            output_dir: Optional directory to save results and images
            format: Output format ("html", "markdown", or "json")
            include_images_int_text: If True, includes images inline in the output content
            save_images_separately: If True and output_dir is provided, saves images as separate files
            max_workers: Maximum number of parallel processes
            
        Returns:
            Dictionary mapping file paths to their processing results
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Modified to use the same output directory for all files
            future_to_path = {
                executor.submit(
                    self.process_document,
                    file_path=file_path,
                    output_dir=output_dir,  # Now using the same output_dir for all files
                    format=format,
                    include_images_int_text=include_images_int_text,
                    save_images_separately=save_images_separately
                ): file_path
                for file_path in file_paths
            }
            
            # Process completed futures as they finish
            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    results[file_path] = future.result()
                except Exception as e:
                    results[file_path] = f"Error processing {file_path}: {str(e)}"
        
        return results

class OCRException(Exception):
    pass