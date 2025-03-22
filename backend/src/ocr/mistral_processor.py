from pathlib import Path
import json
import base64
import re
from typing import Optional, Union, Tuple, Dict
from mistralai import Mistral
from mistralai import DocumentURLChunk
import markdown

class MistralOCRProcessor:
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
        
    def process_document(
        self,
        file_path: Union[str, Path],
        *,
        output_dir: Optional[str] = None,  # Only used for image extraction
        model: str = "mistral-ocr-latest",
        json_output: bool = False,
        html_output: bool = False,
        inline_images: bool = False,
        extract_images: bool = False,
        silent: bool = True
    ) -> Union[str, Tuple[str, Dict]]:
        """
        Process a PDF document using Mistral's OCR.
        File saving is handled by the parent OCRProcessor.
        
        Args:
            file_path: Path to the PDF file
            output_dir: Directory to save output and extracted images
            model: Mistral OCR model to use
            json_output: Return raw JSON instead of markdown
            html_output: Convert markdown to HTML
            inline_images: Include images as data URIs
            extract_images: Extract images as separate files
            silent: Suppress progress messages
            
        Returns:
            If extract_images or inline_images is True:
                Tuple of (content, image_map)
            Otherwise:
                Content string (JSON, HTML, or markdown)
        """
        # Validate options
        if output_dir and not (inline_images or extract_images):
            raise ValueError("output_dir requires either inline_images or extract_images")
        if json_output and output_dir:
            raise ValueError("JSON output is not supported with output_dir")
        if extract_images and not output_dir:
            raise ValueError("extract_images requires output_dir")
        if inline_images and extract_images:
            raise ValueError("Cannot specify both inline_images and extract_images")

        pdf_file = Path(file_path)
        uploaded_file = None
        
        try:
            if not silent:
                print(f"Processing {pdf_file.name}...")
                
            # Upload file
            uploaded_file = self.client.files.upload(
                file={
                    "file_name": pdf_file.stem,
                    "content": pdf_file.read_bytes(),
                },
                purpose="ocr",
            )

            signed_url = self.client.files.get_signed_url(
                file_id=uploaded_file.id, 
                expiry=1
            )

            # Process with OCR
            pdf_response = self.client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model=model,
                include_image_base64=inline_images or extract_images,
            )

            response_dict = json.loads(pdf_response.model_dump_json())
            
            # Handle images
            image_map = {}
            if extract_images and output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                for page in response_dict.get("pages", []):
                    for img in page.get("images", []):
                        if "id" in img and "image_base64" in img:
                            image_data = img["image_base64"]
                            if image_data.startswith("data:image/"):
                                image_data = image_data.split(",", 1)[1]
                            
                            image_filename = img["id"]
                            image_path = output_path / image_filename
                            
                            with open(image_path, "wb") as f:
                                f.write(base64.b64decode(image_data))
                            
                            image_map[image_filename] = image_filename
                            
            elif inline_images:
                for page in response_dict.get("pages", []):
                    for img in page.get("images", []):
                        if "id" in img and "image_base64" in img:
                            image_id = img["id"]
                            image_data = img["image_base64"]
                            if not image_data.startswith("data:"):
                                ext = image_id.split(".")[-1].lower() if "." in image_id else "jpeg"
                                image_data = f"data:image/{ext};base64,{image_data}"
                            image_map[image_id] = image_data

            # Generate output content
            if json_output:
                result = json.dumps(response_dict, indent=4)
            else:
                markdown_contents = [
                    page.get("markdown", "") for page in response_dict.get("pages", [])
                ]
                markdown_text = "\n\n".join(markdown_contents)

                # Handle image references
                for img_id, img_src in image_map.items():
                    markdown_text = re.sub(
                        r"!\[(.*?)\]\(" + re.escape(img_id) + r"\)",
                        r"![\1](" + img_src + r")",
                        markdown_text,
                    )

                if html_output:
                    md = markdown.Markdown(extensions=["tables"])
                    html_content = md.convert(markdown_text)
                    result = self._create_html_document(html_content)
                else:
                    result = markdown_text

            # Handle output
            if output_dir:
                output_file = Path(output_dir)
                if html_output:
                    output_file = output_file / "index.html"
                else:
                    output_file = output_file / "README.md"
                output_file.write_text(result, encoding='utf-8')
                if not silent:
                    print(f"Results saved to {output_file}")

            if inline_images or extract_images:
                return result, image_map
            return result

        finally:
            if uploaded_file:
                try:
                    self.client.files.delete(file_id=uploaded_file.id)
                    if not silent:
                        print("Temporary file deleted.")
                except Exception as e:
                    if not silent:
                        print(f"Warning: Could not delete temporary file: {str(e)}")

    def _create_html_document(self, content: str) -> str:
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Result</title>
    <style>
        body {{ 
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0 auto;
            max-width: 800px;
            padding: 20px;
        }}
        img {{ max-width: 100%; height: auto; }}
        h1, h2, h3 {{ margin-top: 1.5em; }}
        p {{ margin: 1em 0; }}
    </style>
</head>
<body>
{content}
</body>
</html>"""

class MistralOCRException(Exception):
    pass