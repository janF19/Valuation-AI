from fastapi import APIRouter, UploadFile, HTTPException
from ...ocr.processing import OCRProcessor
from ...financials.extractor import FinancialExtractor
from ...dcf.calculator import DCFCalculator

router = APIRouter()

@router.post("/analyze")
async def analyze_company(file: UploadFile):
    try:
        # Save uploaded file
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Execute workflow
        workflow = ValuationWorkflow()
        report = workflow.execute(file_path)
        
        # Return results
        return {
            "status": "success",
            "valuation": report.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))