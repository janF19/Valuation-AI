#!/bin/bash

# Create main project directory
PROJECT_NAME="financial-valuation-system"
mkdir -p $PROJECT_NAME

# Create directory structure
mkdir -p $PROJECT_NAME/{backend,frontend,docker,scripts,docs,api}
mkdir -p $PROJECT_NAME/backend/{src,tests,data,config,migrations}
mkdir -p $PROJECT_NAME/backend/src/{ocr,financials,dcf,reporting,utils,api}
mkdir -p $PROJECT_NAME/backend/src/api/{endpoints,models,schemas,middleware,routers}
mkdir -p $PROJECT_NAME/frontend/{public,src/components,src/hooks,src/services}
mkdir -p $PROJECT_NAME/docker/{compose-files,config}
mkdir -p $PROJECT_NAME/api/{openapi,postman}

# Create base Python files
touch $PROJECT_NAME/backend/requirements.txt
touch $PROJECT_NAME/backend/setup.py
touch $PROJECT_NAME/backend/main.py

# Create core Python modules
touch $PROJECT_NAME/backend/src/__init__.py
touch $PROJECT_NAME/backend/src/ocr/{__init__,processing,parsers}.py
touch $PROJECT_NAME/backend/src/financials/{__init__,extractor,analytics}.py
touch $PROJECT_NAME/backend/src/dcf/{__init__,calculator,assumptions}.py
touch $PROJECT_NAME/backend/src/reporting/{__init__,generator,formatter}.py
touch $PROJECT_NAME/backend/src/utils/{__init__,helpers,validators,auth}.py

# API-specific files
touch $PROJECT_NAME/backend/src/api/__init__.py
touch $PROJECT_NAME/backend/src/api/{config,main}.py
touch $PROJECT_NAME/backend/src/api/models/{__init__,user,company,valuation}.py
touch $PROJECT_NAME/backend/src/api/schemas/{__init__,request,response}.py
touch $PROJECT_NAME/backend/src/api/routers/{__init__,auth,valuations,reports}.py
touch $PROJECT_NAME/backend/src/api/middleware/{__init__,auth,logging}.py

# Configuration files
touch $PROJECT_NAME/backend/config/{__init__,settings,secrets,database}.py

# Test structure
touch $PROJECT_NAME/backend/tests/{__init__,conftest,test_api,test_ocr,test_financials,test_dcf}.py

# Documentation files
touch $PROJECT_NAME/docs/{ARCHITECTURE.md,API_REFERENCE.md,USER_GUIDE.md}
touch $PROJECT_NAME/api/openapi/{specification.yaml,examples}
touch $PROJECT_NAME/api/postman/{collection.json,environment.json}

# Create basic FastAPI app template
cat > $PROJECT_NAME/backend/src/api/main.py << EOF
from fastapi import FastAPI
from .routers import valuations, auth, reports
from .middleware.auth import AuthMiddleware

app = FastAPI(
    title="Financial Valuation API",
    description="API for company valuation analysis",
    version="0.1.0"
)

# Add middleware
app.add_middleware(AuthMiddleware)

# Include routers
app.include_router(auth.router)
app.include_router(valuations.router)
app.include_router(reports.router)

@app.get("/health")
async def health_check():
    return {"status": "ok"}
EOF

# Create sample router
cat > $PROJECT_NAME/backend/src/api/routers/valuations.py << EOF
from fastapi import APIRouter, Depends, HTTPException
from ..schemas.request import ValuationRequest
from ..schemas.response import ValuationResponse
from ..middleware.auth import get_current_user

router = APIRouter(prefix="/api/v1/valuations", tags=["valuations"])

@router.post("/", response_model=ValuationResponse)
async def create_valuation(
    request: ValuationRequest,
    user: dict = Depends(get_current_user)
):
    """
    Submit financial documents for valuation analysis
    """
    try:
        # Implementation would go here
        return {
            "status": "success",
            "dcf_value": 0.0,
            "residual_value": 0.0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
EOF

# Update requirements.txt
cat > $PROJECT_NAME/backend/requirements.txt << EOF
fastapi>=0.68.0
uvicorn>=0.15.0
python-multipart
python-jose[cryptography]
passlib[bcrypt]
python-dotenv
sqlalchemy
pydantic-settings
pandas
numpy
python-docx
requests
python-magic
pdfplumber
openpyxl
EOF

echo "Project structure with API scaffolding created successfully!"