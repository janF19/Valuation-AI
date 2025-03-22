
# from .routers import valuations, auth, reports
# from .middleware.auth import AuthMiddleware

# app = FastAPI(
#     title="Financial Valuation API",
#     description="API for company valuation analysis",
#     version="0.1.0"
# )

# # Add middleware
# app.add_middleware(AuthMiddleware)

# # Include routers
# app.include_router(auth.router)
# app.include_router(valuations.router)
# app.include_router(reports.router)

# @app.get("/health")
# async def health_check():
#     return {"status": "ok"}
