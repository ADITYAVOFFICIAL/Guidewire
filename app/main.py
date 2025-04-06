import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from app.core.config import settings
from app.core.logging_config import setup_logging
# Correct import for the api router based on typical structure
from app.api.v1.api import api_router as api_v1_router # Assuming api.py aggregates endpoints

# Setup logging FIRST
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title=settings.APP_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    version="0.1.0" # Add a version
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)
# Include API router
app.include_router(api_v1_router, prefix=settings.API_V1_STR)

# Root endpoint
@app.get("/", tags=["Root"], summary="Root endpoint for service status")
async def read_root():
    """Returns a welcome message indicating the service is running."""
    return {"message": f"Welcome to the {settings.APP_NAME}"}

# --- Optional: Add global exception handlers ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Request validation error: {exc.errors()}", exc_info=False) # Don't need full stack trace usually
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception during request to {request.url}: {exc}", exc_info=True) # Log full trace for unexpected errors
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred."},
    )

# --- Startup/Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    # You could explicitly check service readiness here if not using Depends
    # if prediction_service.model is None:
    #    logger.critical("PREDICTION MODEL NOT LOADED ON STARTUP!")
    # if settings.ACTION_MODE == "automate" and not k8s_service.is_available():
    #    logger.warning("KUBERNETES CLIENT NOT AVAILABLE ON STARTUP - AUTOMATE MODE MAY FAIL")
    logger.info(f"Application '{settings.APP_NAME}' started successfully.")
    logger.info(f"Prediction Threshold: {settings.PREDICTION_THRESHOLD}")
    logger.info(f"Action Mode: {settings.ACTION_MODE}")
    logger.info(f"Target Namespace: {settings.TARGET_NAMESPACE}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown...")
    # Add any cleanup tasks here if needed
    logger.info("Application shutdown complete.")

# --- Run with Uvicorn (for local development) ---
# This block is typically used for running directly with `python app/main.py`
# It's often removed or conditionalized when using `uvicorn app.main:app ...` command
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "app.main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True, # Enable reload for development
#         log_level=settings.LOG_LEVEL.lower()
#      )