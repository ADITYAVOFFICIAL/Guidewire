# app/api/v1/api.py
from fastapi import APIRouter
from app.api.v1.endpoints import remediation

api_router = APIRouter()

# Include routers from endpoint modules
api_router.include_router(remediation.router, prefix="/remediation", tags=["Remediation"])

# Add other endpoint routers here if you create more
# from app.api.v1.endpoints import other_endpoint
# api_router.include_router(other_endpoint.router, prefix="/other", tags=["Other"])