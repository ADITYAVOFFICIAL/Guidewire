# app/api/v1/__init__.py
from fastapi import APIRouter
from .endpoints import remediation

api_router = APIRouter()
api_router.include_router(remediation.router, prefix="/remediation", tags=["Remediation"])