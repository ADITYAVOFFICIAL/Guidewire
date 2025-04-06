# app/models/remediation.py
from pydantic import BaseModel
from typing import Optional, List

class RemediationAction(BaseModel):
    action_type: str
    reason: str
    recommendation: str
    command: Optional[str] = None
    executed: Optional[bool] = False
    error: Optional[str] = None

class RemediationResponse(BaseModel):
    pod_name: str
    namespace: str
    failure_probability: float
    triggered: bool # Did probability exceed threshold?
    actions_determined: List[RemediationAction] = []
    action_mode: str