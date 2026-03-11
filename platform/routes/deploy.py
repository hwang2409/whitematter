"""
One-click Deploy to API: deploy a trained model to EC2 inference instance.
"""

import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth.dependencies import get_current_user
from db import get_db
from db.auth_models import User, AWSCredential, Deployment, DeploymentStatus
from services.deployment_service import DeploymentService
from deploy.provisioner import InferenceProvisioner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/deploy", tags=["deploy"])
deployment_service = DeploymentService()


# --- Schemas ---

class DeployRequest(BaseModel):
    model_id: str
    target: str = "ec2"
    region: Optional[str] = "us-east-1"
    platform_url: Optional[str] = None  # Override for artifact fetch / callback (e.g. public URL)


class DeployResponse(BaseModel):
    deployment_id: str
    status: str
    message: str


class DeploymentResponse(BaseModel):
    id: str
    model_id: str
    target_type: str
    status: str
    instance_id: Optional[str]
    endpoint_url: Optional[str]
    region: Optional[str]
    error_message: Optional[str]
    created_at: str
    updated_at: Optional[str]


class CallbackBody(BaseModel):
    deployment_id: str
    endpoint_url: str


def _deployment_to_response(d: Deployment) -> dict:
    return {
        "id": d.id,
        "model_id": d.model_id,
        "target_type": d.target_type,
        "status": d.status,
        "instance_id": d.instance_id,
        "endpoint_url": d.endpoint_url,
        "region": d.region,
        "error_message": d.error_message,
        "created_at": d.created_at.isoformat() if d.created_at else None,
        "updated_at": d.updated_at.isoformat() if d.updated_at else None,
    }


# --- User-facing routes (require JWT) ---

@router.post("", response_model=DeployResponse, status_code=202)
async def create_deployment(
    body: DeployRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Start a one-click deploy of a completed model to EC2. Returns deployment_id for polling."""
    if body.target != "ec2":
        raise HTTPException(status_code=400, detail="Only target=ec2 is supported")

    deployment = deployment_service.create_deployment(
        user_id=user.id,
        model_id=body.model_id,
        region=body.region or "us-east-1",
    )
    if not deployment:
        raise HTTPException(
            status_code=400,
            detail="Model not found, not completed, or not deployable (custom image models only)",
        )

    # Re-attach deployment to this request's session for provisioner
    d = db.query(Deployment).filter_by(id=deployment.id).first()
    if not d:
        raise HTTPException(status_code=500, detail="Deployment record not found")

    # User must have AWS credentials
    cred = db.query(AWSCredential).filter_by(user_id=user.id).first()
    if not cred:
        raise HTTPException(
            status_code=400,
            detail="Add AWS credentials in Settings to deploy to EC2",
        )

    platform_url = body.platform_url or os.environ.get("PLATFORM_PUBLIC_URL", "http://localhost:8080")
    provisioner = InferenceProvisioner(platform_url=platform_url)

    try:
        provisioner.launch(db, d, cred, platform_url_override=body.platform_url)
    except Exception as e:
        logger.exception("Launch inference EC2 failed for deployment %s", d.id)
        deployment_service.set_failed(d.id, str(e))
        raise HTTPException(status_code=500, detail=f"Failed to launch instance: {e}")

    return DeployResponse(
        deployment_id=d.id,
        status=d.status,
        message="Instance launching. Poll GET /deployments/{id} for endpoint_url when status is live.",
    )


@router.get("/deployments", response_model=list)
async def list_deployments(
    model_id: Optional[str] = Query(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List deployments for the current user, optionally filtered by model_id."""
    query = db.query(Deployment).filter_by(user_id=user.id)
    if model_id:
        query = query.filter_by(model_id=model_id)
    deployments = query.order_by(Deployment.created_at.desc()).all()
    return [_deployment_to_response(d) for d in deployments]


@router.get("/deployments/{deployment_id}", response_model=dict)
async def get_deployment(
    deployment_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get deployment status and endpoint_url when live."""
    d = db.query(Deployment).filter_by(id=deployment_id, user_id=user.id).first()
    if not d:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return _deployment_to_response(d)


@router.delete("/deployments/{deployment_id}", status_code=204)
async def terminate_deployment(
    deployment_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Terminate the EC2 instance and mark deployment as terminated."""
    d = db.query(Deployment).filter_by(id=deployment_id, user_id=user.id).first()
    if not d:
        raise HTTPException(status_code=404, detail="Deployment not found")

    cred = db.query(AWSCredential).filter_by(user_id=user.id).first()
    if cred:
        provisioner = InferenceProvisioner()
        provisioner.terminate(db, d, cred)
    else:
        d.status = DeploymentStatus.TERMINATED.value
        d.instance_id = None
        d.endpoint_url = None
        db.commit()

    return None


# --- Unauthenticated: artifact fetch (token in query) and callback (token in header) ---

@router.get("/artifacts/{deployment_id}")
async def get_artifacts(
    deployment_id: str,
    token: str = Query(..., description="One-time deployment token"),
):
    """EC2 instance calls this to download the artifact tarball. Token in query."""
    tarball = deployment_service.get_artifacts(deployment_id, token)
    if not tarball:
        raise HTTPException(status_code=404, detail="Invalid deployment or token")
    return Response(
        content=tarball,
        media_type="application/x-tar",
        headers={"Content-Disposition": "attachment; filename=artifacts.tar"},
    )


@router.post("/callback")
async def deployment_callback(
    body: CallbackBody,
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    EC2 instance calls this after starting the inference server.
    Header: Authorization: Bearer <deployment_token>
    Body: { "deployment_id": "...", "endpoint_url": "http://<public-ip>:8080" }
    """
    auth_header = authorization or ""
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization")
    token = auth_header[7:].strip()

    ok = deployment_service.set_live(body.deployment_id, token, body.endpoint_url)
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {"status": "ok"}