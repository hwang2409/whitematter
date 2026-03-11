"""AWS credential management routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db.database import get_db
from db.auth_models import User, AWSCredential
from auth.dependencies import get_current_user
from services.credential_service import CredentialService
from schemas.credential_schemas import AWSCredentialRequest, AWSCredentialResponse

router = APIRouter(prefix="/credentials", tags=["credentials"])


def _cred_service() -> CredentialService:
    return CredentialService()


@router.post("/aws", response_model=AWSCredentialResponse, status_code=201)
def store_credentials(req: AWSCredentialRequest,
                      user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    cred_service = _cred_service()
    existing = db.query(AWSCredential).filter(AWSCredential.user_id == user.id).first()
    if existing:
        raise HTTPException(status_code=409, detail="Credentials already exist. Use PUT to update.")

    cred = AWSCredential(
        user_id=user.id,
        encrypted_access_key=cred_service.encrypt(req.access_key),
        encrypted_secret_key=cred_service.encrypt(req.secret_key),
        default_region=req.default_region,
        default_instance_type=req.default_instance_type,
    )
    db.add(cred)
    db.commit()
    db.refresh(cred)

    return AWSCredentialResponse(
        has_credentials=True,
        default_region=cred.default_region,
        default_instance_type=cred.default_instance_type,
        created_at=cred.created_at.isoformat(),
    )


@router.get("/aws", response_model=AWSCredentialResponse)
def get_credentials(user: User = Depends(get_current_user),
                     db: Session = Depends(get_db)):
    cred = db.query(AWSCredential).filter(AWSCredential.user_id == user.id).first()
    if not cred:
        return AWSCredentialResponse(has_credentials=False)
    return AWSCredentialResponse(
        has_credentials=True,
        default_region=cred.default_region,
        default_instance_type=cred.default_instance_type,
        created_at=cred.created_at.isoformat(),
    )


@router.put("/aws", response_model=AWSCredentialResponse)
def update_credentials(req: AWSCredentialRequest,
                      user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    cred_service = _cred_service()
    cred = db.query(AWSCredential).filter(AWSCredential.user_id == user.id).first()
    if not cred:
        raise HTTPException(status_code=404, detail="No credentials found")

    cred.encrypted_access_key = cred_service.encrypt(req.access_key)
    cred.encrypted_secret_key = cred_service.encrypt(req.secret_key)
    cred.default_region = req.default_region
    cred.default_instance_type = req.default_instance_type
    db.commit()
    db.refresh(cred)

    return AWSCredentialResponse(
        has_credentials=True,
        default_region=cred.default_region,
        default_instance_type=cred.default_instance_type,
        created_at=cred.created_at.isoformat(),
    )


@router.delete("/aws", status_code=204)
def delete_credentials(user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    cred = db.query(AWSCredential).filter(AWSCredential.user_id == user.id).first()
    if not cred:
        raise HTTPException(status_code=404, detail="No credentials found")
    db.delete(cred)
    db.commit()
