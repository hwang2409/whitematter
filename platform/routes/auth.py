"""Authentication routes."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session
from db.database import get_db
from db.auth_models import User
from services.auth_service import AuthService
from schemas.auth_schemas import (
    RegisterRequest, LoginRequest, TokenResponse, RefreshRequest, UserResponse,
    GoogleAuthRequest, ChangePasswordRequest,
)
from auth.dependencies import get_current_user
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])
auth_service = AuthService()


@router.post("/register", response_model=TokenResponse, status_code=201)
@limiter.limit("5/minute")
def register(request: Request, req: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=req.email,
        password_hash=auth_service.hash_password(req.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return TokenResponse(
        access_token=auth_service.create_access_token(user.id, user.email),
        refresh_token=auth_service.create_refresh_token(user.id),
    )


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
def login(request: Request, req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not user.password_hash:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not auth_service.verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return TokenResponse(
        access_token=auth_service.create_access_token(user.id, user.email),
        refresh_token=auth_service.create_refresh_token(user.id),
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh(req: RefreshRequest, db: Session = Depends(get_db)):
    try:
        payload = auth_service.decode_token(req.refresh_token)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")

    user = db.query(User).filter(User.id == payload["sub"]).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return TokenResponse(
        access_token=auth_service.create_access_token(user.id, user.email),
        refresh_token=auth_service.create_refresh_token(user.id),
    )


@router.post("/google", response_model=TokenResponse)
@limiter.limit("10/minute")
def google_auth(request: Request, req: GoogleAuthRequest, db: Session = Depends(get_db)):
    """Authenticate with Google OAuth. Creates account if new user."""
    import httpx

    try:
        resp = httpx.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {req.access_token}"},
            timeout=10.0,
        )
        resp.raise_for_status()
        google_info = resp.json()
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Google token")

    google_id = google_info.get("sub")
    email = google_info.get("email")
    avatar_url = google_info.get("picture")

    if not google_id or not email:
        raise HTTPException(status_code=401, detail="Could not get Google profile")

    # Find or create user
    user = db.query(User).filter(User.google_id == google_id).first()
    if not user:
        # Check if email already exists (link accounts)
        user = db.query(User).filter(User.email == email).first()
        if user:
            # Link Google ID to existing account
            user.google_id = google_id
            if avatar_url:
                user.avatar_url = avatar_url
            user.oauth_provider = "google"
            user.oauth_id = google_id
        else:
            user = User(
                email=email,
                google_id=google_id,
                avatar_url=avatar_url,
                oauth_provider="google",
                oauth_id=google_id,
            )
            db.add(user)
    else:
        # Update avatar on subsequent logins
        if avatar_url:
            user.avatar_url = avatar_url

    db.commit()
    db.refresh(user)

    return TokenResponse(
        access_token=auth_service.create_access_token(user.id, user.email),
        refresh_token=auth_service.create_refresh_token(user.id),
    )


@router.get("/me", response_model=UserResponse)
def get_me(user: User = Depends(get_current_user)):
    return UserResponse(
        id=user.id,
        email=user.email,
        oauth_provider=user.oauth_provider,
        avatar_url=user.avatar_url,
        plan=getattr(user, "plan", "free"),
        created_at=user.created_at.isoformat(),
    )


@router.post("/change-password")
@limiter.limit("5/minute")
def change_password(
    request: Request,
    req: ChangePasswordRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Change the user's password."""
    if not user.password_hash:
        raise HTTPException(status_code=400, detail="OAuth accounts cannot change password")
    if not auth_service.verify_password(req.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    user.password_hash = auth_service.hash_password(req.new_password)
    db.commit()
    return {"message": "Password changed successfully"}


@router.get("/export")
def export_data(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Export all user data as JSON."""
    from db.chat_models import Conversation, ConversationMessage
    from db.models import Model, Dataset, TrainingHistory

    # Conversations with messages
    convs = db.query(Conversation).filter(Conversation.user_id == user.id).all()
    conversations_data = []
    for conv in convs:
        messages = (
            db.query(ConversationMessage)
            .filter(ConversationMessage.conversation_id == conv.id)
            .order_by(ConversationMessage.created_at)
            .all()
        )
        conversations_data.append({
            "id": conv.id,
            "title": conv.title,
            "phase": conv.phase,
            "is_starred": conv.is_starred,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "message_type": m.message_type,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in messages
            ],
        })

    # Models with training history
    models = db.query(Model).filter(Model.user_id == user.id).all()
    models_data = []
    for model in models:
        history = (
            db.query(TrainingHistory)
            .filter(TrainingHistory.model_id == model.id)
            .order_by(TrainingHistory.epoch)
            .all()
        )
        models_data.append({
            "id": model.id,
            "name": model.name,
            "architecture_name": model.architecture_name,
            "status": model.status,
            "best_accuracy": float(model.best_accuracy) if model.best_accuracy else None,
            "final_loss": float(model.final_loss) if model.final_loss else None,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "training_history": [
                {"epoch": h.epoch, "loss": float(h.loss), "accuracy": float(h.accuracy)}
                for h in history
            ],
        })

    # Datasets (metadata only)
    datasets = db.query(Dataset).filter(Dataset.user_id == user.id).all()
    datasets_data = [
        {
            "id": ds.id,
            "name": ds.name,
            "data_type": ds.data_type,
            "format": ds.format,
            "status": ds.status,
            "created_at": ds.created_at.isoformat() if ds.created_at else None,
        }
        for ds in datasets
    ]

    # AWS credentials (presence only — keys are encrypted)
    from db.auth_models import AWSCredential
    aws_cred = db.query(AWSCredential).filter(AWSCredential.user_id == user.id).first()
    credentials_data = None
    if aws_cred:
        credentials_data = {
            "has_credentials": True,
            "default_region": aws_cred.default_region,
            "provider": aws_cred.provider,
            "endpoint_url": aws_cred.endpoint_url,
        }

    return JSONResponse(content={
        "user": {"email": user.email, "plan": getattr(user, "plan", "free")},
        "conversations": conversations_data,
        "models": models_data,
        "datasets": datasets_data,
        "aws_credentials": credentials_data,
    })
