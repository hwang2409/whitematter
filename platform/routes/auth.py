"""Authentication routes."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session
from db.database import get_db
from db.auth_models import User
from services.auth_service import AuthService
from schemas.auth_schemas import (
    RegisterRequest, LoginRequest, TokenResponse, RefreshRequest, UserResponse,
    GoogleAuthRequest,
)
from auth.dependencies import get_current_user
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
