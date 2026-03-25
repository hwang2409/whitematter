import logging
import os
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from db.database import get_db
from db.auth_models import User
from auth.dependencies import get_current_user
from services.billing_service import BillingService, PLANS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/billing", tags=["billing"])
billing_service = BillingService()


@router.post("/checkout")
def create_checkout(
    plan: str,
    interval: str = "month",
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a Stripe Checkout session for Pro or Scale plan."""
    if plan not in ("pro", "scale"):
        raise HTTPException(status_code=400, detail="Invalid plan")
    if interval not in ("month", "year"):
        raise HTTPException(status_code=400, detail="Invalid interval")

    # Ensure user has a Stripe customer ID
    if not user.stripe_customer_id:
        customer_id = billing_service.create_customer(user.id, user.email)
        if not customer_id:
            raise HTTPException(status_code=503, detail="Billing not available")
        user.stripe_customer_id = customer_id
        db.commit()

    base_url = os.environ.get("FRONTEND_URL", "http://localhost:3000")
    url = billing_service.create_checkout_session(
        customer_id=user.stripe_customer_id,
        plan=plan,
        success_url=f"{base_url}/pricing?billing=success",
        cancel_url=f"{base_url}/pricing?billing=cancelled",
        interval=interval,
    )
    if not url:
        raise HTTPException(status_code=503, detail="Could not create checkout session")

    return {"url": url}


@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """Handle Stripe webhooks."""
    if not billing_service.stripe:
        raise HTTPException(status_code=503, detail="Billing not configured")

    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")

    if not webhook_secret:
        raise HTTPException(status_code=503, detail="Webhook secret not configured")

    try:
        event = billing_service.stripe.Webhook.construct_event(
            payload, sig, webhook_secret
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook error: {str(e)}")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        customer_id = session.get("customer")
        plan = session.get("metadata", {}).get("plan", "pro")

        user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
        if user:
            user.plan = plan
            db.commit()
            logger.info("User %s upgraded to %s", user.id, plan)

    elif event["type"] == "customer.subscription.deleted":
        sub = event["data"]["object"]
        customer_id = sub.get("customer")

        user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
        if user:
            user.plan = "free"
            db.commit()
            logger.info("User %s downgraded to free", user.id)

    return {"status": "ok"}


@router.get("/status")
def billing_status(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get current billing status."""
    plan = getattr(user, "plan", "free")
    plan_config = PLANS.get(plan, PLANS["free"])

    result = {
        "plan": plan,
        "plan_details": {
            "price": plan_config.get("price", 0),
            "training_runs_per_day": plan_config.get("training_runs_per_day"),
            "max_models": plan_config.get("max_models"),
            "storage_gb": plan_config.get("storage_gb"),
            "gpu_access": plan_config.get("gpu_access", False),
            "deploy_endpoints": plan_config.get("deploy_endpoints", 0),
        },
        "subscription": None,
    }

    if user.stripe_customer_id:
        result["subscription"] = billing_service.get_subscription(user.stripe_customer_id)

    return result


@router.post("/portal")
def customer_portal(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create Stripe Customer Portal session."""
    if not user.stripe_customer_id:
        raise HTTPException(status_code=400, detail="No billing account")

    base_url = os.environ.get("FRONTEND_URL", "http://localhost:3000")
    url = billing_service.create_portal_session(
        customer_id=user.stripe_customer_id,
        return_url=f"{base_url}/settings",
    )
    if not url:
        raise HTTPException(status_code=503, detail="Could not create portal session")

    return {"url": url}


@router.get("/usage")
def billing_usage(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get usage stats for the current user."""
    from db.models import Model, Dataset
    from db.chat_models import Conversation

    models_count = db.query(Model).filter(Model.user_id == user.id).count()
    datasets_count = db.query(Dataset).filter(Dataset.user_id == user.id).count()
    conversations_count = db.query(Conversation).filter(Conversation.user_id == user.id).count()

    return {
        "models_count": models_count,
        "datasets_count": datasets_count,
        "conversations_count": conversations_count,
    }
