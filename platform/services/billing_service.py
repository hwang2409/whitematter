"""Stripe billing service."""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

PLANS = {
    "free": {
        "price": 0,
        "training_runs_per_day": 5,
        "max_models": 3,
        "storage_gb": 1,
        "gpu_access": False,
        "deploy_endpoints": 0,
    },
    "pro": {
        "stripe_price_id": os.environ.get("STRIPE_PRO_PRICE_ID", ""),
        "price": 29,
        "training_runs_per_day": None,  # unlimited
        "max_models": None,  # unlimited
        "storage_gb": 20,
        "gpu_access": False,
        "deploy_endpoints": 1,
    },
    "scale": {
        "stripe_price_id": os.environ.get("STRIPE_SCALE_PRICE_ID", ""),
        "price": 59,
        "training_runs_per_day": None,
        "max_models": None,
        "storage_gb": 100,
        "gpu_access": True,
        "gpu_price_per_hour": 0.99,
        "deploy_endpoints": 5,
    },
}


class BillingService:
    """Manages Stripe subscriptions and usage billing."""

    def __init__(self):
        self._stripe = None

    @property
    def stripe(self):
        if self._stripe is None:
            try:
                import stripe
                stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
                if stripe.api_key:
                    self._stripe = stripe
                else:
                    logger.warning("STRIPE_SECRET_KEY not set, billing disabled")
            except ImportError:
                logger.warning("stripe package not installed")
        return self._stripe

    def create_customer(self, user_id: str, email: str) -> Optional[str]:
        """Create a Stripe customer. Returns customer ID."""
        if not self.stripe:
            return None
        try:
            customer = self.stripe.Customer.create(
                email=email,
                metadata={"whitematter_user_id": user_id},
            )
            return customer.id
        except Exception as e:
            logger.exception("Failed to create Stripe customer: %s", e)
            return None

    def create_checkout_session(
        self, customer_id: str, plan: str, success_url: str, cancel_url: str
    ) -> Optional[str]:
        """Create Stripe Checkout session. Returns session URL."""
        if not self.stripe:
            return None

        plan_config = PLANS.get(plan)
        if not plan_config or plan == "free":
            return None

        price_id = plan_config.get("stripe_price_id")
        if not price_id:
            logger.error("No Stripe price ID for plan: %s", plan)
            return None

        try:
            session = self.stripe.checkout.Session.create(
                customer=customer_id,
                mode="subscription",
                line_items=[{"price": price_id, "quantity": 1}],
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={"plan": plan},
            )
            return session.url
        except Exception as e:
            logger.exception("Failed to create checkout session: %s", e)
            return None

    def get_subscription(self, customer_id: str) -> Optional[dict]:
        """Get current subscription status."""
        if not self.stripe or not customer_id:
            return None
        try:
            subscriptions = self.stripe.Subscription.list(
                customer=customer_id, status="active", limit=1
            )
            if subscriptions.data:
                sub = subscriptions.data[0]
                return {
                    "id": sub.id,
                    "status": sub.status,
                    "plan": sub.metadata.get("plan", "unknown"),
                    "current_period_end": sub.current_period_end,
                }
            return None
        except Exception as e:
            logger.exception("Failed to get subscription: %s", e)
            return None

    def record_gpu_usage(self, customer_id: str, seconds: int) -> bool:
        """Record GPU usage for metered billing (Scale tier)."""
        if not self.stripe or not customer_id:
            return False
        # This would use Stripe Usage Records for metered billing
        # For now, just log it
        logger.info("GPU usage: customer=%s, seconds=%d", customer_id, seconds)
        return True

    def create_portal_session(self, customer_id: str, return_url: str) -> Optional[str]:
        """Create Stripe Customer Portal session. Returns URL."""
        if not self.stripe or not customer_id:
            return None
        try:
            session = self.stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url,
            )
            return session.url
        except Exception as e:
            logger.exception("Failed to create portal session: %s", e)
            return None

    def check_training_limit(self, plan: str, runs_today: int) -> bool:
        """Check if user can start another training run."""
        plan_config = PLANS.get(plan, PLANS["free"])
        limit = plan_config.get("training_runs_per_day")
        if limit is None:
            return True  # unlimited
        return runs_today < limit

    def check_model_limit(self, plan: str, model_count: int) -> bool:
        """Check if user can create another model."""
        plan_config = PLANS.get(plan, PLANS["free"])
        limit = plan_config.get("max_models")
        if limit is None:
            return True
        return model_count < limit
