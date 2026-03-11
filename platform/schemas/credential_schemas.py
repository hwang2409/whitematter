"""AWS credential schemas."""
from pydantic import BaseModel


class AWSCredentialRequest(BaseModel):
    access_key: str
    secret_key: str
    default_region: str = "us-east-1"
    default_instance_type: str = "g4dn.xlarge"


class AWSCredentialResponse(BaseModel):
    has_credentials: bool
    default_region: str | None = None
    default_instance_type: str | None = None
    created_at: str | None = None
