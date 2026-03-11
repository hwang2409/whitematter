"""AWS and S3-compatible credential schemas."""
from pydantic import BaseModel, Field


class AWSCredentialRequest(BaseModel):
    access_key: str
    secret_key: str
    default_region: str = "us-east-1"
    default_instance_type: str = "g4dn.xlarge"
    endpoint_url: str | None = Field(None, max_length=512)
    provider: str | None = Field(None, pattern="^(aws|r2|b2|custom)$")


class AWSCredentialResponse(BaseModel):
    has_credentials: bool
    default_region: str | None = None
    default_instance_type: str | None = None
    endpoint_url: str | None = None
    provider: str | None = None
    created_at: str | None = None
