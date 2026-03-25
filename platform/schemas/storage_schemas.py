from pydantic import BaseModel


class CreateBucketRequest(BaseModel):
    name: str
    region: str | None = None


class DeleteObjectRequest(BaseModel):
    bucket: str
    key: str


class PresignedUrlRequest(BaseModel):
    bucket: str
    key: str
    expires_in: int = 3600


class PresignedUrlResponse(BaseModel):
    url: str
