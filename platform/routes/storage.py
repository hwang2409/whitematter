from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from db.database import get_db
from db.auth_models import User, AWSCredential
from auth.dependencies import get_current_user
from services.s3_service import S3Service
from schemas.storage_schemas import (
    CreateBucketRequest, DeleteObjectRequest, PresignedUrlRequest, PresignedUrlResponse
)

router = APIRouter(prefix="/storage", tags=["storage"])
s3_service = S3Service()


def _get_aws_cred(user: User, db: Session) -> AWSCredential:
    cred = db.query(AWSCredential).filter(AWSCredential.user_id == user.id).first()
    if not cred:
        raise HTTPException(status_code=400, detail="No AWS credentials configured")
    return cred


def _endpoint_url(cred: AWSCredential) -> str | None:
    return getattr(cred, "endpoint_url", None) or None


@router.get("/buckets")
def list_buckets(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    cred = _get_aws_cred(user, db)
    return s3_service.list_buckets(
        cred.encrypted_access_key, cred.encrypted_secret_key, cred.default_region,
        endpoint_url=_endpoint_url(cred),
    )


@router.post("/buckets", status_code=201)
def create_bucket(req: CreateBucketRequest,
                  user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    cred = _get_aws_cred(user, db)
    region = req.region or cred.default_region
    s3_service.create_bucket(
        cred.encrypted_access_key, cred.encrypted_secret_key, region, req.name,
        endpoint_url=_endpoint_url(cred),
    )
    return {"name": req.name, "region": region}


@router.get("/buckets/{bucket}/objects")
def list_objects(bucket: str, prefix: str = "",
                 user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    cred = _get_aws_cred(user, db)
    return s3_service.list_objects(
        cred.encrypted_access_key, cred.encrypted_secret_key, cred.default_region, bucket, prefix,
        endpoint_url=_endpoint_url(cred),
    )


@router.post("/upload")
async def upload_file(bucket: str = Form(...), key: str = Form(...),
                      file: UploadFile = File(...),
                      user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    cred = _get_aws_cred(user, db)
    s3_service.upload_file(
        cred.encrypted_access_key, cred.encrypted_secret_key, cred.default_region,
        bucket, key, file.file,
        endpoint_url=_endpoint_url(cred),
    )
    return {"bucket": bucket, "key": key, "size": file.size}


@router.delete("/objects", status_code=204)
def delete_object(req: DeleteObjectRequest,
                  user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    cred = _get_aws_cred(user, db)
    s3_service.delete_object(
        cred.encrypted_access_key, cred.encrypted_secret_key, cred.default_region, req.bucket, req.key,
        endpoint_url=_endpoint_url(cred),
    )


@router.post("/presigned-url", response_model=PresignedUrlResponse)
def get_presigned_url(req: PresignedUrlRequest,
                      user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    cred = _get_aws_cred(user, db)
    url = s3_service.generate_presigned_url(
        cred.encrypted_access_key, cred.encrypted_secret_key, cred.default_region,
        req.bucket, req.key, req.expires_in,
        endpoint_url=_endpoint_url(cred),
    )
    return PresignedUrlResponse(url=url)
