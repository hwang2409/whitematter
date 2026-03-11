"""BYOC training routes (launch, status, stop) + callback routes."""
import hashlib
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from db.database import get_db
from db.auth_models import User, AWSCredential, ByocTrainingJob, ByocJobStatus
from auth.dependencies import get_current_user
from byoc.provisioner import ByocProvisioner
from schemas.byoc_schemas import (
    LaunchRequest, JobStatusResponse, HeartbeatRequest,
    LogRequest, MetricsRequest, CompleteRequest, FailRequest,
)

router = APIRouter(tags=["training"])
provisioner = ByocProvisioner()

# Store logs in memory (could move to DB later)
_job_logs: dict[str, list[str]] = {}


# --- User-facing routes (require JWT auth) ---

@router.post("/training/launch", response_model=JobStatusResponse, status_code=201)
def launch_training(req: LaunchRequest,
                    user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    cred = db.query(AWSCredential).filter(AWSCredential.user_id == user.id).first()
    if not cred:
        raise HTTPException(status_code=400, detail="No AWS credentials configured")

    job = ByocTrainingJob(
        user_id=user.id,
        model_config=req.model_config_data,
        dataset_config=req.dataset_config,
        instance_type=req.instance_type,
        region=req.region or cred.default_region,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        provisioner.launch_job(db, cred, job)
    except Exception as e:
        job.status = ByocJobStatus.FAILED.value
        job.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to launch instance: {e}")

    return _job_to_response(job)


@router.get("/training/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str,
                   user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    job = db.query(ByocTrainingJob).filter(
        ByocTrainingJob.id == job_id, ByocTrainingJob.user_id == user.id
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_response(job)


@router.get("/training/{job_id}/logs")
def get_job_logs(job_id: str,
                 user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    job = db.query(ByocTrainingJob).filter(
        ByocTrainingJob.id == job_id, ByocTrainingJob.user_id == user.id
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"logs": _job_logs.get(job_id, [])}


@router.post("/training/{job_id}/stop")
def stop_training(job_id: str,
                  user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    job = db.query(ByocTrainingJob).filter(
        ByocTrainingJob.id == job_id, ByocTrainingJob.user_id == user.id
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    cred = db.query(AWSCredential).filter(AWSCredential.user_id == user.id).first()
    if cred:
        provisioner.stop_job(db, cred, job)

    return _job_to_response(job)


# --- Callback routes (require job token auth) ---

def _verify_job_token(job_id: str, authorization: str, db: Session) -> ByocTrainingJob:
    """Verify one-time job token from GPU instance."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization")
    token = authorization[7:]
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    job = db.query(ByocTrainingJob).filter(ByocTrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.job_token_hash != token_hash:
        raise HTTPException(status_code=401, detail="Invalid job token")
    return job


@router.post("/training/{job_id}/heartbeat")
def heartbeat(job_id: str,
              authorization: str = Header(..., alias="Authorization"),
              db: Session = Depends(get_db)):
    _verify_job_token(job_id, authorization, db)
    db.commit()
    return {"status": "ok"}


@router.post("/training/{job_id}/log")
def append_log(job_id: str, req: LogRequest,
               authorization: str = Header(..., alias="Authorization"),
               db: Session = Depends(get_db)):
    _verify_job_token(job_id, authorization, db)
    if job_id not in _job_logs:
        _job_logs[job_id] = []
    _job_logs[job_id].append(req.message)
    return {"status": "ok"}


@router.post("/training/{job_id}/metrics")
def report_metrics(job_id: str, req: MetricsRequest,
                   authorization: str = Header(..., alias="Authorization"),
                   db: Session = Depends(get_db)):
    job = _verify_job_token(job_id, authorization, db)
    metrics = job.metrics or {}
    if "epochs" not in metrics:
        metrics["epochs"] = []
    metrics["epochs"].append({
        "epoch": req.epoch, "loss": req.loss, "accuracy": req.accuracy
    })
    job.metrics = metrics
    db.commit()
    return {"status": "ok"}


@router.post("/training/{job_id}/complete")
def complete_job(job_id: str, req: CompleteRequest,
                authorization: str = Header(..., alias="Authorization"),
                db: Session = Depends(get_db)):
    job = _verify_job_token(job_id, authorization, db)
    job.status = ByocJobStatus.COMPLETED.value
    job.completed_at = datetime.utcnow()
    job.metrics = req.metrics
    db.commit()
    return {"status": "ok"}


@router.post("/training/{job_id}/fail")
def fail_job(job_id: str, req: FailRequest,
             authorization: str = Header(..., alias="Authorization"),
             db: Session = Depends(get_db)):
    job = _verify_job_token(job_id, authorization, db)
    job.status = ByocJobStatus.FAILED.value
    job.completed_at = datetime.utcnow()
    job.error_message = req.error_message
    db.commit()
    return {"status": "ok"}


def _job_to_response(job: ByocTrainingJob) -> JobStatusResponse:
    return JobStatusResponse(
        id=job.id,
        status=job.status,
        instance_type=job.instance_type,
        instance_id=job.instance_id,
        region=job.region,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        metrics=job.metrics,
        error_message=job.error_message,
    )
