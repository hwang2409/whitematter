import hashlib
import secrets
from datetime import datetime
import boto3
from sqlalchemy.orm import Session
from db.auth_models import AWSCredential, ByocTrainingJob, ByocJobStatus
from services.credential_service import CredentialService
from byoc.user_data import generate_user_data

# AWS Deep Learning AMI (Ubuntu 22.04) — updated per-region
DEEP_LEARNING_AMIS = {
    "us-east-1": "ami-0a0c8eebcdd6dcbd0",
    "us-west-2": "ami-0a0c8eebcdd6dcbd0",
}
DEFAULT_AMI = "ami-0a0c8eebcdd6dcbd0"


class ByocProvisioner:
    def __init__(self, platform_url: str = "http://localhost:8080",
                 cred_service: CredentialService | None = None):
        self.platform_url = platform_url
        self.cred_service = cred_service or CredentialService()

    def _get_ec2_client(self, cred: AWSCredential):
        return boto3.client(
            "ec2",
            aws_access_key_id=self.cred_service.decrypt(cred.encrypted_access_key),
            aws_secret_access_key=self.cred_service.decrypt(cred.encrypted_secret_key),
            region_name=cred.default_region,
        )

    def launch_job(self, db: Session, cred: AWSCredential, job: ByocTrainingJob) -> str:
        """Launch a GPU EC2 instance for training. Returns instance ID."""
        ec2 = self._get_ec2_client(cred)

        job_token = secrets.token_urlsafe(48)
        job.job_token_hash = hashlib.sha256(job_token.encode()).hexdigest()
        job.status = ByocJobStatus.LAUNCHING.value
        db.commit()

        user_data = generate_user_data(
            platform_url=self.platform_url,
            job_id=job.id,
            job_token=job_token,
        )

        ami = DEEP_LEARNING_AMIS.get(cred.default_region, DEFAULT_AMI)

        resp = ec2.run_instances(
            ImageId=ami,
            InstanceType=job.instance_type or cred.default_instance_type,
            MinCount=1,
            MaxCount=1,
            UserData=user_data,
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"whitematter-job-{job.id}"},
                    {"Key": "whitematter-job-id", "Value": job.id},
                ],
            }],
            InstanceInitiatedShutdownBehavior="terminate",
        )

        instance_id = resp["Instances"][0]["InstanceId"]
        job.instance_id = instance_id
        job.status = ByocJobStatus.RUNNING.value
        job.started_at = datetime.utcnow()
        job.region = cred.default_region
        db.commit()

        return instance_id

    def stop_job(self, db: Session, cred: AWSCredential, job: ByocTrainingJob):
        """Terminate the EC2 instance for a job."""
        if not job.instance_id:
            return
        ec2 = self._get_ec2_client(cred)
        ec2.terminate_instances(InstanceIds=[job.instance_id])
        job.status = ByocJobStatus.STOPPED.value
        job.completed_at = datetime.utcnow()
        db.commit()

    def check_instance_status(self, cred: AWSCredential, instance_id: str) -> str:
        """Check if an EC2 instance is still running."""
        ec2 = self._get_ec2_client(cred)
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        state = resp["Reservations"][0]["Instances"][0]["State"]["Name"]
        return state
