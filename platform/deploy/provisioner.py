"""EC2 provisioner for inference-only instances (Deploy to API)."""

import logging
from datetime import datetime
from typing import Optional

import boto3
from sqlalchemy.orm import Session

from db.auth_models import AWSCredential, Deployment, DeploymentStatus
from services.credential_service import CredentialService
from deploy.inference_user_data import generate_inference_user_data

logger = logging.getLogger(__name__)

# Ubuntu 22.04 LTS (minimal, no GPU stack)
INFERENCE_AMIS = {
    "us-east-1": "ami-0c02fb55b1c4e89c4",
    "us-west-2": "ami-0d705db840ec5f0c5",
}
DEFAULT_INFERENCE_AMI = "ami-0c02fb55b1c4e89c4"


class InferenceProvisioner:
    """Launch and terminate small EC2 instances for model inference."""

    def __init__(self, platform_url: str = "http://localhost:8080", cred_service: Optional[CredentialService] = None):
        self.platform_url = platform_url.rstrip("/")
        self.cred_service = cred_service or CredentialService()

    def _ec2_client(self, cred: AWSCredential):
        return boto3.client(
            "ec2",
            aws_access_key_id=self.cred_service.decrypt(cred.encrypted_access_key),
            aws_secret_access_key=self.cred_service.decrypt(cred.encrypted_secret_key),
            region_name=cred.default_region,
        )

    def launch(
        self,
        db: Session,
        deployment: Deployment,
        cred: AWSCredential,
        platform_url_override: Optional[str] = None,
    ) -> str:
        """
        Launch an EC2 instance for inference. Updates deployment.instance_id and status.
        Returns instance_id.
        """
        url = (platform_url_override or self.platform_url).rstrip("/")
        if not deployment.deployment_token:
            raise ValueError("Deployment has no token")

        user_data = generate_inference_user_data(
            platform_url=url,
            deployment_id=deployment.id,
            deployment_token=deployment.deployment_token,
        )

        ec2 = self._ec2_client(cred)
        ami = INFERENCE_AMIS.get(deployment.region or cred.default_region, DEFAULT_INFERENCE_AMI)

        resp = ec2.run_instances(
            ImageId=ami,
            InstanceType="t3.micro",
            MinCount=1,
            MaxCount=1,
            UserData=user_data,
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"whitematter-infer-{deployment.id}"},
                    {"Key": "whitematter-deployment-id", "Value": deployment.id},
                ],
            }],
            InstanceInitiatedShutdownBehavior="stop",
        )

        instance_id = resp["Instances"][0]["InstanceId"]
        deployment.instance_id = instance_id
        deployment.status = DeploymentStatus.LAUNCHING.value
        deployment.region = deployment.region or cred.default_region
        db.commit()

        logger.info("Launched inference instance %s for deployment %s", instance_id, deployment.id)
        return instance_id

    def terminate(self, db: Session, deployment: Deployment, cred: AWSCredential) -> None:
        """Terminate the EC2 instance and set deployment to TERMINATED."""
        if not deployment.instance_id:
            deployment.status = DeploymentStatus.TERMINATED.value
            db.commit()
            return
        ec2 = self._ec2_client(cred)
        try:
            ec2.terminate_instances(InstanceIds=[deployment.instance_id])
        except Exception as e:
            logger.warning("Terminate instance %s failed: %s", deployment.instance_id, e)
        deployment.status = DeploymentStatus.TERMINATED.value
        deployment.instance_id = None
        deployment.endpoint_url = None
        deployment.updated_at = datetime.utcnow()
        db.commit()
