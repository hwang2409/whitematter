"""GPU provisioner — manages on-demand GPU instances for Scale tier users."""
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Tag for identifying our GPU pool instances
POOL_TAG = "whitematter-gpu-pool"
IDLE_TIMEOUT_MINUTES = 5


class GpuProvisioner:
    """
    Manages on-demand GPU instances on our AWS account.
    Spins up g4dn.xlarge spot instances for Scale tier training.
    Terminates after idle timeout.
    """

    def __init__(self):
        self._ec2_client = None
        self.enabled = bool(
            os.environ.get("AWS_ACCESS_KEY_ID")
            and os.environ.get("AWS_SECRET_ACCESS_KEY")
        )

    @property
    def ec2(self):
        """Lazy EC2 client initialization."""
        if self._ec2_client is None and self.enabled:
            import boto3
            self._ec2_client = boto3.client(
                "ec2",
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
            )
        return self._ec2_client

    def get_warm_instance(self) -> Optional[dict]:
        """Check for an existing warm (running, idle) GPU instance."""
        if not self.ec2:
            return None

        try:
            resp = self.ec2.describe_instances(
                Filters=[
                    {"Name": "tag:pool", "Values": [POOL_TAG]},
                    {"Name": "instance-state-name", "Values": ["running"]},
                    {"Name": "tag:status", "Values": ["idle"]},
                ],
            )
            for reservation in resp.get("Reservations", []):
                for instance in reservation.get("Instances", []):
                    return {
                        "instance_id": instance["InstanceId"],
                        "ip": instance.get("PublicIpAddress")
                            or instance.get("PrivateIpAddress"),
                        "type": instance["InstanceType"],
                    }
        except Exception as e:
            logger.error("Failed to check warm instances: %s", e)

        return None

    def launch_instance(self, job_data: Optional[dict] = None) -> Optional[dict]:
        """
        Launch a new GPU spot instance.
        Returns instance info or None if launch fails.
        """
        if not self.ec2:
            logger.warning("GPU provisioner not enabled (no AWS credentials)")
            return None

        try:
            from byoc.user_data import generate_user_data

            # Generate bootstrap script
            user_data = generate_user_data(
                platform_url=os.environ.get("PLATFORM_URL", "http://localhost:8080"),
                job_id=job_data.get("job_id", "unknown") if job_data else "warmup",
                job_token="gpu-pool-token",
                whitematter_version="0.1.0",
                training_config=job_data or {},
            )

            # Try spot first, fall back to on-demand
            instance_type = os.environ.get("GPU_INSTANCE_TYPE", "g4dn.xlarge")
            ami = os.environ.get("GPU_AMI", "ami-0a0c8eebcdd6dcbd0")

            launch_params = {
                "ImageId": ami,
                "InstanceType": instance_type,
                "MinCount": 1,
                "MaxCount": 1,
                "UserData": user_data,
                "TagSpecifications": [
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Name", "Value": f"whitematter-gpu"},
                            {"Key": "pool", "Value": POOL_TAG},
                            {"Key": "status", "Value": "busy"},
                            {"Key": "launched_at", "Value": datetime.utcnow().isoformat()},
                        ],
                    }
                ],
                "InstanceInitiatedShutdownBehavior": "terminate",
            }

            # Try spot instance
            try:
                launch_params["InstanceMarketOptions"] = {
                    "MarketType": "spot",
                    "SpotOptions": {
                        "SpotInstanceType": "one-time",
                        "InstanceInterruptionBehavior": "terminate",
                    },
                }
                resp = self.ec2.run_instances(**launch_params)
            except Exception:
                logger.info("Spot instance unavailable, using on-demand")
                del launch_params["InstanceMarketOptions"]
                resp = self.ec2.run_instances(**launch_params)

            instance = resp["Instances"][0]
            instance_id = instance["InstanceId"]

            logger.info("Launched GPU instance %s (%s)", instance_id, instance_type)

            # Wait for public IP
            waiter = self.ec2.get_waiter("instance_running")
            waiter.wait(InstanceIds=[instance_id], WaiterConfig={"MaxAttempts": 30})

            desc = self.ec2.describe_instances(InstanceIds=[instance_id])
            inst = desc["Reservations"][0]["Instances"][0]

            return {
                "instance_id": instance_id,
                "ip": inst.get("PublicIpAddress") or inst.get("PrivateIpAddress"),
                "type": instance_type,
            }

        except Exception as e:
            logger.exception("Failed to launch GPU instance: %s", e)
            return None

    def get_or_launch_instance(self, job_data: Optional[dict] = None) -> Optional[dict]:
        """Get a warm instance or launch a new one."""
        warm = self.get_warm_instance()
        if warm:
            # Mark as busy
            try:
                self.ec2.create_tags(
                    Resources=[warm["instance_id"]],
                    Tags=[{"Key": "status", "Value": "busy"}],
                )
            except Exception:
                pass
            return warm

        return self.launch_instance(job_data)

    def mark_idle(self, instance_id: str):
        """Mark an instance as idle (available for reuse)."""
        if not self.ec2:
            return
        try:
            self.ec2.create_tags(
                Resources=[instance_id],
                Tags=[
                    {"Key": "status", "Value": "idle"},
                    {"Key": "idle_since", "Value": datetime.utcnow().isoformat()},
                ],
            )
        except Exception as e:
            logger.error("Failed to mark instance idle: %s", e)

    def terminate_instance(self, instance_id: str):
        """Terminate a specific instance."""
        if not self.ec2:
            return
        try:
            self.ec2.terminate_instances(InstanceIds=[instance_id])
            logger.info("Terminated GPU instance %s", instance_id)
        except Exception as e:
            logger.error("Failed to terminate instance %s: %s", instance_id, e)

    def cleanup_idle_instances(self):
        """Terminate instances that have been idle for too long."""
        if not self.ec2:
            return

        try:
            resp = self.ec2.describe_instances(
                Filters=[
                    {"Name": "tag:pool", "Values": [POOL_TAG]},
                    {"Name": "instance-state-name", "Values": ["running"]},
                    {"Name": "tag:status", "Values": ["idle"]},
                ],
            )

            cutoff = datetime.utcnow() - timedelta(minutes=IDLE_TIMEOUT_MINUTES)

            for reservation in resp.get("Reservations", []):
                for instance in reservation.get("Instances", []):
                    idle_since = None
                    for tag in instance.get("Tags", []):
                        if tag["Key"] == "idle_since":
                            try:
                                idle_since = datetime.fromisoformat(tag["Value"])
                            except ValueError:
                                pass

                    if idle_since and idle_since < cutoff:
                        self.terminate_instance(instance["InstanceId"])

        except Exception as e:
            logger.error("Failed to cleanup idle instances: %s", e)

    def get_pool_status(self) -> dict:
        """Get status of the GPU pool."""
        if not self.ec2:
            return {"enabled": False, "instances": []}

        try:
            resp = self.ec2.describe_instances(
                Filters=[
                    {"Name": "tag:pool", "Values": [POOL_TAG]},
                    {"Name": "instance-state-name", "Values": ["running", "pending"]},
                ],
            )

            instances = []
            for reservation in resp.get("Reservations", []):
                for instance in reservation.get("Instances", []):
                    tags = {t["Key"]: t["Value"] for t in instance.get("Tags", [])}
                    instances.append({
                        "instance_id": instance["InstanceId"],
                        "state": instance["State"]["Name"],
                        "type": instance["InstanceType"],
                        "ip": instance.get("PublicIpAddress"),
                        "status": tags.get("status", "unknown"),
                        "launched_at": tags.get("launched_at"),
                    })

            return {"enabled": True, "instances": instances}

        except Exception as e:
            logger.error("Failed to get pool status: %s", e)
            return {"enabled": True, "instances": [], "error": str(e)}


# Singleton
_gpu_provisioner: Optional[GpuProvisioner] = None


def get_gpu_provisioner() -> GpuProvisioner:
    """Get the global GPU provisioner instance."""
    global _gpu_provisioner
    if _gpu_provisioner is None:
        _gpu_provisioner = GpuProvisioner()
    return _gpu_provisioner
