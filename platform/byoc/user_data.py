"""EC2 user-data script generation for BYOC training."""
import json


def generate_user_data(
    platform_url: str,
    job_id: str,
    job_token: str,
    whitematter_version: str,
    training_config: dict,
) -> str:
    """Generate bash user-data script for EC2 instance."""
    config_json = json.dumps(training_config).replace("\\", "\\\\").replace("'", "\\'")
    return f"""#!/bin/bash
set -e

# Log everything
exec > /var/log/whitematter-training.log 2>&1

echo "=== Whitematter BYOC Training ==="
echo "Job ID: {job_id}"
echo "Platform: {platform_url}"

# Install whitematter
pip install whitematter=={whitematter_version}

# Create training script
cat > /tmp/train.py << 'TRAIN_SCRIPT'
import json
import sys
import time
import urllib.request

PLATFORM_URL = "{platform_url}"
JOB_ID = "{job_id}"
JOB_TOKEN = "{job_token}"
CONFIG = json.loads('''{config_json}''')

def api_call(path, data=None):
    url = f"{{PLATFORM_URL}}{{path}}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        url, data=body,
        headers={{"Authorization": f"Bearer {{JOB_TOKEN}}", "Content-Type": "application/json"}}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"API call failed: {{e}}")
        return None

def heartbeat():
    api_call(f"/training/{{JOB_ID}}/heartbeat")

def log_line(msg):
    api_call(f"/training/{{JOB_ID}}/log", {{"message": msg}})

def report_metrics(epoch, loss, accuracy):
    api_call(f"/training/{{JOB_ID}}/metrics", {{
        "epoch": epoch, "loss": loss, "accuracy": accuracy
    }})

def complete(metrics):
    api_call(f"/training/{{JOB_ID}}/complete", {{"metrics": metrics}})

def fail(error):
    api_call(f"/training/{{JOB_ID}}/fail", {{"error_message": error}})

try:
    import whitematter as wm
    heartbeat()
    log_line("whitematter loaded, starting training...")

    # TODO: Build model from CONFIG, download data from S3, train
    # This will be filled in when pybind11 bindings are extended

    log_line("Training complete")
    complete({{"status": "done"}})
except Exception as e:
    fail(str(e))
    sys.exit(1)
TRAIN_SCRIPT

python /tmp/train.py

# Self-terminate
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
"""
