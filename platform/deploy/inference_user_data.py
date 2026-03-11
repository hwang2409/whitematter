"""EC2 user-data script for inference-only instances (Deploy to API)."""


def generate_inference_user_data(
    platform_url: str,
    deployment_id: str,
    deployment_token: str,
) -> str:
    """Generate bash user-data that installs deps, fetches artifacts, runs inference server."""
    # Token and URL must be safe in the script
    platform_url = platform_url.rstrip("/")
    return f"""#!/bin/bash
set -e
exec > /var/log/whitematter-inference.log 2>&1

echo "=== Whitematter Inference Instance ==="
DEPLOYMENT_ID="{deployment_id}"
TOKEN="{deployment_token}"
PLATFORM="{platform_url}"

# Install Python and Flask
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq python3 python3-pip curl > /dev/null
pip3 install --break-system-packages -q flask pillow numpy

# Fetch artifacts (one-time token)
mkdir -p /opt/whitematter-infer
cd /opt/whitematter-infer
curl -sf -H "Authorization: Bearer $TOKEN" "$PLATFORM/deploy/artifacts/$DEPLOYMENT_ID" -o artifacts.tar
tar xf artifacts.tar
chmod +x infer

# Start inference server in background
python3 server.py &
sleep 2

# Notify platform that we're live
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
ENDPOINT="http://$PUBLIC_IP:8080"
curl -sf -X POST "$PLATFORM/deploy/callback" \\
  -H "Authorization: Bearer $TOKEN" \\
  -H "Content-Type: application/json" \\
  -d "{{\\"endpoint_url\\": \\"$ENDPOINT\\"}}"

echo "Inference endpoint: $ENDPOINT"
"""
