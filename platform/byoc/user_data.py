def generate_user_data(
    platform_url: str,
    job_id: str,
    job_token: str,
) -> str:
    return f"""#!/bin/bash
set -e
exec > /var/log/whitematter-training.log 2>&1

echo "=== Whitematter BYOC Training ==="
echo "Job ID: {job_id}"

# Install build deps (safety — DL AMI usually has these)
apt-get update -qq && apt-get install -y -qq build-essential > /dev/null 2>&1 || true

# Create working directory
mkdir -p /opt/wm-train && cd /opt/wm-train

# Download training artifacts
curl -sf "{platform_url}/training/{job_id}/artifacts?token={job_token}" -o artifacts.tar
tar xf artifacts.tar --strip-components=1

# Build library and training binary
make train 2>&1 | tail -20
if [ $? -ne 0 ]; then
    python3 -c "import urllib.request,json; urllib.request.urlopen(urllib.request.Request('{platform_url}/training/{job_id}/fail', data=json.dumps({{'error_message':'build failed'}}).encode(), headers={{'Authorization':'Bearer {job_token}','Content-Type':'application/json'}}))"
    INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
    REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
    exit 1
fi

# Create training monitor script
cat > monitor.py << 'MONITOR_EOF'
import subprocess, sys, time, json, threading, re
import urllib.request

PLATFORM_URL = "PLACEHOLDER_URL"
JOB_ID = "PLACEHOLDER_JOB_ID"
JOB_TOKEN = "PLACEHOLDER_TOKEN"

def api_call(path, data=None, method="POST"):
    url = f"{{PLATFORM_URL}}{{path}}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        url, data=body, method=method,
        headers={{"Authorization": f"Bearer {{JOB_TOKEN}}", "Content-Type": "application/json"}}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"API error: {{e}}")
        return None

def heartbeat():
    api_call(f"/training/{{JOB_ID}}/heartbeat")

def report_metrics(epoch, loss, accuracy):
    api_call(f"/training/{{JOB_ID}}/metrics", {{"epoch": epoch, "loss": loss, "accuracy": accuracy}})

def complete(metrics):
    api_call(f"/training/{{JOB_ID}}/complete", {{"metrics": metrics}})

def fail(error):
    api_call(f"/training/{{JOB_ID}}/fail", {{"error_message": error}})

def upload_weights(filepath):
    \"\"\"Upload model weights via multipart POST.\"\"\"
    import mimetypes
    boundary = "----WhiteMatterUpload"
    with open(filepath, "rb") as f:
        file_data = f.read()
    body = (
        f"--{{boundary}}\\r\\n"
        f'Content-Disposition: form-data; name="file"; filename="model.bin"\\r\\n'
        f"Content-Type: application/octet-stream\\r\\n\\r\\n"
    ).encode() + file_data + f"\\r\\n--{{boundary}}--\\r\\n".encode()
    req = urllib.request.Request(
        f"{{PLATFORM_URL}}/training/{{JOB_ID}}/weights",
        data=body,
        headers={{
            "Authorization": f"Bearer {{JOB_TOKEN}}",
            "Content-Type": f"multipart/form-data; boundary={{boundary}}",
        }},
    )
    urllib.request.urlopen(req, timeout=120)

# Heartbeat thread
done = False
def heartbeat_loop():
    while not done:
        heartbeat()
        time.sleep(30)

t = threading.Thread(target=heartbeat_loop, daemon=True)
t.start()

try:
    heartbeat()

    proc = subprocess.Popen(
        ["./train", "./data", "./model.bin"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )

    final_metrics = {{}}
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        print(line)

        # Parse: "Epoch  1 | Loss: 0.1234 | Test Acc: 45.67% | Best: 45.67%"
        if "Epoch" in line and "Loss:" in line:
            try:
                parts = line.split("|")
                epoch = int(parts[0].split()[-1])
                loss = float(parts[1].split(":")[1].strip())
                acc = 0.0
                for p in parts:
                    if "Test Acc:" in p or "Acc:" in p:
                        acc = float(re.search(r"[\\d.]+", p.split(":")[1]).group())
                        break
                report_metrics(epoch, loss, acc)
                final_metrics = {{"epoch": epoch, "loss": loss, "accuracy": acc}}
            except (ValueError, IndexError):
                pass

    proc.wait()
    done = True

    if proc.returncode == 0:
        upload_weights("./model.bin")
        complete(final_metrics)
    else:
        fail(f"Training exited with code {{proc.returncode}}")

except Exception as e:
    done = True
    fail(str(e))
    sys.exit(1)
MONITOR_EOF

# Replace placeholders with actual values
sed -i "s|PLACEHOLDER_URL|{platform_url}|g" monitor.py
sed -i "s|PLACEHOLDER_JOB_ID|{job_id}|g" monitor.py
sed -i "s|PLACEHOLDER_TOKEN|{job_token}|g" monitor.py

python3 monitor.py

# Self-terminate
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
"""
