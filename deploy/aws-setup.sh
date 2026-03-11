#!/bin/bash
set -euo pipefail

echo "=== Whitematter Platform Setup ==="
echo "This script sets up whitematter on a fresh Ubuntu 22.04 EC2 instance."
echo ""

# Install Docker
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo "Docker installed. You may need to log out and back in for group changes."
fi

# Install Docker Compose plugin
if ! docker compose version &> /dev/null; then
    echo "Installing Docker Compose plugin..."
    sudo apt-get update && sudo apt-get install -y docker-compose-plugin
fi

# Clone or update repo
if [ ! -d "/opt/whitematter" ]; then
    echo "Cloning whitematter..."
    sudo git clone https://github.com/hwang2409/whitematter.git /opt/whitematter
    sudo chown -R $USER:$USER /opt/whitematter
else
    echo "Updating whitematter..."
    cd /opt/whitematter && git pull
fi

cd /opt/whitematter

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env

    # Generate secrets
    JWT_SECRET=$(openssl rand -base64 32)
    POSTGRES_PASSWORD=$(openssl rand -base64 16)
    FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null || openssl rand -base64 32)

    sed -i "s|generate-a-random-secret-min-32-chars|$JWT_SECRET|" .env
    sed -i "s|changeme|$POSTGRES_PASSWORD|g" .env
    sed -i "s|generate-a-fernet-key|$FERNET_KEY|" .env

    # Get public IP
    PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "localhost")
    sed -i "s|http://your-domain-or-ip|http://$PUBLIC_IP|" .env

    echo "Generated .env with random secrets."
    echo "Edit /opt/whitematter/.env to add your ANTHROPIC_API_KEY if needed."
fi

# Build and start
echo "Building and starting whitematter..."
docker compose -f docker-compose.prod.yml up -d --build

echo ""
echo "=== Setup Complete ==="
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "localhost")
echo "Platform: http://$PUBLIC_IP"
echo "API:      http://$PUBLIC_IP:8080/health"
echo ""
echo "Logs:     docker compose -f docker-compose.prod.yml logs -f"
echo "Stop:     docker compose -f docker-compose.prod.yml down"
