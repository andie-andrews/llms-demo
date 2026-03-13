# Systemd deployment

This guide covers deploying llama.cpp server as a systemd service on the host OS for production use.

## Prerequisites

- Linux host with systemd
- NVIDIA GPU with CUDA drivers installed
- Root/sudo access

## Quick start

### 1. Build llama.cpp on host

```bash
cd /opt
sudo git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
sudo cmake -B build -DGGML_CUDA=ON
sudo cmake --build build --config Release -j$(nproc)
```

> **Note:** Building takes several minutes and compiles CUDA kernels for your GPU.

### 2. Set up directories and models

```bash
# Create model directory
sudo mkdir -p /opt/models
```

Download models using one of these methods:

**Option A: Download on host with HF_HOME set**

```bash
# Set HF_HOME to download directly to /opt/models
sudo HF_HOME=/opt/models python3 /path/to/llms-demo/utils/download_gpt_oss_20b.py
```

**Option B: Copy from dev container**

If you already downloaded models in the dev container:

```bash
# From your host OS (outside container)
sudo cp -r /path/to/llms-demo/models/hugging_face /opt/models/
```

**Option C: Manual download**

Download GGUF files directly from HuggingFace:

```bash
# Example for GPT-OSS-20B
cd /opt/models
sudo wget https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-mxfp4.gguf
```

> **Note:** The download scripts respect the `HF_HOME` environment variable. Without setting it, models download to `~/.cache/huggingface/` by default.

### 3. Create service user

```bash
sudo useradd -r -s /bin/false -d /opt/llama.cpp llama
sudo chown -R llama:llama /opt/llama.cpp
sudo chown -R llama:llama /opt/models
```

### 4. Generate API key

```bash
API_KEY=$(openssl rand -base64 32)
echo "Your API key: $API_KEY"
# Save this key securely!
```

### 5. Install and configure service

```bash
# Copy unit file to systemd
sudo cp utils/llamacpp.service /etc/systemd/system/

# Edit the service file with your API key and model path
sudo nano /etc/systemd/system/llamacpp.service
# Update these lines:
#   - Replace YOUR_API_KEY_HERE with your generated key
#   - Adjust model path
#   - Modify --n-cpu-moe if using MoE model
```

### 6. Enable and start

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable llamacpp.service

# Start the service
sudo systemctl start llamacpp.service

# Check status
sudo systemctl status llamacpp.service
```

## Monitoring

### View logs

```bash
# Follow logs in real-time
sudo journalctl -u llamacpp.service -f

# View last 100 lines
sudo journalctl -u llamacpp.service -n 100

# View logs since boot
sudo journalctl -u llamacpp.service -b
```

### Check metrics

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
    http://localhost:8502/metrics
```

### Service management

```bash
# Stop service
sudo systemctl stop llamacpp.service

# Restart service
sudo systemctl restart llamacpp.service

# Disable service (don't start on boot)
sudo systemctl disable llamacpp.service

# View service status
sudo systemctl status llamacpp.service
```

## Configuration

### Model-specific settings

**GPT-OSS-120B** (120B MoE):
```bash
ExecStart=/opt/llama.cpp/build/bin/llama-server \
    -m /opt/models/hub/models--ggml-org--gpt-oss-120b-GGUF/snapshots/*/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
    --n-gpu-layers 999 \
    --n-cpu-moe 36 \
    -c 0 \
    --flash-attn on \
    --jinja \
    --host 0.0.0.0 \
    --port 8502 \
    --api-key YOUR_API_KEY \
    --metrics \
    --log-timestamps
```

**GPT-OSS-20B** (21B):
```bash
ExecStart=/opt/llama.cpp/build/bin/llama-server \
    -m /opt/models/hub/models--ggml-org--gpt-oss-20b-GGUF/snapshots/*/gpt-oss-20b-mxfp4.gguf \
    --n-gpu-layers 999 \
    -c 8192 \
    --flash-attn on \
    --jinja \
    --host 0.0.0.0 \
    --port 8502 \
    --api-key YOUR_API_KEY \
    --metrics \
    --log-timestamps
```

**Qwen3.5-35B-A3B** (35B MoE):
```bash
ExecStart=/opt/llama.cpp/build/bin/llama-server \
    -m /opt/models/hub/models--noctrex--Qwen3.5-35B-A3B-MXFP4_MOE-GGUF/snapshots/*/Qwen3.5-35B-A3B-MXFP4_MOE_BF16.gguf \
    --n-gpu-layers 999 \
    --n-cpu-moe 40 \
    -c 0 \
    --flash-attn on \
    --jinja \
    --host 0.0.0.0 \
    --port 8502 \
    --api-key YOUR_API_KEY \
    --metrics \
    --log-timestamps
```

### Firewall configuration

If accessing the server remotely, configure your firewall:

```bash
# UFW
sudo ufw allow 8502/tcp

# firewalld
sudo firewall-cmd --permanent --add-port=8502/tcp
sudo firewall-cmd --reload

# iptables
sudo iptables -A INPUT -p tcp --dport 8502 -j ACCEPT
sudo iptables-save | sudo tee /etc/iptables/rules.v4
```

### Reverse proxy (optional)

For production deployments, consider using nginx or Apache as a reverse proxy with SSL:

```nginx
# /etc/nginx/sites-available/llama-server
server {
    listen 443 ssl http2;
    server_name llm.example.com;

    ssl_certificate /etc/letsencrypt/live/llm.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/llm.example.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8502;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # For streaming responses
        proxy_buffering off;
        proxy_cache off;
    }
}
```

## Troubleshooting

### OpenSSL warning during cmake

If cmake prints the following warning, HTTPS support will be disabled:

```
CMake Warning at vendor/cpp-httplib/CMakeLists.txt:150 (message):
  OpenSSL not found, HTTPS support disabled
```

Install the OpenSSL development libraries and re-run cmake:

```bash
# Debian/Ubuntu
sudo apt install -y libssl-dev

# RHEL/Fedora
sudo dnf install -y openssl-devel

# Then re-run cmake
cd /opt/llama.cpp
sudo cmake -B build -DGGML_CUDA=ON
sudo cmake --build build --config Release -j$(nproc)
```

### Service won't start

Check logs for errors:
```bash
sudo journalctl -u llamacpp.service -n 50
```

Common issues:
- **Model file not found**: Verify path in ExecStart
- **Permission denied**: Check ownership with `ls -la /opt/llama.cpp`
- **CUDA errors**: Ensure NVIDIA drivers are installed (`nvidia-smi`)
- **Port already in use**: Check if another process is using port 8502

### Performance issues

Check resource usage:
```bash
# CPU/memory
top -p $(pgrep llama-server)

# GPU
nvidia-smi -l 1

# Detailed GPU stats
nvidia-smi dmon -s u
```

### Update llama.cpp

```bash
# Stop service
sudo systemctl stop llamacpp.service

# Update and rebuild
cd /opt/llama.cpp
sudo git pull
sudo cmake --build build --config Release -j$(nproc)

# Start service
sudo systemctl start llamacpp.service
```

## Security considerations

1. **API key**: Use a strong random key (32+ characters)
2. **Firewall**: Only expose port 8502 to trusted networks
3. **User isolation**: Run as dedicated `llama` user (not root)
4. **File permissions**: Ensure models are readable only by `llama` user
5. **SSL/TLS**: Use reverse proxy with HTTPS for remote access
6. **Rate limiting**: Consider implementing rate limits at proxy level
