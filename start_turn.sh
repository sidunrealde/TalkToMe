#!/bin/bash
# Start coturn TURN server for WSL2 WebRTC relay
# Listens on 127.0.0.1 (shared loopback between Windows and WSL)

set -e

CONF=/tmp/turnserver.conf
LOG=/tmp/coturn.log

# Stop any existing coturn
sudo systemctl stop coturn 2>/dev/null || true
sudo pkill -f turnserver 2>/dev/null || true
sleep 1

# Write minimal config
cat > "$CONF" << 'EOF'
listening-ip=127.0.0.1
listening-port=3478
relay-ip=127.0.0.1
min-port=49152
max-port=49252
realm=talktome
user=talktome:talktome123
lt-cred-mech
no-tls
no-dtls
no-cli
log-file=/tmp/coturn.log
simple-log
verbose
EOF

echo "Starting coturn on 127.0.0.1:3478..."
sudo turnserver -c "$CONF" &
sleep 2

# Verify it's running
if pgrep -f turnserver > /dev/null; then
    echo "✓ coturn TURN server running on 127.0.0.1:3478"
    echo "  Credentials: talktome / talktome123"
    echo "  Log: /tmp/coturn.log"
    tail -5 "$LOG" 2>/dev/null || true
else
    echo "✗ coturn failed to start"
    cat "$LOG" 2>/dev/null || true
    exit 1
fi
