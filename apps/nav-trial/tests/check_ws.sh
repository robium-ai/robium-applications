#!/usr/bin/env bash
# Foxglove WebSocket handshake probe: expect HTTP 101 with the foxglove
# subprotocol. Usage: check_ws.sh http://localhost:8765 (or https://...).
# foxglove_bridge >= 3.x (Foxglove-SDK based) expects subprotocol
# "foxglove.sdk.v1" — the pre-SDK "foxglove.websocket.v1" is rejected 400
# "Missing expected sec-websocket-protocol header" (verified 2026-07-12
# against ros-jazzy-foxglove-bridge 3.4.1 by grepping libfoxglove.so).
set -uo pipefail
BASE="${1:?usage: check_ws.sh <base-url>}"
RESP=$(curl -s -i -N --http1.1 --max-time 15 \
  -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: $(openssl rand -base64 16)" \
  -H "Sec-WebSocket-Protocol: foxglove.sdk.v1" \
  "$BASE/?session=${2:-smoke}" | head -5)
echo "$RESP" | head -1
if echo "$RESP" | head -1 | grep -q " 101 "; then
  echo "WS HANDSHAKE OK"
else
  echo "WS HANDSHAKE FAIL"; exit 1
fi
