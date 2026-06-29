#!/usr/bin/env bash
# Serve the goban + WebGPU-engine demo over http (WASM needs http, not file://).
# Open the printed URL in a WebGPU-capable browser (desktop Chrome/Edge).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/web/demo"
PORT="${PORT:-8080}"
if [ ! -f "$DIR/engine.js" ]; then
  echo "engine.js not built yet — run scripts/build-web.sh first."; exit 1
fi
echo "==> Serving $DIR"
echo "==> Open http://localhost:$PORT/  in Chrome/Edge (needs WebGPU)."
cd "$DIR"
exec python3 -c "
import http.server, socketserver
H = http.server.SimpleHTTPRequestHandler
H.extensions_map['.wasm'] = 'application/wasm'
H.extensions_map['.js'] = 'text/javascript'
socketserver.TCPServer(('', $PORT), H).serve_forever()
"
