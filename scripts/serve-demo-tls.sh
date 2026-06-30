#!/usr/bin/env bash
# Serve the goban + WebGPU-engine demo over HTTPS (self-signed).
#
# WebGPU is a Secure-Context API: navigator.gpu is only exposed over https:// or
# localhost. Plain http://<LAN-IP> is NOT a secure context, so WebGPU is hidden.
# This serves over TLS so a remote browser gets a secure context (accept the
# self-signed cert warning once -> "Proceed").
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/web/demo"
CERTDIR="$ROOT/web/.certs"
PORT="${PORT:-8443}"

if [ ! -f "$DIR/engine.js" ]; then
  echo "engine.js not built yet — run scripts/build-web.sh first."; exit 1
fi

mkdir -p "$CERTDIR"
if [ ! -f "$CERTDIR/cert.pem" ]; then
  echo "==> generating self-signed cert"
  IPS=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.' | sed 's/^/IP:/' | paste -sd, -)
  openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -keyout "$CERTDIR/key.pem" -out "$CERTDIR/cert.pem" \
    -subj "/CN=saigo-webgpu-demo" \
    -addext "subjectAltName=${IPS},DNS:localhost,IP:127.0.0.1" 2>/dev/null
fi

echo "==> Serving $DIR over HTTPS on :$PORT (all interfaces)"
echo "==> Open https://<this-host-ip>:$PORT/ in Chrome/Edge; accept the cert warning."
cd "$DIR"
exec python3 -c "
import http.server, socketserver, ssl
class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      '.wasm': 'application/wasm', '.js': 'text/javascript'}
    def end_headers(self):
        # COOP/COEP -> cross-origin isolation -> SharedArrayBuffer -> wasm pthreads
        # (needed for the threaded WebGPU path; harmless for the single-thread demo).
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain('$CERTDIR/cert.pem', '$CERTDIR/key.pem')
httpd = socketserver.TCPServer(('', $PORT), H)
httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
print('listening on :$PORT (https)')
httpd.serve_forever()
"
