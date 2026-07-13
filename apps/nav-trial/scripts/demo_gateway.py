#!/usr/bin/env python3
"""Demo session gateway — single process on $PORT (8765).

Routes:
  * WebSocket upgrade (any path)  -> raw byte tunnel to the bridge :8766.
      First tunnel claims the instance for the request's ?session=UUID;
      a second concurrent viewer gets 503 (Cloud Run routes their retry to
      a fresh instance because this one is busy).
  * GET  /status?session=UUID     -> 200 JSON (contract in the plan header);
      409 if the instance is claimed by a different session.
  * POST /shutdown?session=UUID   -> 200 + SIGTERM PID 1 (container exits);
      403 on session mismatch.

stdlib only; runs alongside ros2 launch inside the demo container.
"""
import asyncio
import base64
import contextvars
import hashlib
import json
import os
import pty
import signal
import time
import urllib.request
from urllib.parse import parse_qs, quote, urlsplit

# Per-request CORS origin (async-safe: each connection task has its own copy).
_origin = contextvars.ContextVar('origin', default='https://robium.org')

PORT = int(os.environ.get('PORT', '8765'))
BRIDGE = ('127.0.0.1', 8766)
STATUS_PATH = '/tmp/demo_status.json'
SESSION_SECONDS = 1800
FLEET_BUDGET = 5  # keep in sync with Cloud Run --max-instances
FLEET_CACHE_S = 30
WORKSPACE_ROOT = '/ws'  # fs API jail
WS_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'


def ws_accept(head):
    key = ''
    for line in head.split('\r\n'):
        if line.lower().startswith('sec-websocket-key:'):
            key = line.split(':', 1)[1].strip()
    return base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()


def ws_frame(data: bytes, opcode=0x2) -> bytes:
    n = len(data)
    if n < 126:
        header = bytes([0x80 | opcode, n])
    elif n < 65536:
        header = bytes([0x80 | opcode, 126]) + n.to_bytes(2, 'big')
    else:
        header = bytes([0x80 | opcode, 127]) + n.to_bytes(8, 'big')
    return header + data


def ws_unframe(buf: bytes):
    """(payload|None, remaining, closed). None payload => need more bytes."""
    if len(buf) < 2:
        return None, buf, False
    b1 = buf[1]
    masked = b1 & 0x80
    ln = b1 & 0x7f
    idx = 2
    if ln == 126:
        if len(buf) < 4:
            return None, buf, False
        ln = int.from_bytes(buf[2:4], 'big'); idx = 4
    elif ln == 127:
        if len(buf) < 10:
            return None, buf, False
        ln = int.from_bytes(buf[2:10], 'big'); idx = 10
    opcode = buf[0] & 0x0f
    need = idx + (4 if masked else 0) + ln
    if len(buf) < need:
        return None, buf, False
    if masked:
        mask = buf[idx:idx + 4]; idx += 4
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(buf[idx:idx + ln]))
    else:
        payload = buf[idx:idx + ln]
    if opcode == 0x8:
        return None, buf[need:], True
    return payload, buf[need:], False


def safe_path(p):
    full = os.path.realpath(os.path.join(WORKSPACE_ROOT, (p or '').lstrip('/')))
    if full != WORKSPACE_ROOT and not full.startswith(WORKSPACE_ROOT + '/'):
        return None
    return full


async def pty_bridge(reader, writer, head):
    """Terminate the WebSocket here and bridge it to a bash PTY."""
    writer.write((f'HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n'
                  f'Connection: Upgrade\r\nSec-WebSocket-Accept: {ws_accept(head)}\r\n\r\n').encode())
    await writer.drain()
    pid, fd = pty.fork()
    if pid == 0:
        os.environ['TERM'] = 'xterm-256color'
        os.chdir(WORKSPACE_ROOT)
        os.execvp('bash', ['bash'])
    loop = asyncio.get_event_loop()

    async def pty_to_ws():
        while True:
            try:
                data = await loop.run_in_executor(None, os.read, fd, 4096)
            except OSError:
                break
            if not data:
                break
            writer.write(ws_frame(data))
            await writer.drain()

    async def ws_to_pty():
        buf = b''
        while True:
            chunk = await reader.read(4096)
            if not chunk:
                return
            buf += chunk
            while True:
                payload, buf, closed = ws_unframe(buf)
                if closed:
                    return
                if payload is None:
                    break
                os.write(fd, payload)

    try:
        await asyncio.gather(pty_to_ws(), ws_to_pty())
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            writer.close()
        except Exception:
            pass


async def logs_stream(reader, writer, head):
    """Read-only: push new status-file log lines as ws text frames."""
    writer.write((f'HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n'
                  f'Connection: Upgrade\r\nSec-WebSocket-Accept: {ws_accept(head)}\r\n\r\n').encode())
    await writer.drain()
    sent = 0
    try:
        while True:
            log = read_status_file().get('log', [])
            if len(log) != sent:
                text = '\r\n'.join(log[sent:]) + '\r\n'
                writer.write(ws_frame(text.encode(), opcode=0x1))
                await writer.drain()
                sent = len(log)
            await asyncio.sleep(1.0)
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass

state = {'session': None, 'tunnel_open': False, 'claimed_at': None}
fleet_cache = {'at': 0.0, 'running': None}


def fleet_running():
    # Live count disabled in v4: the demo container runs as a zero-IAM-role
    # service account (security — a stolen metadata token must grant nothing)
    # and egress is locked down, so it cannot query Cloud Monitoring. The page
    # shows the static budget. JSON shape kept stable so a future
    # separately-permissioned endpoint can restore the live number with no
    # frontend change.
    return None


ALLOWED_ORIGINS = ('https://robium.org',)


def cors_origin(head):
    """Reflect an allowed Origin (exact-origin CORS with credentials — ACAO:*
    is invalid alongside credentials). Prod is robium.org; localhost:* is
    allowed so `npm run dev` can iterate the frontend against this gateway."""
    for line in head.split('\r\n'):
        if line.lower().startswith('origin:'):
            o = line.split(':', 1)[1].strip()
            if o in ALLOWED_ORIGINS or o.startswith('http://localhost:') or o.startswith('http://127.0.0.1:'):
                return o
    return ALLOWED_ORIGINS[0]


def http_response(status, body, extra=''):
    # Connection: close is load-bearing behind Cloud Run's proxy (pools
    # keep-alive; a silent close = edge 503 "malformed"). Exact-origin CORS +
    # credentials so the affinity cookie rides on same-site fetches.
    return (f'HTTP/1.1 {status}\r\nContent-Type: application/json\r\n'
            f'Content-Length: {len(body)}\r\n'
            f'Access-Control-Allow-Origin: {_origin.get()}\r\n'
            f'Access-Control-Allow-Credentials: true\r\n'
            f'Connection: close\r\n'
            f'{extra}\r\n{body}').encode()


def read_status_file():
    try:
        with open(STATUS_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {'start': time.time(), 'ready': False, 'rtf': None,
                'nodes': 0, 'log': ['stack booting…']}


async def pipe(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(reader, writer):
    try:
        raw = await asyncio.wait_for(reader.readuntil(b'\r\n\r\n'), timeout=30)
    except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError):
        writer.close()
        return
    head = raw.decode('latin1')
    request_line = head.split('\r\n', 1)[0]
    parts = request_line.split(' ')
    if len(parts) < 2:
        writer.close()
        return
    method, target = parts[0], parts[1]
    url = urlsplit(target)
    session = parse_qs(url.query).get('session', [None])[0]
    is_upgrade = 'upgrade: websocket' in head.lower()
    _origin.set(cors_origin(head))

    if method == 'OPTIONS':  # CORS preflight
        writer.write(('HTTP/1.1 204 No Content\r\n'
                      f'Access-Control-Allow-Origin: {_origin.get()}\r\n'
                      'Access-Control-Allow-Credentials: true\r\n'
                      'Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n'
                      'Access-Control-Allow-Headers: content-type\r\n'
                      'Connection: close\r\nContent-Length: 0\r\n\r\n').encode())
        await writer.drain(); writer.close(); return

    # Session-guard for the control/fs/pty/logs surfaces (not the Foxglove
    # tunnel, which claims). Foreign session on a claimed instance → reject.
    def guarded_ok():
        return not state['session'] or session == state['session']

    if is_upgrade and url.path == '/pty':
        if not guarded_ok():
            writer.write(http_response('403 Forbidden', json.dumps({'error': 'forbidden'})))
            await writer.drain(); writer.close(); return
        state['session'] = session or state['session'] or 'anonymous'
        state['claimed_at'] = state['claimed_at'] or time.time()
        await pty_bridge(reader, writer, head)
        return

    if is_upgrade and url.path == '/logs':
        if not guarded_ok():
            writer.write(http_response('403 Forbidden', json.dumps({'error': 'forbidden'})))
            await writer.drain(); writer.close(); return
        await logs_stream(reader, writer, head)
        return

    if is_upgrade:
        # A claim is sacred only while its tunnel is LIVE: a concurrent ws
        # from any session -> 503 (hijack guard; that visitor's retry gets a
        # fresh instance). With no live tunnel, a new session may take over
        # the claim — this is the page-reload path (new UUID + affinity
        # cookie routes to the old, already-booted instance: instant ready).
        if state['tunnel_open']:
            writer.write(http_response('503 Busy', json.dumps({'error': 'busy'})))
            await writer.drain()
            writer.close()
            return
        if session != state['session']:
            state['claimed_at'] = time.time()  # new visitor: fresh clock
        state['session'] = session or 'anonymous'
        state['tunnel_open'] = True
        state['claimed_at'] = state['claimed_at'] or time.time()
        try:
            br, bw = await asyncio.open_connection(*BRIDGE)
        except OSError:
            state['tunnel_open'] = False
            writer.write(http_response('502 Bad Gateway', json.dumps({'error': 'bridge not up'})))
            await writer.drain()
            writer.close()
            return
        bw.write(raw)
        await bw.drain()
        try:
            await asyncio.gather(pipe(reader, bw), pipe(br, writer))
        finally:
            state['tunnel_open'] = False
        return

    if url.path == '/start' and method == 'POST':
        # Explicit session claim, before any viewer connects. Busy only if a
        # live tunnel exists or another session actively holds the claim
        # with a live tunnel; an idle claim is takeable (reload semantics).
        if state['tunnel_open'] and session != state['session']:
            writer.write(http_response('503 Busy', json.dumps({'error': 'busy'})))
        else:
            if session != state['session']:
                state['claimed_at'] = time.time()
            state['session'] = session or 'anonymous'
            state['claimed_at'] = state['claimed_at'] or time.time()
            writer.write(http_response('200 OK', json.dumps({'ok': True})))
        await writer.drain()
        writer.close()
        return

    if url.path == '/status':
        if state['session'] and session != state['session']:
            writer.write(http_response('409 Conflict', json.dumps({'error': 'not your instance'})))
        else:
            s = read_status_file()
            up = int(time.time() - (state['claimed_at'] or s['start']))
            body = json.dumps({
                'claimed': state['session'] is not None,
                'ready': s['ready'], 'rtf': s['rtf'], 'nodes': s['nodes'],
                'uptime_s': up, 'remaining_s': max(0, SESSION_SECONDS - up),
                'fleet': {'running': fleet_running(), 'budget': FLEET_BUDGET},
                'log': s['log'],
            })
            writer.write(http_response('200 OK', body))
        await writer.drain()
        writer.close()
        return

    if url.path == '/shutdown' and method == 'POST':
        if state['session'] is None or session != state['session']:
            writer.write(http_response('403 Forbidden', json.dumps({'error': 'forbidden'})))
            await writer.drain()
            writer.close()
            return
        writer.write(http_response('200 OK', json.dumps({'bye': True})))
        await writer.drain()
        writer.close()
        await asyncio.sleep(0.2)
        # SIGINT, not SIGTERM: PID 1 (ros2 launch) has no SIGTERM handler
        # installed, and unhandled signals to PID 1 are ignored by the
        # kernel — SIGINT triggers launch's real shutdown (verified).
        os.kill(1, signal.SIGINT)
        return

    if url.path.startswith('/fs/'):
        if not guarded_ok():
            writer.write(http_response('403 Forbidden', json.dumps({'error': 'forbidden'})))
            await writer.drain(); writer.close(); return
        full = safe_path(parse_qs(url.query).get('path', [''])[0])
        if full is None:
            writer.write(http_response('400 Bad Request', json.dumps({'error': 'bad path'})))
        elif url.path == '/fs/list':
            try:
                entries = [{'name': e, 'dir': os.path.isdir(os.path.join(full, e))}
                           for e in sorted(os.listdir(full))]
                writer.write(http_response('200 OK', json.dumps({'path': full, 'entries': entries})))
            except OSError:
                writer.write(http_response('404 Not Found', json.dumps({'error': 'no dir'})))
        elif url.path == '/fs/read':
            try:
                with open(full, 'rb') as f:
                    raw_bytes = f.read(512 * 1024)
                content = raw_bytes.decode('utf-8')
                writer.write(http_response('200 OK', json.dumps({'path': full, 'content': content})))
            except (OSError, UnicodeDecodeError):
                writer.write(http_response('404 Not Found', json.dumps({'error': 'unreadable'})))
        elif url.path == '/fs/write' and method == 'POST':
            clen = 0
            for line in head.split('\r\n'):
                if line.lower().startswith('content-length:'):
                    clen = int(line.split(':', 1)[1].strip())
            body = await reader.readexactly(clen) if clen else b''
            try:
                with open(full, 'wb') as f:
                    f.write(body)
                writer.write(http_response('200 OK', json.dumps({'ok': True})))
            except OSError:
                writer.write(http_response('400 Bad Request', json.dumps({'error': 'write failed'})))
        else:
            writer.write(http_response('404 Not Found', json.dumps({'error': 'unknown fs op'})))
        await writer.drain()
        writer.close()
        return

    writer.write(http_response('200 OK', json.dumps({'service': 'robium demo gateway'})))
    await writer.drain()
    writer.close()


async def main():
    server = await asyncio.start_server(handle, '0.0.0.0', PORT)
    print(f'demo_gateway listening on :{PORT}', flush=True)
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    asyncio.run(main())
