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
import json
import os
import signal
import time
from urllib.parse import parse_qs, urlsplit

PORT = int(os.environ.get('PORT', '8765'))
BRIDGE = ('127.0.0.1', 8766)
STATUS_PATH = '/tmp/demo_status.json'
SESSION_SECONDS = 1800

state = {'session': None, 'tunnel_open': False, 'claimed_at': None}


def http_response(status, body, extra=''):
    # Connection: close is load-bearing behind Cloud Run's proxy: it pools
    # keep-alive connections to the instance, and closing the socket after a
    # response without declaring it surfaces as "malformed HTTP response"
    # 503s at the edge (verified 2026-07-13).
    return (f'HTTP/1.1 {status}\r\nContent-Type: application/json\r\n'
            f'Content-Length: {len(body)}\r\nAccess-Control-Allow-Origin: *\r\n'
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

    if is_upgrade:
        # One session per instance lifetime: the first ws claims; later ws
        # connections must carry the SAME session (viewer auto-reconnects),
        # anything else is another visitor -> busy (their retry gets a fresh
        # instance). Prevents session hijack during boot-retry gaps.
        if state['tunnel_open']:
            writer.write(http_response('503 Busy', json.dumps({'error': 'busy'})))
            await writer.drain()
            writer.close()
            return
        if state['session'] is not None and session != state['session']:
            writer.write(http_response('503 Busy', json.dumps({'error': 'claimed'})))
            await writer.drain()
            writer.close()
            return
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
