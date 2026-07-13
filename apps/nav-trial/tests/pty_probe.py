"""PTY smoke: open /pty, echo a marker, and assert egress is blocked.

Usage: pty_probe.py <host> <session> [--expect-egress-blocked]
Local runs omit the flag (egress isn't firewalled locally); the cloud gate
passes it so a reachable internet = hard failure.
"""
import asyncio
import base64
import os
import ssl
import sys


async def main(host, session, expect_blocked):
    hostname, _, port = host.partition(':')
    if port:  # explicit port → local plaintext container
        reader, writer = await asyncio.open_connection(hostname, int(port))
    else:     # bare host → cloud, TLS on 443
        ctx = ssl.create_default_context()
        reader, writer = await asyncio.open_connection(hostname, 443, ssl=ctx)
    key = base64.b64encode(os.urandom(16)).decode()
    writer.write((f'GET /pty?session={session} HTTP/1.1\r\nHost: {host}\r\n'
                  f'Upgrade: websocket\r\nConnection: Upgrade\r\n'
                  f'Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
    await writer.drain()
    await reader.readuntil(b'\r\n\r\n')

    def frame(s):
        d = s.encode(); n = len(d); mask = os.urandom(4)
        return bytes([0x81, 0x80 | n]) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(d))

    writer.write(frame('echo hello-pty\n')); await writer.drain()
    writer.write(frame('curl -sS --max-time 5 https://example.com >/dev/null 2>&1; echo EGRESS_$?\n'))
    await writer.drain()

    got = b''
    try:
        for _ in range(300):
            got += await asyncio.wait_for(reader.read(4096), timeout=20)
            if b'hello-pty' in got and b'EGRESS_' in got:
                break
    except asyncio.TimeoutError:
        pass
    text = got.decode('latin1')
    assert 'hello-pty' in text, 'PTY did not echo'
    if expect_blocked:
        assert 'EGRESS_0' not in text, 'EGRESS WAS OPEN — hardening failed'
        print('PTY OK + EGRESS BLOCKED')
    else:
        print('PTY OK (egress assertion skipped — local)')


asyncio.run(main(sys.argv[1], sys.argv[2], '--expect-egress-blocked' in sys.argv))
