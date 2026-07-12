#!/usr/bin/env python3
"""Map sanity: yaml+pgm exist, and the map has meaningful occupied AND free area."""
import os
import re
import sys


def read_pgm(path):
    with open(path, 'rb') as f:
        data = f.read()
    if not data.startswith(b'P5'):
        sys.exit(f'FAIL: {path} is not a binary PGM (P5)')
    # strip comments, parse header: magic, width, height, maxval
    tokens = []
    for line in data.split(b'\n'):
        if line.startswith(b'#'):
            continue
        tokens.extend(line.split())
        if len(tokens) >= 4:
            break
    width, height, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    header_len = data.find(bytes(tokens[3])) + len(bytes(tokens[3])) + 1
    pixels = data[header_len:header_len + width * height]
    return width, height, maxval, pixels


def main():
    if len(sys.argv) != 2:
        sys.exit('usage: check_map.py <map.yaml>')
    yaml_path = sys.argv[1]
    text = open(yaml_path).read()
    m = re.search(r'image:\s*(\S+)', text)
    if not m:
        sys.exit('FAIL: no image key in map yaml')
    pgm_path = os.path.join(os.path.dirname(yaml_path), m.group(1))

    width, height, maxval, pixels = read_pgm(pgm_path)
    total = width * height
    # map_saver trinary PGM: free=254 (white), unknown=205 (gray),
    # occupied=0 (black). The free threshold must sit ABOVE the unknown
    # gray (205/255 = 0.80), or an all-unknown map counts as free.
    free = sum(1 for p in pixels if p >= 0.9 * maxval)
    occ = sum(1 for p in pixels if p <= 0.35 * maxval)
    unknown = total - free - occ

    print(f'{width}x{height} free={free} occ={occ} '
          f'unknown={unknown} total={total}')
    if total < 10_000:
        sys.exit('FAIL: map suspiciously small')
    if free < 0.05 * total:
        sys.exit('FAIL: almost no free space mapped')
    if occ < 100:
        sys.exit('FAIL: almost no obstacles mapped (walls/pillars missing)')
    print('PASS: map is non-trivial')


if __name__ == '__main__':
    main()
