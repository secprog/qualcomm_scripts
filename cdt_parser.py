#!/usr/bin/env python3
"""
Qualcomm CDT (Configuration Data Table) v1 parser.

Format (v1):
  Header:
    0x00: 4  bytes  magic "CDT\\0"
    0x04: 2  bytes  cdt_version (LE)   (usually 1)
    0x06: 4  bytes  reserved (LE)
    0x0A: 4  bytes  reserved (LE)
  0x0E: metadata table: repeated entries:
        uint16 offset_to_cdb (LE), uint16 cdb_size (LE)
        Count is inferred from first offset: count = (first_offset - 0x0E)/4
  Then CDB blocks live at the offsets listed above.

Platform-ID CDB (common):
  Qualcomm boot code shows a packed struct with fields like:
    nVersion, nPlatform, hw_major, hw_minor, subtype, num_kvps, kvps...
  Some variants use subtype as u8, others as u16 (seen in samples).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple


def u16le(b: bytes, off: int) -> int:
    if off + 2 > len(b):
        raise ValueError(f"Need 2 bytes at offset {off}, file too small")
    return int.from_bytes(b[off:off + 2], "little")


def u32le(b: bytes, off: int) -> int:
    if off + 4 > len(b):
        raise ValueError(f"Need 4 bytes at offset {off}, file too small")
    return int.from_bytes(b[off:off + 4], "little")


@dataclass
class MetaEntry:
    index: int
    offset: int
    size: int


@dataclass
class PlatformKVP:
    key: int
    value: int


@dataclass
class PlatformInfo:
    schema_version: int
    platform: int
    hw_major: int
    hw_minor: int
    subtype: int
    kvps: List[PlatformKVP]


@dataclass
class CdbBlock:
    index: int
    offset: int
    size: int
    hex: str
    platform_info: Optional[PlatformInfo] = None


@dataclass
class CdtParsed:
    path: str
    file_size: int
    magic: str
    cdt_version: int
    reserved1: int
    reserved2: int
    meta: List[MetaEntry]
    blocks: List[CdbBlock]


def try_parse_platform_info(block: bytes) -> Optional[PlatformInfo]:
    """
    Best-effort decode of the Platform-ID CDB. For older layouts, subtype is u8.
    For some newer layouts, subtype appears as u16 (RubikPi3 sample).
    KVPs are parsed as (key,value) byte pairs.
    """
    if len(block) < 6:
        return None

    v = block[0]
    platform = block[1]
    hw_major = block[2]
    hw_minor = block[3]

    # Heuristic:
    # - v >= 5: subtype is u16 at [4:6], num_kvps at [6], kvps from [7:]
    # - else: subtype is u8 at [4], num_kvps at [5], kvps from [6:]
    if v >= 5:
        if len(block) < 7:
            return None
        subtype = int.from_bytes(block[4:6], "little")
        num_kvps = block[6]
        kvp_off = 7
    else:
        subtype = block[4]
        num_kvps = block[5]
        kvp_off = 6

    rem = block[kvp_off:]
    # KVPs are 2 bytes each in common bootloader layouts (key,value).
    max_kvps = len(rem) // 2
    if num_kvps > max_kvps:
        # Don’t explode; clamp and keep going.
        num_kvps = max_kvps

    kvps: List[PlatformKVP] = []
    for i in range(num_kvps):
        key = rem[i * 2]
        val = rem[i * 2 + 1]
        kvps.append(PlatformKVP(key=key, value=val))

    return PlatformInfo(
        schema_version=v,
        platform=platform,
        hw_major=hw_major,
        hw_minor=hw_minor,
        subtype=subtype,
        kvps=kvps,
    )


def parse_cdt(data: bytes, path: str = "<bytes>") -> CdtParsed:
    if len(data) < 0x0E:
        raise ValueError("Too small to be a CDT")

    magic = data[0:4]
    if magic != b"CDT\x00":
        raise ValueError(f"Bad magic: {magic!r} (expected b'CDT\\x00')")

    cdt_version = u16le(data, 0x04)
    reserved1 = u32le(data, 0x06)
    reserved2 = u32le(data, 0x0A)

    meta_start = 0x0E
    first_off = u16le(data, meta_start)

    meta: List[MetaEntry] = []

    # Preferred way: infer count from first offset (matches Qualcomm boot arrays).
    if first_off >= meta_start and (first_off - meta_start) % 4 == 0:
        count = (first_off - meta_start) // 4
        pos = meta_start
        for i in range(count):
            off = u16le(data, pos)
            size = u16le(data, pos + 2)
            meta.append(MetaEntry(index=i, offset=off, size=size))
            pos += 4
    else:
        # Fallback: scan until offsets stop making sense.
        pos = meta_start
        i = 0
        while pos + 4 <= len(data) and i < 1024:
            off = u16le(data, pos)
            size = u16le(data, pos + 2)
            if off == 0 or size == 0:
                break
            meta.append(MetaEntry(index=i, offset=off, size=size))
            pos += 4
            i += 1

    blocks: List[CdbBlock] = []
    for m in meta:
        if m.offset + m.size > len(data):
            # Many CDT partitions are padded to sector size; but offsets should still fit.
            # If it doesn't, keep a safe slice.
            blk = data[m.offset: min(len(data), m.offset + m.size)]
        else:
            blk = data[m.offset:m.offset + m.size]

        pinfo = try_parse_platform_info(blk)

        blocks.append(CdbBlock(
            index=m.index,
            offset=m.offset,
            size=m.size,
            hex=blk.hex(),
            platform_info=pinfo
        ))

    return CdtParsed(
        path=path,
        file_size=len(data),
        magic=magic.decode("ascii", errors="replace"),
        cdt_version=cdt_version,
        reserved1=reserved1,
        reserved2=reserved2,
        meta=meta,
        blocks=blocks,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse Qualcomm CDT (Configuration Data Table) v1")
    ap.add_argument("file", type=Path, help="cdt.bin / CDT blob")
    ap.add_argument("--json", action="store_true", help="Output JSON")
    ap.add_argument("--dump-blocks", type=Path, default=None, help="Write each CDB block to DIR")
    args = ap.parse_args()

    data = args.file.read_bytes()
    parsed = parse_cdt(data, path=str(args.file))

    if args.dump_blocks:
        args.dump_blocks.mkdir(parents=True, exist_ok=True)
        for b in parsed.blocks:
            out = args.dump_blocks / f"cdb_{b.index:02d}_off{b.offset}_sz{b.size}.bin"
            out.write_bytes(bytes.fromhex(b.hex))

    if args.json:
        print(json.dumps(asdict(parsed), indent=2))
    else:
        print(f"{parsed.path}: CDT v{parsed.cdt_version}, size={parsed.file_size} bytes")
        for m, b in zip(parsed.meta, parsed.blocks):
            print(f"  CDB[{m.index}] offset={m.offset} size={m.size}")
            if b.platform_info:
                pi = b.platform_info
                print(
                    f"    PlatformInfo: schema={pi.schema_version} "
                    f"platform=0x{pi.platform:02x} hw={pi.hw_major}.{pi.hw_minor} "
                    f"subtype={pi.subtype} kvps={[(k.key, k.value) for k in pi.kvps]}"
                )


if __name__ == "__main__":
    main()
