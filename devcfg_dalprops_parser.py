#!/usr/bin/env python3
"""
Qualcomm DEVCFG (MBN/ELF) DALProps directory parser.

Many Qualcomm `devcfg*.mbn` images are ELF64 files that embed a DAL property
directory (DALProps). This script:

  • parses ELF64 program headers
  • finds the DALProps header inside the first LOAD segment
  • reads the directory table (name_ptr, djb2(name), rel_offset)
  • slices per-entry data blocks (length inferred from next rel_offset;
    last entry ends at the header's p2 pointer)
  • can emit JSON, hexdump previews, extract blobs, and diff two images.

It intentionally does NOT attempt to fully decode each entry's internal
schema (it's highly target-/build-specific). You still get a clean directory
and stable blob boundaries for diffing/reversing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ELF_MAGIC = b"\x7fELF"
PT_LOAD = 1

def djb2_32_ascii(s: bytes) -> int:
    h = 5381
    for b in s:
        h = ((h << 5) + h + b) & 0xFFFFFFFF
    return h


def hexdump(buf: bytes, base: int = 0, width: int = 16) -> str:
    out = []
    for i in range(0, len(buf), width):
        chunk = buf[i:i + width]
        hx = " ".join(f"{b:02x}" for b in chunk)
        asc = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        out.append(f"{base + i:08x}  {hx:<{width*3-1}}  {asc}")
    return "\n".join(out)


@dataclass
class LoadSeg:
    offset: int
    vaddr: int
    filesz: int
    memsz: int
    flags: int


@dataclass
class DalPropsHeader:
    file_off: int          # file offset where the 4-qword header starts
    p1: int                # pointer into segment (often points near string table)
    p2: int                # pointer into segment (used here as property-data end)
    count: int             # number of directory entries
    dirptr: int            # pointer to directory table
    entry_size: int = 0x28 # observed on these images


@dataclass
class DirEntry:
    name: str
    name_ptr: int
    hash: int
    rel: int
    entry_file_off: int
    data_file_off: int
    data_len: int

    def sha256(self, blob: bytes) -> str:
        return hashlib.sha256(blob).hexdigest()


class DevCfgDalProps:
    def __init__(self, path: Path):
        self.path = path
        self.data = path.read_bytes()
        self.loads: List[LoadSeg] = self._parse_load_segments()
        self.main_seg: LoadSeg = self._pick_main_segment()
        self.header: DalPropsHeader = self._find_header()
        self.entries: List[DirEntry] = self._read_entries()

    # ---------- ELF parsing ----------
    def _parse_load_segments(self) -> List[LoadSeg]:
        d = self.data
        if len(d) < 0x40 or d[:4] != ELF_MAGIC:
            raise ValueError("Not an ELF file")
        if d[4] != 2 or d[5] != 1:
            raise ValueError("Expected ELF64 little-endian")

        e_phoff = struct.unpack_from("<Q", d, 0x20)[0]
        e_phentsize = struct.unpack_from("<H", d, 0x36)[0]
        e_phnum = struct.unpack_from("<H", d, 0x38)[0]

        loads: List[LoadSeg] = []
        for i in range(e_phnum):
            off = e_phoff + i * e_phentsize
            p_type, p_flags = struct.unpack_from("<II", d, off)
            p_offset, p_vaddr, _p_paddr, p_filesz, p_memsz, _p_align = struct.unpack_from("<QQQQQQ", d, off + 8)
            if p_type == PT_LOAD and p_filesz > 0:
                loads.append(LoadSeg(offset=p_offset, vaddr=p_vaddr, filesz=p_filesz, memsz=p_memsz, flags=p_flags))
        if not loads:
            raise ValueError("No PT_LOAD segments found")
        return loads

    def _pick_main_segment(self) -> LoadSeg:
        # These devcfg images keep DALProps in the LOAD segment with lowest vaddr (often 0x1c00xxxx).
        return sorted(self.loads, key=lambda s: s.vaddr)[0]

    def _in_seg(self, ptr: int) -> bool:
        s = self.main_seg
        return s.vaddr <= ptr < (s.vaddr + s.filesz)

    def _fileoff_from_vaddr(self, ptr: int) -> int:
        s = self.main_seg
        return s.offset + (ptr - s.vaddr)

    # ---------- DALProps parsing ----------
    def _find_header(self) -> DalPropsHeader:
        d = self.data
        s = self.main_seg
        # Search the first 0x200 bytes of the segment for the 4-qword signature:
        # (p1, p2, count, dirptr) where p1/p2/dirptr are in-segment pointers
        # and dirptr points to entries that validate as (name_ptr, djb2(name), rel, padding...).
        for rel_off in range(0, min(0x200, s.filesz - 0x20), 8):
            base = s.offset + rel_off
            p1, p2, count, dirptr = struct.unpack_from("<QQQQ", d, base)
            if not (self._in_seg(p1) and self._in_seg(p2) and self._in_seg(dirptr)):
                continue
            if not (0 < count < 0x4000):
                continue

            dir_file = self._fileoff_from_vaddr(dirptr)
            # validate first entry
            try:
                name_ptr = struct.unpack_from("<Q", d, dir_file)[0]
                if not self._in_seg(name_ptr):
                    continue
                name_file = self._fileoff_from_vaddr(name_ptr)
                end = d.find(b"\x00", name_file, name_file + 512)
                if end == -1:
                    continue
                name_bytes = d[name_file:end]
                if not name_bytes or not all(32 <= b < 127 for b in name_bytes):
                    continue
                h, _rel = struct.unpack_from("<II", d, dir_file + 8)
                if djb2_32_ascii(name_bytes) != h:
                    continue
            except struct.error:
                continue

            return DalPropsHeader(file_off=base, p1=p1, p2=p2, count=count, dirptr=dirptr)

        raise ValueError("DALProps header not found in the first LOAD segment")

    def _read_entries(self) -> List[DirEntry]:
        d = self.data
        s = self.main_seg
        h = self.header

        dir_file = self._fileoff_from_vaddr(h.dirptr)
        prop_end_rel = h.p2 - s.vaddr  # end marker for last entry length

        # Read raw directory entries
        raw: List[Tuple[str, int, int, int, int]] = []
        for i in range(h.count):
            eoff = dir_file + i * h.entry_size
            name_ptr = struct.unpack_from("<Q", d, eoff)[0]
            hh, rel = struct.unpack_from("<II", d, eoff + 8)
            name_file = self._fileoff_from_vaddr(name_ptr)
            end = d.find(b"\x00", name_file, name_file + 2048)
            if end == -1:
                end = name_file
            name_bytes = d[name_file:end]
            name = name_bytes.decode("ascii", errors="replace")
            raw.append((name, name_ptr, hh, rel, eoff))

        # Compute lengths by sorting rel offsets
        rels_sorted = sorted(set(r[3] for r in raw))
        rel_to_next: Dict[int, int] = {}
        for i, r in enumerate(rels_sorted):
            rel_to_next[r] = rels_sorted[i + 1] if (i + 1) < len(rels_sorted) else prop_end_rel

        out: List[DirEntry] = []
        for name, name_ptr, hh, rel, eoff in raw:
            nxt = rel_to_next[rel]
            data_file = s.offset + rel
            data_len = max(0, nxt - rel)
            out.append(
                DirEntry(
                    name=name,
                    name_ptr=name_ptr,
                    hash=hh,
                    rel=rel,
                    entry_file_off=eoff,
                    data_file_off=data_file,
                    data_len=data_len,
                )
            )
        return out

    # ---------- High-level helpers ----------
    def get_blob(self, entry: DirEntry) -> bytes:
        return self.data[entry.data_file_off: entry.data_file_off + entry.data_len]

    def to_dict(self, include_hashes: bool = True) -> Dict:
        s = self.main_seg
        out = {
            "file": str(self.path),
            "file_size": len(self.data),
            "load_segments": [
                {"offset": x.offset, "vaddr": x.vaddr, "filesz": x.filesz, "memsz": x.memsz, "flags": x.flags}
                for x in self.loads
            ],
            "main_segment": {"offset": s.offset, "vaddr": s.vaddr, "filesz": s.filesz, "flags": s.flags},
            "dalprops_header": {
                "file_off": self.header.file_off,
                "p1": self.header.p1,
                "p2": self.header.p2,
                "count": self.header.count,
                "dirptr": self.header.dirptr,
                "entry_size": self.header.entry_size,
            },
            "entries": [],
        }
        for e in self.entries:
            item = {
                "name": e.name,
                "djb2_hash": e.hash,
                "rel": e.rel,
                "data_file_off": e.data_file_off,
                "data_len": e.data_len,
            }
            if include_hashes:
                item["sha256"] = e.sha256(self.get_blob(e))
            out["entries"].append(item)
        return out


def sanitize_filename(name: str) -> str:
    # Keep it filesystem-friendly; preserve some structure.
    name = name.strip().replace("\\", "/")
    name = name.replace("/", "__")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:180] if len(name) > 180 else name


def cmd_list(dc: DevCfgDalProps, args: argparse.Namespace) -> None:
    for e in dc.entries:
        if args.filter and args.filter not in e.name:
            continue
        blob = dc.get_blob(e)
        sha = hashlib.sha256(blob).hexdigest()
        print(f"{e.name:40} rel=0x{e.rel:04x} len={e.data_len:6} sha256={sha[:16]}…")


def cmd_show(dc: DevCfgDalProps, args: argparse.Namespace) -> None:
    matches = [e for e in dc.entries if e.name == args.name]
    if not matches:
        raise SystemExit(f"Entry not found: {args.name!r}")
    e = matches[0]
    blob = dc.get_blob(e)
    print(f"Name: {e.name}")
    print(f"djb2: 0x{e.hash:08x}")
    print(f"rel : 0x{e.rel:x}")
    print(f"file: 0x{e.data_file_off:x}")
    print(f"len : {e.data_len} bytes")
    print(f"sha256: {hashlib.sha256(blob).hexdigest()}")
    print()
    print(hexdump(blob[: args.max_bytes], base=e.data_file_off))


def cmd_json(dc: DevCfgDalProps, args: argparse.Namespace) -> None:
    obj = dc.to_dict(include_hashes=not args.no_hashes)
    txt = json.dumps(obj, indent=2)
    if args.out:
        Path(args.out).write_text(txt, encoding="utf-8")
    else:
        print(txt)


def cmd_extract(dc: DevCfgDalProps, args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for e in dc.entries:
        if args.filter and args.filter not in e.name:
            continue
        blob = dc.get_blob(e)
        sha = hashlib.sha256(blob).hexdigest()[:16]
        fn = sanitize_filename(e.name)
        path = outdir / f"{fn}__rel_{e.rel:04x}__len_{e.data_len}__{sha}.bin"
        path.write_bytes(blob)
    print(f"Extracted {len(dc.entries)} blobs to {outdir}")


def cmd_diff(dc_a: DevCfgDalProps, dc_b: DevCfgDalProps) -> None:
    a = {e.name: e for e in dc_a.entries}
    b = {e.name: e for e in dc_b.entries}

    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    common = sorted(set(a) & set(b))

    changed = []
    for name in common:
        ea, eb = a[name], b[name]
        ha = hashlib.sha256(dc_a.get_blob(ea)).digest()
        hb = hashlib.sha256(dc_b.get_blob(eb)).digest()
        if ha != hb or ea.data_len != eb.data_len:
            changed.append(name)

    if added:
        print("Added:")
        for n in added:
            print(f"  + {n}")
    if removed:
        print("Removed:")
        for n in removed:
            print(f"  - {n}")
    if changed:
        print("Changed:")
        for n in changed:
            ea, eb = a[n], b[n]
            sa = hashlib.sha256(dc_a.get_blob(ea)).hexdigest()[:16]
            sb = hashlib.sha256(dc_b.get_blob(eb)).hexdigest()[:16]
            print(f"  * {n}\n      A: rel=0x{ea.rel:04x} len={ea.data_len:6} sha={sa}…\n      B: rel=0x{eb.rel:04x} len={eb.data_len:6} sha={sb}…")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parse Qualcomm DEVCFG (ELF64 MBN) DALProps directory and slice entry blobs."
    )
    p.add_argument("mbn", help="Path to devcfg*.mbn (ELF64)")
    sub = p.add_subparsers(dest="cmd", required=False)

    p_list = sub.add_parser("list", help="List directory entries")
    p_list.add_argument("--filter", help="Only show entries whose name contains this substring")
    p_list.set_defaults(_fn=lambda dc, a: cmd_list(dc, a))

    p_show = sub.add_parser("show", help="Hexdump a single entry by exact name")
    p_show.add_argument("name", help="Exact entry name (use `list` to see names)")
    p_show.add_argument("--max-bytes", type=int, default=256, help="Max bytes to hexdump")
    p_show.set_defaults(_fn=lambda dc, a: cmd_show(dc, a))

    p_json = sub.add_parser("json", help="Emit JSON")
    p_json.add_argument("--out", help="Write JSON to a file (default: stdout)")
    p_json.add_argument("--no-hashes", action="store_true", help="Skip computing sha256 for blobs")
    p_json.set_defaults(_fn=lambda dc, a: cmd_json(dc, a))

    p_ex = sub.add_parser("extract", help="Extract all blobs to a directory")
    p_ex.add_argument("outdir", help="Directory to write extracted blobs")
    p_ex.add_argument("--filter", help="Only extract entries whose name contains this substring")
    p_ex.set_defaults(_fn=lambda dc, a: cmd_extract(dc, a))

    p_diff = sub.add_parser("diff", help="Diff this image against another")
    p_diff.add_argument("other", help="Other devcfg*.mbn to compare")
    p_diff.set_defaults(_fn=None)

    return p


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    dc = DevCfgDalProps(Path(args.mbn))

    if args.cmd == "diff":
        other = DevCfgDalProps(Path(args.other))
        cmd_diff(dc, other)
        return

    # default command if none given
    if args.cmd is None:
        cmd_list(dc, argparse.Namespace(filter=None))
        return

    args._fn(dc, args)


if __name__ == "__main__":
    main()
