#!/usr/bin/env python3
"""
xbl_config_parser.py

CFGL header format:
  "CFGL" + u16(version) + u16(count) + u32(total_len) + u32(reserved)

Important: CFGL entry names are NOT guaranteed to contain "/".
Some builds store DTB entries like "post-ddr-foo.dtb" (no leading slash).
This script parses entry names WITHOUT relying on "/" being present.

What this tool does:
- Parse CFGL inside an xbl_config*.elf
- List CFGL entries (pretty print)
- Extract *all* CFGL payload files to a folder and generate a spec.json that points to them
- Build a new ELF from a base + spec:
    * CFGL is rebuilt
    * payload files are read from disk
    * entries can be added/removed/reordered by editing the spec
    * program headers rebuilt to ONLY PT_LOAD (1 for CFGL + 1 per payload)
    * p_flags and p_align are FORCED:
        - p_flags = 0x01000007
        - p_align = 1
    * section headers stripped
    * output is truncated (no random tail garbage), but kept large enough to still include PHDR if moved

Commands:
  - list <base.elf>
  - extract <base.elf> <out_dir> [--spec spec.json] [--overwrite]
  - build <base.elf> <spec.json> <out.elf> [--files-dir DIR]

Spec format (supports "type"):
{
  "type_templates": {
    "dtb": "post-ddr-foo.dtb",
    "pmic": "/pmic_settings.bin",
    "dcb": "/6018_8_0100_0_dcb.bin"
  },
  "entries": [
    {"name": "/pmic_settings.bin", "type": "pmic", "file": "pmic_settings.bin"},
    {"name": "post-ddr-foo.dtb",   "type": "dtb",  "file": "dtbs/post-ddr-foo.dtb"},
    {"name": "/6018_8_0100_0_dcb.bin", "type": "dcb", "file": "dcb/6018_8_0100_0_dcb.bin"}
  ]
}

Template selection for NEW entries:
1) If type_templates[type] exists -> clone that base entry
2) Else if an entry of that type exists in base -> clone the first one
3) Else -> clone the first base entry

Type detection heuristics:
- dtb: name/file ends with .dtb/.dtbo OR payload starts with 0xD00DFEED (DTB magic, big-endian)
- dcb: name/file contains "dcb" or ends with "_dcb.bin"
- pmic: name/file contains "pmic"
- other: "other"
"""

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable

MAGIC_CFGL = b"CFGL"

# ELF64 constants
ELF64_EHDR_SIZE = 0x40
ELF64_PHDR_SIZE = 56
PT_LOAD = 1

# FORCED segment fields (as requested)
FORCED_P_FLAGS = 0x01000007
FORCED_P_ALIGN = 1

# DTB magic is big-endian 0xD00DFEED at file start
DTB_MAGIC_BE = b"\xD0\x0D\xFE\xED"


# ----------------------------- helpers -----------------------------

def align4(x: int) -> int:
    return (x + 3) & ~3

def read_u16(b: bytes, off: int) -> int:
    return struct.unpack_from("<H", b, off)[0]

def read_u32(b: bytes, off: int) -> int:
    return struct.unpack_from("<I", b, off)[0]

def read_u64(b: bytes, off: int) -> int:
    return struct.unpack_from("<Q", b, off)[0]

def write_u16(b: bytearray, off: int, v: int) -> None:
    struct.pack_into("<H", b, off, v & 0xFFFF)

def write_u64(b: bytearray, off: int, v: int) -> None:
    struct.pack_into("<Q", b, off, v & 0xFFFFFFFFFFFFFFFF)

def elf_check(b: bytes) -> None:
    if len(b) < ELF64_EHDR_SIZE:
        raise RuntimeError("Too small to be ELF64")
    if b[:4] != b"\x7fELF":
        raise RuntimeError("Not ELF")
    if b[4] != 2:
        raise RuntimeError("Not ELF64")
    if b[5] != 1:
        raise RuntimeError("Not little-endian")

def strip_shdr(b: bytearray) -> None:
    # ELF64: e_shoff@0x28(u64), e_shentsize@0x3A(u16), e_shnum@0x3C(u16), e_shstrndx@0x3E(u16)
    write_u64(b, 0x28, 0)
    write_u16(b, 0x3A, 0)
    write_u16(b, 0x3C, 0)
    write_u16(b, 0x3E, 0)

def safe_join_outdir(out_dir: Path, cfgl_name: str) -> Path:
    name = cfgl_name.lstrip("/")
    name = name.replace("..", "_")
    return out_dir / name

def mostly_printable(s: str) -> bool:
    if not s:
        return False
    printable = sum(1 for ch in s if ch.isprintable())
    return printable >= max(1, len(s) // 2)

def guess_type_from_name(name: str, file_hint: Optional[str] = None) -> str:
    n = (name or "").lower()
    fh = (file_hint or "").lower()

    if n.endswith(".dtb") or n.endswith(".dtbo") or fh.endswith(".dtb") or fh.endswith(".dtbo"):
        return "dtb"
    if "dcb" in n or "dcb" in fh or n.endswith("_dcb.bin") or fh.endswith("_dcb.bin"):
        return "dcb"
    if "pmic" in n or "pmic" in fh:
        return "pmic"
    return "other"

def guess_type_from_payload(payload: bytes, fallback: str) -> str:
    if payload.startswith(DTB_MAGIC_BE):
        return "dtb"
    return fallback


# ----------------------------- CFGL parsing -----------------------------

def find_cfgl(b: bytes) -> Tuple[int, int, int, int]:
    off = b.find(MAGIC_CFGL)
    if off < 0:
        raise RuntimeError("CFGL not found")
    if off + 16 > len(b):
        raise RuntimeError("CFGL header truncated")

    ver = read_u16(b, off + 4)
    cnt = read_u16(b, off + 6)
    total_len = read_u32(b, off + 8)

    if total_len < 16:
        raise RuntimeError(f"CFGL total_len too small: {total_len}")
    if off + total_len > len(b):
        raise RuntimeError("CFGL total_len out of bounds")

    return off, cnt, ver, total_len


@dataclass
class EntryLayout:
    name: str
    pre: List[int]
    off_i: int
    size_i: int
    rel: bool


def _walk_entry_candidates(cfgl_block: bytes, start: int, max_name_len: int = 4096) -> Iterable[Tuple[int, str, List[int], int]]:
    n = len(cfgl_block)
    max_scan_u32 = min(4096, (n - start) // 4)

    for k in range(max_scan_u32):
        len_off = start + k * 4
        if len_off + 4 > n:
            break

        name_len = read_u32(cfgl_block, len_off)
        if name_len == 0 or name_len > max_name_len:
            continue
        name_start = len_off + 4
        name_end = name_start + name_len
        if name_end > n:
            continue

        name_bytes = cfgl_block[name_start:name_end]
        if not any(c != 0 for c in name_bytes):
            continue

        name = name_bytes.decode("utf-8", errors="replace")
        if not mostly_printable(name):
            continue

        nxt = align4(name_end)
        if nxt <= start or nxt > n:
            continue

        preamble_b = cfgl_block[start:len_off]
        if len(preamble_b) % 4 != 0:
            continue
        pre_u32 = list(struct.unpack_from("<" + "I" * (len(preamble_b) // 4), preamble_b, 0))

        yield nxt, name, pre_u32, name_len


def _collect_phdr_pairs(b: bytes, cfgl_off: int) -> Tuple[set, set]:
    elf_check(b)
    e_phoff = read_u64(b, 0x20)
    e_phentsize = read_u16(b, 0x36)
    e_phnum = read_u16(b, 0x38)

    if e_phentsize != ELF64_PHDR_SIZE:
        raise RuntimeError(f"Unexpected e_phentsize={e_phentsize} (expected 56)")
    if e_phoff + e_phentsize * e_phnum > len(b):
        raise RuntimeError("PHDR table out of bounds")

    abs_pairs = set()
    rel_pairs = set()
    for i in range(e_phnum):
        o = e_phoff + i * e_phentsize
        if read_u32(b, o + 0x00) != PT_LOAD:
            continue
        p_offset = int(read_u64(b, o + 0x08))
        p_filesz = int(read_u64(b, o + 0x20))
        abs_pairs.add((p_offset, p_filesz))
        if p_offset >= cfgl_off:
            rel_pairs.add((p_offset - cfgl_off, p_filesz))

    return abs_pairs, rel_pairs


def parse_cfgl_entries(b: bytes) -> Tuple[int, int, int, int, List[EntryLayout], List[Tuple[str, List[int], int]]]:
    cfgl_off, cnt, ver, total_len = find_cfgl(b)
    abs_pairs, rel_pairs = _collect_phdr_pairs(b, cfgl_off)

    cfgl_block = b[cfgl_off:cfgl_off + total_len]
    pos = 16

    layouts: List[EntryLayout] = []
    raw_entries: List[Tuple[str, List[int], int]] = []

    for _ in range(cnt):
        chosen = None
        for pos2, name, pre, name_len in _walk_entry_candidates(cfgl_block, pos):
            found: Optional[Tuple[int, bool]] = None
            for j in range(len(pre) - 1):
                a, c = pre[j], pre[j + 1]
                if (a, c) in abs_pairs:
                    found = (j, False)
                    break
                if (a, c) in rel_pairs:
                    found = (j, True)
                    break
            if found is None:
                continue

            idx, is_rel = found
            chosen = (pos2, name, pre, name_len, idx, is_rel)
            break

        if chosen is None:
            raise RuntimeError(f"CFGL parse failed at pos=0x{pos:X}: couldn't locate a valid (offset,size) pair + name_len")

        pos2, name, pre, name_len, idx, is_rel = chosen
        layouts.append(EntryLayout(name=name, pre=pre, off_i=idx, size_i=idx + 1, rel=is_rel))
        raw_entries.append((name, pre, name_len))
        pos = pos2

    return cfgl_off, cnt, ver, total_len, layouts, raw_entries


def parse_base_entry_layouts(b: bytes) -> Tuple[int, int, int, int, List[EntryLayout]]:
    cfgl_off, cnt, ver, total_len, layouts, _raw = parse_cfgl_entries(b)
    return cfgl_off, cnt, ver, total_len, layouts


def get_entry_offset_size(cfgl_off: int, lay: EntryLayout, entry_pre: List[int]) -> Tuple[int, int, int]:
    off_val = int(entry_pre[lay.off_i])
    size = int(entry_pre[lay.size_i])
    abs_off = (cfgl_off + off_val) if lay.rel else off_val
    return off_val, abs_off, size


def build_cfgl_from_layouts(cfgl_off: int, ver: int, layouts: List[EntryLayout], abs_off_sizes: List[Tuple[int, int]]) -> bytes:
    out = bytearray()
    out += MAGIC_CFGL
    out += struct.pack("<H", ver)
    out += struct.pack("<H", len(layouts))
    out += struct.pack("<I", 0)
    out += struct.pack("<I", 0)

    for lay, (abs_off, size) in zip(layouts, abs_off_sizes):
        pre = lay.pre[:]
        off_val = (abs_off - cfgl_off) if lay.rel else abs_off
        pre[lay.off_i] = off_val & 0xFFFFFFFF
        pre[lay.size_i] = size & 0xFFFFFFFF

        out += struct.pack("<" + "I" * len(pre), *pre)
        name_b = lay.name.encode("utf-8")
        out += struct.pack("<I", len(name_b))
        out += name_b
        while len(out) % 4:
            out += b"\x00"

    struct.pack_into("<I", out, 8, len(out))
    return bytes(out)


def write_phdr_table_at(buf: bytearray, phoff: int, loads: List[Tuple[int, int, int]], p_flags: int, p_align: int) -> None:
    for i, (p_offset, p_filesz, p_vaddr) in enumerate(loads):
        o = phoff + i * ELF64_PHDR_SIZE
        struct.pack_into(
            "<IIQQQQQQ",
            buf, o,
            PT_LOAD,
            p_flags,
            p_offset,
            p_vaddr,
            p_vaddr,
            p_filesz,
            p_filesz,
            p_align
        )


def rebuild_phdr_table_movable(outb: bytearray, loads: List[Tuple[int, int, int]]) -> Tuple[int, int]:
    elf_check(outb)

    old_phoff = int(read_u64(outb, 0x20))
    old_phentsize = int(read_u16(outb, 0x36))
    old_phnum = int(read_u16(outb, 0x38))
    if old_phentsize != ELF64_PHDR_SIZE:
        raise RuntimeError("Unexpected e_phentsize (expected 56)")

    need = len(loads)
    if need <= old_phnum:
        write_phdr_table_at(outb, old_phoff, loads, FORCED_P_FLAGS, FORCED_P_ALIGN)
        for i in range(need, old_phnum):
            o = old_phoff + i * ELF64_PHDR_SIZE
            outb[o:o + ELF64_PHDR_SIZE] = b"\x00" * ELF64_PHDR_SIZE
        write_u16(outb, 0x38, need)
        return old_phoff, need

    new_phoff = align4(len(outb))
    new_size = need * ELF64_PHDR_SIZE
    outb.extend(b"\x00" * (new_phoff + new_size - len(outb)))

    write_phdr_table_at(outb, new_phoff, loads, FORCED_P_FLAGS, FORCED_P_ALIGN)

    write_u64(outb, 0x20, new_phoff)
    write_u16(outb, 0x38, need)
    return new_phoff, need


def get_delta_from_first_pt_load(base: bytes) -> int:
    elf_check(base)
    e_phoff = int(read_u64(base, 0x20))
    e_phentsize = int(read_u16(base, 0x36))
    e_phnum = int(read_u16(base, 0x38))

    if e_phentsize != ELF64_PHDR_SIZE:
        raise RuntimeError("Unexpected e_phentsize (expected 56)")

    for i in range(e_phnum):
        o = e_phoff + i * e_phentsize
        if read_u32(base, o + 0x00) != PT_LOAD:
            continue
        p_offset = int(read_u64(base, o + 0x08))
        p_vaddr = int(read_u64(base, o + 0x10))
        return int((p_vaddr - p_offset) & 0xFFFFFFFFFFFFFFFF)

    raise RuntimeError("No PT_LOAD found in base")


def choose_templates_from_base(base_layouts: List[EntryLayout]) -> Dict[str, EntryLayout]:
    templates: Dict[str, EntryLayout] = {}
    for lay in base_layouts:
        t = guess_type_from_name(lay.name, None)
        if t != "other" and t not in templates:
            templates[t] = lay
    return templates


def pick_template_for_new_entry(
    base_map: Dict[str, EntryLayout],
    base_layouts: List[EntryLayout],
    pinned: Dict[str, str],
    inferred_templates: Dict[str, EntryLayout],
    entry_type: str,
) -> EntryLayout:
    if entry_type in pinned:
        nm = pinned[entry_type]
        if nm in base_map:
            return base_map[nm]
        raise RuntimeError(f"type_templates maps '{entry_type}' to '{nm}', but that name is not present in base CFGL")

    if entry_type in inferred_templates:
        return inferred_templates[entry_type]

    return base_layouts[0]


# ----------------------------- commands -----------------------------

def cmd_list(base_elf: str) -> None:
    b = Path(base_elf).read_bytes()
    cfgl_off, cnt, ver, total_len, layouts, raw = parse_cfgl_entries(b)

    print(f"CFGL @ 0x{cfgl_off:X}")
    print(f"  version : {ver}")
    print(f"  count   : {cnt}")
    print(f"  len     : {total_len} (0x{total_len:X})")
    print("")

    for idx in range(cnt):
        name, pre, name_len = raw[idx]
        lay = layouts[idx]
        off_val, abs_off, size = get_entry_offset_size(cfgl_off, lay, pre)

        rel_txt = "REL" if lay.rel else "ABS"
        print(f"[{idx:02d}] {name}")
        print(f"     {rel_txt} off_val : {off_val} (0x{off_val:X})")
        print(f"     abs offset : {abs_off} (0x{abs_off:X})")
        print(f"     size       : {size} (0x{size:X})")
        print(f"     pre_u32     : {len(lay.pre)} dwords | off_idx={lay.off_i} size_idx={lay.size_i} | name_len={name_len}")
        print("")


def cmd_extract(base_elf: str, out_dir: str, spec_path: Optional[str], overwrite: bool) -> None:
    base_path = Path(base_elf)
    b = base_path.read_bytes()

    cfgl_off, cnt, ver, total_len, layouts, raw = parse_cfgl_entries(b)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    spec_p = Path(spec_path) if spec_path else (out_dir_p / "spec.json")

    entries_out = []
    extracted = 0

    for idx in range(cnt):
        name, pre, _name_len = raw[idx]
        lay = layouts[idx]
        _off_val, abs_off, size = get_entry_offset_size(cfgl_off, lay, pre)

        if size < 0 or abs_off < 0 or abs_off + size > len(b):
            raise RuntimeError(f"Entry '{name}' points out of file bounds: abs_off=0x{abs_off:X} size=0x{size:X}")

        payload = b[abs_off:abs_off + size]
        out_path = safe_join_outdir(out_dir_p, name)

        if out_path.exists() and not overwrite:
            raise RuntimeError(f"Refusing to overwrite existing file: {out_path} (use --overwrite)")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(payload)
        extracted += 1

        t0 = guess_type_from_name(name, str(out_path))
        t = guess_type_from_payload(payload, t0)

        try:
            file_rel = str(out_path.relative_to(spec_p.parent))
        except Exception:
            file_rel = str(out_path)

        entries_out.append({"name": name, "type": t, "file": file_rel})

    type_templates: Dict[str, str] = {}
    for e in entries_out:
        t = e.get("type")
        if t in ("dtb", "dcb", "pmic") and t not in type_templates:
            type_templates[t] = e["name"]

    spec_obj = {"type_templates": type_templates, "entries": entries_out}
    spec_p.parent.mkdir(parents=True, exist_ok=True)
    spec_p.write_text(json.dumps(spec_obj, indent=2), encoding="utf-8")

    print(f"[+] extracted {extracted} payload file(s) to: {out_dir_p}")
    print(f"[+] wrote spec: {spec_p}")
    print(f"    base: {base_path.name}")
    print(f"    CFGL @0x{cfgl_off:X} ver={ver} count={cnt} len={total_len} (0x{total_len:X})")


def resolve_payload_path(spec_json_path: Path, files_dir: Optional[Path], file_field: str) -> Path:
    p = Path(file_field)
    if p.is_absolute():
        return p
    if files_dir is not None:
        return files_dir / p
    return spec_json_path.parent / p


def cmd_build(base_elf: str, spec_json: str, out_elf: str, files_dir: Optional[str]) -> None:
    base_path = Path(base_elf)
    base = base_path.read_bytes()

    spec_path = Path(spec_json)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if "entries" not in spec or not isinstance(spec["entries"], list):
        raise RuntimeError("Spec must have 'entries' list")

    pinned: Dict[str, str] = spec.get("type_templates", {}) or {}
    if not isinstance(pinned, dict):
        raise RuntimeError("'type_templates' must be an object/dict if present")

    files_dir_p = Path(files_dir) if files_dir else None

    cfgl_off, _cnt, ver, _total_len, base_layouts = parse_base_entry_layouts(base)
    base_map: Dict[str, EntryLayout] = {l.name: l for l in base_layouts}

    inferred_templates = choose_templates_from_base(base_layouts)

    layouts: List[EntryLayout] = []
    blobs: List[bytes] = []

    for e in spec["entries"]:
        name = e.get("name")
        file_field = e.get("file")
        entry_type = (e.get("type") or "other").lower()

        if not name:
            raise RuntimeError("Each entry must have 'name'")
        if not file_field:
            raise RuntimeError(f"Entry '{name}' missing 'file' path")

        payload_path = resolve_payload_path(spec_path, files_dir_p, file_field)
        if not payload_path.exists():
            raise RuntimeError(f"Missing payload file for '{name}': {payload_path}")

        blob = payload_path.read_bytes()
        blobs.append(blob)

        if entry_type not in ("dtb", "dcb", "pmic", "other"):
            entry_type = "other"
        if entry_type == "other":
            g0 = guess_type_from_name(name, str(payload_path))
            entry_type = guess_type_from_payload(blob, g0)

        if name in base_map:
            layouts.append(base_map[name])
        else:
            clone = pick_template_for_new_entry(
                base_map=base_map,
                base_layouts=base_layouts,
                pinned=pinned,
                inferred_templates=inferred_templates,
                entry_type=entry_type,
            )
            layouts.append(EntryLayout(
                name=name,
                pre=clone.pre[:],
                off_i=clone.off_i,
                size_i=clone.size_i,
                rel=clone.rel
            ))

    dummy_abs = [(0, len(b)) for b in blobs]
    cfgl_dummy = build_cfgl_from_layouts(cfgl_off, ver, layouts, dummy_abs)
    cfgl_len = len(cfgl_dummy)

    cursor = align4(cfgl_off + cfgl_len)
    abs_off_sizes: List[Tuple[int, int]] = []
    for blob in blobs:
        abs_off_sizes.append((cursor, len(blob)))
        cursor = align4(cursor + len(blob))
    end_of_payloads = cursor

    cfgl_final = build_cfgl_from_layouts(cfgl_off, ver, layouts, abs_off_sizes)

    outb = bytearray(base)
    if len(outb) < end_of_payloads:
        outb.extend(b"\x00" * (end_of_payloads - len(outb)))

    outb[cfgl_off:cfgl_off + len(cfgl_final)] = cfgl_final

    payload_start = align4(cfgl_off + len(cfgl_final))
    outb[payload_start:end_of_payloads] = b"\x00" * (end_of_payloads - payload_start)
    for (off, sz), blob in zip(abs_off_sizes, blobs):
        outb[off:off + sz] = blob

    strip_shdr(outb)

    delta = get_delta_from_first_pt_load(base)
    ph_loads: List[Tuple[int, int, int]] = []
    ph_loads.append((cfgl_off, len(cfgl_final), delta + cfgl_off))
    for off, sz in abs_off_sizes:
        ph_loads.append((off, sz, delta + off))
    ph_loads.sort(key=lambda x: x[0])

    phoff, phnum = rebuild_phdr_table_movable(outb, ph_loads)
    ph_end = phoff + phnum * ELF64_PHDR_SIZE

    final_end = align4(max(end_of_payloads, ph_end))
    outb = outb[:final_end]

    Path(out_elf).write_bytes(outb)

    print("[+] built:", out_elf)
    print(f"    base: {base_path.name}")
    print(f"    spec: {spec_path.name}")
    if files_dir_p:
        print(f"    files-dir: {files_dir_p}")
    print(f"    CFGL @0x{cfgl_off:X} ver={ver} cfgl_len={len(cfgl_final)}")
    print(f"    entries={len(layouts)} -> PT_LOAD count={len(ph_loads)} (1 CFGL + N payloads)")
    print(f"    forced flags=0x{FORCED_P_FLAGS:X} align={FORCED_P_ALIGN}")
    print(f"    file_end=0x{len(outb):X} (truncated)")
    if pinned:
        print(f"    type_templates: {pinned}")


def main() -> None:
    ap = argparse.ArgumentParser(description="CFGL editor/list/extract tool for xbl_config*.elf files")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="Pretty-print CFGL entries from base ELF")
    p.add_argument("base_elf")

    p = sub.add_parser("extract", help="Extract CFGL payloads to a folder and generate a spec.json pointing to them")
    p.add_argument("base_elf")
    p.add_argument("out_dir")
    p.add_argument("--spec", default=None, help="Path to write the spec.json (default: <out_dir>/spec.json)")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing extracted files")

    p = sub.add_parser("build", help="Build new ELF from base+spec (reads payload files from disk)")
    p.add_argument("base_elf")
    p.add_argument("spec_json")
    p.add_argument("output_elf")
    p.add_argument("--files-dir", default=None, help="Base directory for relative 'file' paths in spec.json")

    args = ap.parse_args()

    if args.cmd == "list":
        cmd_list(args.base_elf)
    elif args.cmd == "extract":
        cmd_extract(args.base_elf, args.out_dir, args.spec, args.overwrite)
    elif args.cmd == "build":
        cmd_build(args.base_elf, args.spec_json, args.output_elf, args.files_dir)
    else:
        raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
