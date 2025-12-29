#!/usr/bin/env python3
import argparse
import hashlib
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

PT_LOAD = 1

def align_up(x: int, a: int) -> int:
    return (x + (a - 1)) & ~(a - 1)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def is_probably_ascii_sig(sig: bytes) -> bool:
    if len(sig) != 4:
        return False
    return all((48 <= b <= 57) or (65 <= b <= 90) or (97 <= b <= 122) or b == 95 for b in sig)

def acpi_checksum_byte(buf: bytes) -> int:
    """Return buf sum modulo 256."""
    return sum(buf) & 0xFF

def patch_acpi_table_checksum(table: bytes) -> Tuple[bytes, int, int]:
    """
    Patch standard ACPI table checksum (byte offset 9 in the standard 36-byte ACPI header).
    Returns (patched_table, old_ck, new_ck).
    If table is too small, returns original.
    """
    if len(table) < 36:
        return table, -1, -1

    old = table[9]
    b = bytearray(table)
    b[9] = 0
    new = (-sum(b)) & 0xFF
    b[9] = new
    return bytes(b), old, new

@dataclass
class Phdr:
    p_type: int
    p_offset: int
    p_vaddr: int
    p_paddr: int
    p_filesz: int
    p_memsz: int
    p_flags: int
    p_align: int

@dataclass
class AcpiTable:
    idx: int
    sig: str
    rel_off: int          # offset within payload/segment
    file_off: int         # absolute file offset
    length: int
    alloc: int
    checksum_old: Optional[int] = None
    checksum_new: Optional[int] = None
    checksum_ok_before: Optional[bool] = None
    checksum_ok_after: Optional[bool] = None

def parse_elf32_headers(blob: bytes) -> Tuple[dict, List[Phdr]]:
    if len(blob) < 52:
        raise ValueError("Too small to be ELF32")
    if blob[0:4] != b"\x7fELF":
        raise ValueError("Not an ELF file")
    # e_ident[4] = 1 => 32-bit, e_ident[5] = 1 => little-endian
    if blob[4] != 1 or blob[5] != 1:
        raise ValueError("This script expects ELF32 little-endian")

    eh = struct.unpack_from("<16sHHIIIIIHHHHHH", blob, 0)
    keys = [
        "e_ident","e_type","e_machine","e_version","e_entry",
        "e_phoff","e_shoff","e_flags","e_ehsize","e_phentsize",
        "e_phnum","e_shentsize","e_shnum","e_shstrndx"
    ]
    hdr = dict(zip(keys, eh))

    phdrs: List[Phdr] = []
    phoff = hdr["e_phoff"]
    phentsize = hdr["e_phentsize"]
    phnum = hdr["e_phnum"]

    if phoff + phnum * phentsize > len(blob):
        raise ValueError("Program header table extends beyond file")

    for i in range(phnum):
        off = phoff + i * phentsize
        p = struct.unpack_from("<IIIIIIII", blob, off)
        phdrs.append(Phdr(*p))

    return hdr, phdrs

def find_acpi_toc_in_segment(seg: bytes) -> Optional[int]:
    """
    Find an ACPI TOC. In your file it starts at the very start of the segment, but
    we also scan the first 4 KiB just in case.
    Returns offset within seg, or None.
    """
    scan_limit = min(len(seg), 0x1000)
    for off in range(0, scan_limit - 16 + 1, 4):
        if seg[off:off+4] != b"ACPI":
            continue
        # expect: magic + count + first_off + first_len
        count = struct.unpack_from("<I", seg, off + 4)[0]
        first_off = struct.unpack_from("<I", seg, off + 8)[0]
        first_len = struct.unpack_from("<I", seg, off + 12)[0]
        # sanity:
        if not (1 <= count <= 128):
            continue
        if first_off == 0 or first_off > len(seg):
            continue
        if first_len == 0 or first_len > len(seg):
            continue
        # check that the first table signature at first_off looks plausible
        if first_off + 4 <= len(seg):
            sig = seg[first_off:first_off+4]
            if not is_probably_ascii_sig(sig):
                continue
        return off
    return None

def select_acpi_load_segment(blob: bytes, phdrs: List[Phdr]) -> Tuple[Phdr, bytes, int]:
    """
    Return (phdr, segment_bytes, toc_off_in_segment)
    Picks the first PT_LOAD segment that contains a plausible ACPI TOC.
    """
    for ph in phdrs:
        if ph.p_type != PT_LOAD or ph.p_filesz <= 0:
            continue
        if ph.p_offset + ph.p_filesz > len(blob):
            continue
        seg = blob[ph.p_offset:ph.p_offset + ph.p_filesz]
        toc_off = find_acpi_toc_in_segment(seg)
        if toc_off is not None:
            return ph, seg, toc_off
    raise ValueError("Could not find an ACPI TOC in any PT_LOAD segment")

def parse_acpi_toc(seg: bytes, toc_off: int, ph_file_off: int) -> Tuple[bytes, List[AcpiTable]]:
    """
    TOC format (as seen in your blob):
      u8  magic[4] = "ACPI"
      u32 count
      u32 first_table_off
      u32 first_table_len
      then (count-1) entries, each:
         u32 off
         u32 len
         u32 alloc
    Tables live in the same segment; signatures are read from each table blob.
    Returns (toc_page_bytes, tables)
    """
    if seg[toc_off:toc_off+4] != b"ACPI":
        raise ValueError("TOC magic mismatch")
    count = struct.unpack_from("<I", seg, toc_off + 4)[0]
    first_off = struct.unpack_from("<I", seg, toc_off + 8)[0]
    first_len = struct.unpack_from("<I", seg, toc_off + 12)[0]

    toc_page = seg[toc_off:first_off]

    tables: List[AcpiTable] = []

    # Table 0
    t0_sig = seg[first_off:first_off+4].decode("ascii", errors="replace")
    t0_alloc = align_up(first_len, 0x1000)
    tables.append(AcpiTable(
        idx=0,
        sig=t0_sig,
        rel_off=first_off,
        file_off=ph_file_off + first_off,
        length=first_len,
        alloc=t0_alloc,
    ))

    entry_base = toc_off + 0x10
    entry_size = 12
    needed = entry_base + (count - 1) * entry_size
    if needed > len(seg):
        raise ValueError(f"TOC claims {count} tables but entries run past segment end")

    for i in range(1, count):
        off = struct.unpack_from("<I", seg, entry_base + (i - 1) * entry_size + 0)[0]
        ln  = struct.unpack_from("<I", seg, entry_base + (i - 1) * entry_size + 4)[0]
        al  = struct.unpack_from("<I", seg, entry_base + (i - 1) * entry_size + 8)[0]
        if off + ln > len(seg):
            raise ValueError(f"Table {i} extends beyond segment (off=0x{off:x}, len=0x{ln:x})")
        sig = seg[off:off+4].decode("ascii", errors="replace")
        tables.append(AcpiTable(
            idx=i,
            sig=sig,
            rel_off=off,
            file_off=ph_file_off + off,
            length=ln,
            alloc=al,
        ))

    tables.sort(key=lambda t: t.rel_off)
    return toc_page, tables

def write_zip_from_dir(zip_path: Path, folder: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(folder)))

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract ACPI tables from an ACPI.elf-like ELF container and optionally patch checksums."
    )
    ap.add_argument("elf", type=Path, help="Path to ACPI.elf")
    ap.add_argument("--out", type=Path, default=Path("."), help="Output base directory (default: current dir)")
    ap.add_argument("--prefix", type=str, default="acpi", help="Output name prefix (default: acpi)")
    ap.add_argument("--write-patched-elf", action="store_true", help="Also write a patched ELF with updated checksums in-place")
    args = ap.parse_args()

    elf_path: Path = args.elf
    out_base: Path = args.out
    prefix: str = args.prefix

    blob = elf_path.read_bytes()
    hdr, phdrs = parse_elf32_headers(blob)
    ph, seg, toc_off = select_acpi_load_segment(blob, phdrs)
    toc_page, tables = parse_acpi_toc(seg, toc_off, ph.p_offset)

    print(f"[+] File: {elf_path}")
    print(f"[+] SHA-256: {sha256_file(elf_path)}")
    print(f"[+] ELF32 machine={hdr['e_machine']} entry=0x{hdr['e_entry']:08x} phnum={hdr['e_phnum']}")
    print(f"[+] Using PT_LOAD: file_off=0x{ph.p_offset:x} vaddr=0x{ph.p_vaddr:08x} filesz=0x{ph.p_filesz:x}")
    print(f"[+] ACPI TOC offset in segment: 0x{toc_off:x}")
    print(f"[+] Tables parsed: {len(tables)}")
    print()

    raw_dir = out_base / f"{prefix}_extract"
    patched_dir = out_base / f"{prefix}_extract_patched"
    raw_dir.mkdir(parents=True, exist_ok=True)
    patched_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "00_TOC.bin").write_bytes(toc_page)
    (patched_dir / "00_TOC.bin").write_bytes(toc_page)

    patched_blob = bytearray(blob) if args.write_patched_elf else None

    for n, t in enumerate(tables, start=1):
        table_bytes = seg[t.rel_off:t.rel_off + t.length]
        sig = t.sig

        if sig != "FACS" and len(table_bytes) >= 36:
            ok_before = (acpi_checksum_byte(table_bytes) == 0)
            t.checksum_ok_before = ok_before
            t.checksum_old = table_bytes[9]

            patched_table, old_ck, new_ck = patch_acpi_table_checksum(table_bytes)
            ok_after = (acpi_checksum_byte(patched_table) == 0)
            t.checksum_new = new_ck
            t.checksum_ok_after = ok_after

            patched_bytes = patched_table
        else:
            patched_bytes = table_bytes

        fname = f"{n:02d}_{sig}.bin"
        (raw_dir / fname).write_bytes(table_bytes)
        (patched_dir / fname).write_bytes(patched_bytes)

        if patched_blob is not None and patched_bytes != table_bytes:
            file_off = t.file_off
            patched_blob[file_off:file_off + t.length] = patched_bytes

    raw_zip = out_base / f"{prefix}_extract.zip"
    patched_zip = out_base / f"{prefix}_extract_patched.zip"
    write_zip_from_dir(raw_zip, raw_dir)
    write_zip_from_dir(patched_zip, patched_dir)

    if patched_blob is not None:
        patched_elf_path = out_base / f"{prefix}_patched.elf"
        patched_elf_path.write_bytes(bytes(patched_blob))
        print(f"[+] Wrote patched ELF: {patched_elf_path}")

    print("Idx  Sig   RelOff     FileOff    Length    Alloc     Ck(old->new)   SumOK(before->after)")
    print("---- ----- ---------- ---------- --------- --------- ------------- --------------------")
    for t in tables:
        if t.sig == "FACS":
            ck = "(FACS n/a)"
            ok = "(n/a)"
        elif t.checksum_old is None:
            ck = "(nohdr)"
            ok = "(n/a)"
        else:
            ck = f"{t.checksum_old:02x}->{t.checksum_new:02x}"
            ok = f"{'ok' if t.checksum_ok_before else 'BAD'}->{'ok' if t.checksum_ok_after else 'BAD'}"
        print(f"{t.idx:>3}  {t.sig:<5}  0x{t.rel_off:08x}  0x{t.file_off:08x}  {t.length:>9}  0x{t.alloc:05x}   {ck:<13} {ok}")

    print()
    print(f"[+] Raw extracted dir:      {raw_dir}")
    print(f"[+] Patched extracted dir:  {patched_dir}")
    print(f"[+] Raw zip:                {raw_zip}")
    print(f"[+] Patched zip:            {patched_zip}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
