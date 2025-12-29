#!/usr/bin/env python3
"""
qc_uefi_unwrap.py

Unwrap Qualcomm UEFI packaging:
- If input is ELF (uefi.elf / uefi.mbn), extract largest PT_LOAD payload (usually an FV).
- Parse the outer FV and locate FV_IMAGE FFS files (type 0x0B).
- Inside those, locate GUID_DEFINED sections that use Qualcomm QC-gzip GUID
  1d301fe9-be79-4353-91c2-d23bc959ae0c.
- Decompress and carve out embedded inner firmware volumes (FVs).
- Write each inner FV as inner_fv_XX.bin.

Usage:
  python3 qc_uefi_unwrap.py uefi.elf
  python3 qc_uefi_unwrap.py uefi_fv_padded.bin
"""

import gzip
import os
import struct
import sys
import uuid
from typing import Iterator, Tuple, List

try:
    from elftools.elf.elffile import ELFFile
except ImportError:
    ELFFile = None

QC_GZIP_GUID = uuid.UUID("1d301fe9-be79-4353-91c2-d23bc959ae0c")

# --- helpers ---------------------------------------------------------------

def u24(b: bytes) -> int:
    return b[0] | (b[1] << 8) | (b[2] << 16)

def align(x: int, a: int) -> int:
    return (x + (a - 1)) & ~(a - 1)

def read_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def write_file(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)

def is_elf(buf: bytes) -> bool:
    return len(buf) >= 4 and buf[:4] == b"\x7fELF"

def is_fv(buf: bytes) -> bool:
    # FV signature "_FVH" at offset 0x28
    return len(buf) >= 0x2C and buf[0x28:0x2C] == b"_FVH"

# --- ELF extraction --------------------------------------------------------

def extract_largest_pt_load_elf(path: str) -> bytes:
    if ELFFile is None:
        raise RuntimeError("pyelftools not installed. Run: pip install pyelftools")

    with open(path, "rb") as f:
        ef = ELFFile(f)
        loads = []
        for seg in ef.iter_segments():
            if seg["p_type"] == "PT_LOAD" and seg["p_filesz"] > 0:
                loads.append((seg["p_filesz"], seg["p_offset"]))
        if not loads:
            raise RuntimeError("No PT_LOAD segments found in ELF.")
        p_filesz, p_offset = max(loads, key=lambda x: x[0])
        f.seek(p_offset)
        return f.read(p_filesz)

# --- FV + FFS parsing ------------------------------------------------------

def parse_fv_header(fv: bytes) -> Tuple[uuid.UUID, int, bytes, int]:
    """
    Returns (fs_guid, fv_length, signature, header_length)
    """
    if not is_fv(fv):
        raise RuntimeError("Not a firmware volume (no _FVH signature).")
    fs_guid = uuid.UUID(bytes_le=fv[16:32])
    fv_len = struct.unpack_from("<Q", fv, 0x20)[0]
    sig = fv[0x28:0x2C]
    hdr_len = struct.unpack_from("<H", fv, 0x30)[0]
    return fs_guid, fv_len, sig, hdr_len

def iter_ffs_files(fv: bytes) -> Iterator[Tuple[int, uuid.UUID, int, int, int]]:
    """
    Iterate FFS files in an FV.
    Yields: (file_offset, file_guid, file_type, file_size, header_size)
    """
    fs_guid, fv_len, _, hdr_len = parse_fv_header(fv)
    off = hdr_len

    # Basic sanity: clamp to actual buffer length
    fv_len = min(fv_len, len(fv))

    while off + 24 <= fv_len:
        # empty space sentinel
        if fv[off:off+24] == b"\xFF" * 24:
            break

        file_guid = uuid.UUID(bytes_le=fv[off:off+16])
        # integrity byte at off+16, state at off+17
        file_type = fv[off+18]
        attrs = fv[off+19]
        size = u24(fv[off+20:off+23])

        header_size = 24
        if attrs & 0x01:  # FFS_ATTRIB_LARGE_FILE
            if off + 32 > fv_len:
                break
            size = struct.unpack_from("<Q", fv, off+24)[0]
            header_size = 32

        if size < header_size or off + size > fv_len:
            break

        yield off, file_guid, file_type, size, header_size

        off = align(off + size, 8)  # FFS files 8-byte aligned

def iter_sections(blob: bytes, start: int) -> Iterator[Tuple[int, int, int, int]]:
    """
    Iterate UEFI sections inside an FFS file body.
    Yields: (section_offset, section_size, section_type, section_header_size)
    """
    off = start
    end = len(blob)
    while off + 4 <= end:
        sz = u24(blob[off:off+3])
        stype = blob[off+3]
        hdr = 4

        if sz == 0xFFFFFF:
            if off + 8 > end:
                break
            sz = struct.unpack_from("<I", blob, off+4)[0]
            hdr = 8

        if sz == 0 or off + sz > end:
            break

        yield off, sz, stype, hdr
        off = align(off + sz, 4)  # sections 4-byte aligned

# --- GUID-defined QC gzip extraction --------------------------------------

def extract_inner_fvs_from_outer_fv(outer_fv: bytes) -> List[bytes]:
    """
    Finds FV_IMAGE FFS (type 0x0B), then within those:
      GUID_DEFINED sections using QC gzip GUID, decompresses to inner FV.
    Returns a list of inner FV byte strings (trimmed to FV length).
    """
    inner_fvs: List[bytes] = []
    _, outer_len, _, _ = parse_fv_header(outer_fv)
    outer_len = min(outer_len, len(outer_fv))

    for foff, fguid, ftype, fsize, fhdr in iter_ffs_files(outer_fv[:outer_len]):
        if ftype != 0x0B:  # EFI_FV_FILETYPE_FIRMWARE_VOLUME_IMAGE
            continue

        ffs = outer_fv[foff:foff+fsize]
        # sections begin at fhdr
        for soff, ssz, stype, shdr in iter_sections(ffs, fhdr):
            if stype != 0x02:  # EFI_SECTION_GUID_DEFINED
                continue

            # GUID-defined section header begins at (soff + shdr)
            base = soff + shdr
            if base + 20 > soff + ssz:
                continue

            sec_guid = uuid.UUID(bytes_le=ffs[base:base+16])
            data_offset = struct.unpack_from("<H", ffs, base+16)[0]

            if sec_guid != QC_GZIP_GUID:
                continue

            # section data starts at (soff + data_offset)
            comp = ffs[soff + data_offset: soff + ssz]
            try:
                decomp = gzip.decompress(comp)
            except Exception:
                continue

            # Locate embedded FV by finding "_FVH" then back up 0x28 to start of header
            idx = decomp.find(b"_FVH")
            if idx < 0 or idx < 0x28:
                continue
            fv_base = idx - 0x28
            if fv_base >= len(decomp):
                continue
            candidate = decomp[fv_base:]
            if not is_fv(candidate):
                continue

            fv_len = struct.unpack_from("<Q", candidate, 0x20)[0]
            if fv_len <= 0 or fv_len > len(candidate):
                # some builds may pad; clamp
                fv_len = min(len(candidate), fv_len) if fv_len > 0 else len(candidate)

            inner_fvs.append(candidate[:fv_len])

    return inner_fvs

# --- main ------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("usage: python3 qc_uefi_unwrap.py <uefi.elf|uefi.bin|uefi_fv.bin>")
        sys.exit(1)

    in_path = sys.argv[1]
    data = read_file(in_path)

    # Step 1: if ELF, extract PT_LOAD
    if is_elf(data):
        print("[*] Input is ELF. Extracting largest PT_LOAD...")
        outer = extract_largest_pt_load_elf(in_path)
    else:
        outer = data

    # Step 2: ensure we have an FV
    if not is_fv(outer):
        # Sometimes FV isn't at offset 0; try find _FVH
        pos = outer.find(b"_FVH")
        if pos >= 0 and pos >= 0x28:
            outer = outer[pos - 0x28:]
        if not is_fv(outer):
            raise SystemExit("Could not locate an FV (_FVH). Is this the right image?")

    # Optional: trim/pad to FV length
    _, fv_len, _, _ = parse_fv_header(outer)
    if len(outer) < fv_len:
        print(f"[*] Outer FV smaller than header length: padding to 0x{fv_len:x} with 0xFF")
        outer = outer + b"\xFF" * (fv_len - len(outer))
    else:
        outer = outer[:fv_len]

    out_dir = os.path.dirname(os.path.abspath(in_path)) or "."
    outer_path = os.path.join(out_dir, "outer_fv.bin")
    write_file(outer_path, outer)
    print(f"[*] Wrote outer FV: {outer_path} (0x{len(outer):x} bytes)")

    # Step 3: extract inner FVs from QC gzip sections
    inner_fvs = extract_inner_fvs_from_outer_fv(outer)
    if not inner_fvs:
        print("[!] No QC-gzip inner FVs found. Your image may use a different wrapper.")
        sys.exit(2)

    for i, fv in enumerate(inner_fvs):
        p = os.path.join(out_dir, f"inner_fv_{i:02d}.bin")
        write_file(p, fv)
        print(f"[+] Wrote inner FV #{i}: {p} (0x{len(fv):x} bytes)")

    print("\nDone. Open inner_fv_*.bin in UEFITool NE.")

if __name__ == "__main__":
    main()
