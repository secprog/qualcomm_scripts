#!/usr/bin/env python3
"""
Make a "bare" ELF containing only the PT_LOAD payload, using a template ELF
to copy machine/entry/load addresses/flags/alignment.

This strips Qualcomm auth segments (PT_NULL etc.) by rebuilding a new ELF with:
- 1x PT_LOAD
- no section headers

Usage:
  python3 make_unsigned_elf_from_template.py uefi_old.elf bnmm.rom uefi_unsigned.elf
"""

import sys
import struct

PT_LOAD = 1

def align_up(x, a):
    if a <= 1:
        return x
    return (x + (a - 1)) & ~(a - 1)

def main(template_elf, fv_blob, out_elf):
    tmpl = open(template_elf, "rb").read()
    fv   = open(fv_blob, "rb").read()

    if tmpl[:4] != b"\x7fELF" or tmpl[4] != 2 or tmpl[5] != 1:
        raise SystemExit("Template must be ELF64 little-endian")

    # ELF64 header
    (e_type, e_machine, e_version, e_entry, e_phoff, e_shoff, e_flags,
     e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx) = struct.unpack_from(
        "<HHIQQQIHHHHHH", tmpl, 16
    )

    # Find first PT_LOAD in template
    load = None
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align = struct.unpack_from(
            "<IIQQQQQQ", tmpl, off
        )
        if p_type == PT_LOAD and p_filesz > 0:
            load = (p_flags, p_vaddr, p_paddr, p_align or 0x1000)
            break

    if not load:
        raise SystemExit("No PT_LOAD found in template")

    p_flags, p_vaddr, p_paddr, p_align = load

    # Build a new ELF with exactly 1 PT_LOAD at file offset 0x1000
    e_ident = tmpl[:16]
    ehdr_size = 64
    phent_size = 56
    phnum = 1
    phoff = ehdr_size

    header_total = ehdr_size + phnum * phent_size
    seg_off = align_up(header_total, 0x1000)  # conventional page alignment

    # ELF header: no section headers
    ehdr = struct.pack(
        "<16sHHIQQQIHHHHHH",
        e_ident,
        e_type,
        e_machine,
        e_version,
        e_entry,
        phoff,
        0,          # e_shoff
        e_flags,
        ehdr_size,
        phent_size,
        phnum,
        0, 0, 0     # no sections
    )

    phdr = struct.pack(
        "<IIQQQQQQ",
        PT_LOAD,
        p_flags,
        seg_off,
        p_vaddr,
        p_paddr,
        len(fv),
        len(fv),
        p_align
    )

    out = bytearray()
    out += ehdr
    out += phdr

    # pad to seg_off
    if len(out) < seg_off:
        out += b"\x00" * (seg_off - len(out))

    out[seg_off:seg_off+len(fv)] = fv

    with open(out_elf, "wb") as f:
        f.write(out)

    print(f"[+] wrote {out_elf}")
    print(f"    entry=0x{e_entry:x}, loadaddr=0x{p_paddr:x}, payload_size=0x{len(fv):x}, payload_off=0x{seg_off:x}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: make_unsigned_elf_from_template.py <template.elf> <fv.bin> <out.elf>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3]) 