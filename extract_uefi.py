#!/usr/bin/env python3
import struct, sys

PT_LOAD = 1

def extract_fv(in_elf, out_bin):
    with open(in_elf, "rb") as f:
        ident = f.read(16)
        if ident[:4] != b"\x7fELF":
            raise SystemExit("Not an ELF file")

        # Read ELF header (assume little-endian; Qualcomm images are)
        ehdr = f.read(64-16)
        e_phoff = struct.unpack_from("<Q", ehdr, 16)[0]
        e_phentsize = struct.unpack_from("<H", ehdr, 38)[0]
        e_phnum = struct.unpack_from("<H", ehdr, 40)[0]

        # Read program headers, pick largest PT_LOAD
        f.seek(e_phoff)
        loads = []
        for _ in range(e_phnum):
            p = f.read(e_phentsize)
            p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align = struct.unpack("<IIQQQQQQ", p[:56])
            if p_type == PT_LOAD and p_filesz > 0:
                loads.append((p_filesz, p_offset))
        if not loads:
            raise SystemExit("No PT_LOAD segments found")

        p_filesz, p_offset = max(loads, key=lambda x: x[0])
        f.seek(p_offset)
        data = f.read(p_filesz)

    # If it looks like an FV, pad to FV header's FvLength
    if len(data) >= 0x30 and data[0x28:0x2C] == b"_FVH":
        fv_len = struct.unpack_from("<Q", data, 32)[0]
        if fv_len > len(data):
            data += b"\xFF" * (fv_len - len(data))

    with open(out_bin, "wb") as out:
        out.write(data)

    print(f"Wrote {out_bin} ({len(data)} bytes). _FVH @ 0x28 = {data[0x28:0x2C] == b'_FVH'}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} uefi.elf uefi_fv.bin")
        sys.exit(1)
    extract_fv(sys.argv[1], sys.argv[2])
