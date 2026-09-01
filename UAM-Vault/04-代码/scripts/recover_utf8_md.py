# -*- coding: utf-8 -*-
"""Recover MD files corrupted by PowerShell Set-Content truncating UTF-8 trailing bytes to '?'."""
from pathlib import Path

KNOWN = {
    (0xE4, 0xBB): 0x85,  # 仅
    (0xE5, 0x90): 0x8D,  # 名
}


def recover(path: Path) -> bool:
    b = bytearray(path.read_bytes())
    fixes = 0
    i = 0
    while i < len(b) - 2:
        if b[i] == 0xEF and b[i + 1] == 0xBC and b[i + 2] == 0x3F:
            nxt = bytes(b[i + 3 : i + 10])
            # ： before * / English / AC ; ， before v1
            if nxt.startswith(b"v") or b"v1" in nxt[:4]:
                third = 0x8C  # ，
            elif nxt.startswith(b"\n") or nxt.startswith(b"\r"):
                third = 0x8C
            else:
                third = 0x9A  # ：
            b[i + 2] = third
            fixes += 1
            i += 3
            continue
        if b[i] == 0xE2 and b[i + 1] == 0x80 and b[i + 2] == 0x3F:
            b[i + 2] = 0x94  # —
            fixes += 1
            i += 3
            continue
        if b[i] == 0xE3 and b[i + 1] == 0x80 and b[i + 2] == 0x3F:
            nxt = bytes(b[i + 3 : i + 6])
            b[i + 2] = 0x82 if nxt[:1] in (b"\n", b"\r") else 0x81
            fixes += 1
            i += 3
            continue
        if (
            0xE4 <= b[i] <= 0xE9
            and 0x80 <= b[i + 1] <= 0xBF
            and b[i + 2] == 0x3F
        ):
            key = (b[i], b[i + 1])
            best = KNOWN.get(key)
            if best is None:
                for t in range(0x80, 0xC0):
                    try:
                        ch = bytes([b[i], b[i + 1], t]).decode("utf-8")
                    except Exception:
                        continue
                    o = ord(ch)
                    if 0x4E00 <= o <= 0x9FFF:
                        best = t
                        break
            if best is None:
                best = 0x80
            b[i + 2] = best
            fixes += 1
            i += 3
            continue
        i += 1

    out = bytes(b)
    path.write_bytes(out)
    try:
        text = out.decode("utf-8")
        print(path.name, "fixes", fixes, "OK", "len", len(text))
        print("head:", text[:100].replace("\n", " | "))
        return True
    except UnicodeDecodeError as e:
        print(path.name, "fixes", fixes, "STILL BAD", e)
        return False


if __name__ == "__main__":
    for p in [
        Path(r"f:\UAM\paper\ac_dsgf_cn\AC_DSGF_CN.md"),
        Path(r"f:\UAM\paper\ac_dsgf\AC_DSGF_EN.md"),
    ]:
        recover(p)
