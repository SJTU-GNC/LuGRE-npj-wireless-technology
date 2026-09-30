from pathlib import Path
import struct
import math
from collections import defaultdict


ROOT = Path(__file__).resolve().parent


def ole_stream(path, wanted=("Workbook", "Book")):
    data = path.read_bytes()
    if data[:8] != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        raise ValueError("Not an OLE compound file")
    sector_shift = struct.unpack_from("<H", data, 30)[0]
    mini_shift = struct.unpack_from("<H", data, 32)[0]
    sector_size = 1 << sector_shift
    mini_size = 1 << mini_shift
    first_dir = struct.unpack_from("<i", data, 48)[0]
    mini_cutoff = struct.unpack_from("<I", data, 56)[0]
    first_mini_fat = struct.unpack_from("<i", data, 60)[0]
    n_mini_fat = struct.unpack_from("<I", data, 64)[0]
    difat = list(struct.unpack_from("<109i", data, 76))

    def sec_off(sec):
        return (sec + 1) * sector_size

    # Extra DIFAT sectors are not expected for these files, but support the common case.
    first_difat = struct.unpack_from("<i", data, 68)[0]
    n_difat = struct.unpack_from("<I", data, 72)[0]
    sec = first_difat
    for _ in range(n_difat):
        if sec < 0:
            break
        off = sec_off(sec)
        vals = list(struct.unpack_from("<" + "i" * (sector_size // 4), data, off))
        difat.extend(vals[:-1])
        sec = vals[-1]
    fat_secs = [s for s in difat if s >= 0]
    fat = []
    for s in fat_secs:
        off = sec_off(s)
        fat.extend(struct.unpack_from("<" + "i" * (sector_size // 4), data, off))

    def chain(start):
        out, seen, s = [], set(), start
        while s >= 0 and s not in seen:
            seen.add(s)
            out.append(s)
            s = fat[s]
        return out

    def read_chain(start, size=None):
        buf = b"".join(data[sec_off(s): sec_off(s) + sector_size] for s in chain(start))
        return buf if size is None else buf[:size]

    directory = read_chain(first_dir)
    entries = []
    for i in range(0, len(directory), 128):
        ent = directory[i:i + 128]
        if len(ent) < 128:
            continue
        name_len = struct.unpack_from("<H", ent, 64)[0]
        if name_len < 2:
            continue
        name = ent[:name_len - 2].decode("utf-16le", "ignore")
        typ = ent[66]
        start = struct.unpack_from("<i", ent, 116)[0]
        size = struct.unpack_from("<Q", ent, 120)[0]
        entries.append((name, typ, start, size))

    root = next((e for e in entries if e[1] == 5), None)
    mini_stream = read_chain(root[2], root[3]) if root else b""
    mini_fat = []
    s = first_mini_fat
    for _ in range(n_mini_fat):
        if s < 0:
            break
        off = sec_off(s)
        mini_fat.extend(struct.unpack_from("<" + "i" * (sector_size // 4), data, off))
        s = fat[s]

    def read_mini_chain(start, size):
        parts, seen, s = [], set(), start
        while s >= 0 and s not in seen:
            seen.add(s)
            off = s * mini_size
            parts.append(mini_stream[off: off + mini_size])
            s = mini_fat[s]
        return b"".join(parts)[:size]

    for name in wanted:
        ent = next((e for e in entries if e[0] == name), None)
        if ent:
            if ent[3] < mini_cutoff and ent[2] >= 0:
                return read_mini_chain(ent[2], ent[3])
            return read_chain(ent[2], ent[3])
    raise ValueError("Workbook stream not found")


def read_xl_string(buf, pos):
    cch = struct.unpack_from("<H", buf, pos)[0]
    pos += 2
    flags = buf[pos]
    pos += 1
    rich = struct.unpack_from("<H", buf, pos)[0] if flags & 0x08 else 0
    if flags & 0x08:
        pos += 2
    ext = struct.unpack_from("<I", buf, pos)[0] if flags & 0x04 else 0
    if flags & 0x04:
        pos += 4
    is16 = flags & 0x01
    nbytes = cch * (2 if is16 else 1)
    raw = buf[pos: pos + nbytes]
    pos += nbytes
    text = raw.decode("utf-16le" if is16 else "latin1", "ignore")
    pos += rich * 4 + ext
    return text, pos


def rk_value(raw):
    mult100 = raw & 0x01
    is_int = raw & 0x02
    val = raw & 0xFFFFFFFC
    if is_int:
        if val & 0x80000000:
            val -= 0x100000000
        num = val >> 2
    else:
        packed = struct.pack("<II", 0, val)
        num = struct.unpack("<d", packed)[0]
    return num / 100.0 if mult100 else num


def parse_workbook(path):
    wb = ole_stream(path)
    bounds = []
    sst = []
    pos = 0
    while pos + 4 <= len(wb):
        rt, ln = struct.unpack_from("<HH", wb, pos)
        rec = wb[pos + 4: pos + 4 + ln]
        if rt == 0x0085 and len(rec) >= 8:
            off = struct.unpack_from("<I", rec, 0)[0]
            name_len = rec[6]
            flags = rec[7]
            raw = rec[8:]
            name = raw[:name_len * (2 if flags & 1 else 1)].decode("utf-16le" if flags & 1 else "latin1", "ignore")
            bounds.append((name, off))
        elif rt == 0x00FC and len(rec) >= 8:
            # Most state-vector files have simple non-split SSTs. This parser is lenient.
            total, unique = struct.unpack_from("<II", rec, 0)
            p = 8
            for _ in range(min(unique, 5000)):
                if p + 3 > len(rec):
                    break
                try:
                    txt, p = read_xl_string(rec, p)
                except Exception:
                    break
                sst.append(txt)
        pos += 4 + ln

    sheets = {}
    for name, off in bounds:
        cells = {}
        pos = off
        while pos + 4 <= len(wb):
            rt, ln = struct.unpack_from("<HH", wb, pos)
            rec = wb[pos + 4: pos + 4 + ln]
            if rt == 0x000A:
                break
            if rt == 0x00FD and len(rec) >= 10:
                r, c = struct.unpack_from("<HH", rec, 0)
                idx = struct.unpack_from("<I", rec, 6)[0]
                cells[(r, c)] = sst[idx] if idx < len(sst) else f"SST[{idx}]"
            elif rt == 0x0204 and len(rec) >= 8:
                r, c = struct.unpack_from("<HH", rec, 0)
                l = struct.unpack_from("<H", rec, 6)[0]
                cells[(r, c)] = rec[8:8 + l].decode("latin1", "ignore")
            elif rt == 0x0203 and len(rec) >= 14:
                r, c = struct.unpack_from("<HH", rec, 0)
                cells[(r, c)] = struct.unpack_from("<d", rec, 6)[0]
            elif rt == 0x027E and len(rec) >= 10:
                r, c = struct.unpack_from("<HH", rec, 0)
                raw = struct.unpack_from("<I", rec, 6)[0]
                cells[(r, c)] = rk_value(raw)
            elif rt == 0x00BD and len(rec) >= 6:
                r, c0 = struct.unpack_from("<HH", rec, 0)
                c1 = rec[-1]
                p = 4
                for c in range(c0, c1 + 1):
                    if p + 6 > len(rec) - 1:
                        break
                    raw = struct.unpack_from("<I", rec, p + 2)[0]
                    cells[(r, c)] = rk_value(raw)
                    p += 6
            pos += 4 + ln
        sheets[name] = cells
    return sheets


def summarize(path):
    sheets = parse_workbook(path)
    print(f"\n{path.name}")
    for name, cells in sheets.items():
        if not cells:
            continue
        maxr = max(r for r, c in cells)
        maxc = max(c for r, c in cells)
        print(f"  sheet={name!r} rows~{maxr+1} cols~{maxc+1}")
        for r in range(0, min(8, maxr + 1)):
            vals = [cells.get((r, c), "") for c in range(0, min(10, maxc + 1))]
            print("   ", vals)


if __name__ == "__main__":
    for p in sorted(ROOT.glob("WGC_StateVector_*.xls")):
        summarize(p)
