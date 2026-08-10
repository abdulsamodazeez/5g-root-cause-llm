"""Parse Cassava RCA questions into structured records."""
import csv
import math
import re
import sys

csv.field_size_limit(sys.maxsize)


def _table_after(q, marker):
    """Extract the pipe table that follows `marker`, stopping at a blank line
    followed by non-table text or end of string."""
    m = re.search(re.escape(marker) + r"：?\s*\n", q)
    if not m:
        return None
    lines = []
    for line in q[m.end():].split("\n"):
        if "|" in line:
            lines.append(line.strip())
        elif lines:
            break
        elif line.strip():
            break
    return lines or None


def parse_question(q):
    """Split a question into drive-test rows and engineering-parameter rows.
    Section order varies across question variants."""
    dt_lines = _table_after(q, "drive test data as follows")
    ep_lines = _table_after(q, "parameters data as follows")
    if not dt_lines or not ep_lines:
        raise ValueError("data sections not found")
    dt_header = dt_lines[0].split("|")
    dt_rows = [dict(zip(dt_header, l.split("|"))) for l in dt_lines[1:]]
    ep_header = ep_lines[0].split("|")
    ep_rows = [dict(zip(ep_header, l.split("|"))) for l in ep_lines[1:]]
    return dt_rows, ep_rows


def f(v):
    """float or None"""
    if v is None:
        return None
    v = v.strip()
    if v in ("-", "", "NA", "NaN"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


DT_KEYS = {
    "speed": "GPS Speed (km/h)",
    "pci": "5G KPI PCell RF Serving PCI",
    "rsrp": "5G KPI PCell RF Serving SS-RSRP [dBm]",
    "sinr": "5G KPI PCell RF Serving SS-SINR [dB]",
    "thp": "5G KPI PCell Layer2 MAC DL Throughput [Mbps]",
    "rb": "5G KPI PCell Layer1 DL RB Num (Including 0)",
}
NBR_PCI = [
    f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} PCI" for i in range(1, 6)
]
NBR_RSRP = [
    f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} Filtered Tx BRSRP [dBm]"
    for i in range(1, 6)
]


def structure(q):
    """Return (samples, cells): cleaned drive-test samples and eng-param cells keyed by PCI."""
    dt_rows, ep_rows = parse_question(q)
    samples = []
    for r in dt_rows:
        s = {
            "lon": f(r.get("Longitude")),
            "lat": f(r.get("Latitude")),
            "speed": f(r.get(DT_KEYS["speed"])),
            "pci": f(r.get(DT_KEYS["pci"])),
            "rsrp": f(r.get(DT_KEYS["rsrp"])),
            "sinr": f(r.get(DT_KEYS["sinr"])),
            "thp": f(r.get(DT_KEYS["thp"])),
            "rb": f(r.get(DT_KEYS["rb"])),
            "nbrs": [],
        }
        for pk, rk in zip(NBR_PCI, NBR_RSRP):
            p, rs = f(r.get(pk)), f(r.get(rk))
            if p is not None:
                s["nbrs"].append((int(p), rs))
        if s["pci"] is not None:
            s["pci"] = int(s["pci"])
        samples.append(s)

    cells = {}
    for r in ep_rows:
        pci = f(r.get("PCI"))
        if pci is None:
            continue
        beam = (r.get("Beam Scenario") or "DEFAULT").strip().upper()
        m = re.search(r"(\d+)", beam)
        beam_n = int(m.group(1)) if m else 0
        if beam_n >= 12:
            vbw = 25.0
        elif beam_n >= 6:
            vbw = 12.0
        else:
            vbw = 6.0
        dig = f(r.get("Digital Tilt"))
        dig_deg = 6.0 if (dig is None or dig == 255) else dig
        cells[int(pci)] = {
            "gnb": (r.get("gNodeB ID") or "").strip(),
            "cellid": (r.get("Cell ID") or "").strip(),
            "lon": f(r.get("Longitude")),
            "lat": f(r.get("Latitude")),
            "mech_tilt": f(r.get("Mechanical Downtilt")),
            "dig_tilt_raw": dig,
            "dig_tilt_deg": dig_deg,
            "azimuth": f(r.get("Mechanical Azimuth")),
            "beam": beam,
            "vbw": vbw,
            "height": f(r.get("Height")),
        }
    return samples, cells


def load(path, answer_col=None):
    rows = list(csv.DictReader(open(path)))
    return rows
