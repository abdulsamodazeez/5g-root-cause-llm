"""Family-2 (9-cause, <100Mbps) parsing and diagnosis."""
import csv
import re
import sys

import numpy as np

csv.field_size_limit(sys.maxsize)


def md_table(block):
    lines = [l.strip() for l in block.strip().split("\n") if l.strip().startswith("|")]
    if len(lines) < 2:
        return []
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for l in lines[1:]:
        if re.match(r"^\|[\s:\-|]+\|$", l):
            continue
        cells = [c.strip() for c in l.strip("|").split("|")]
        rows.append(dict(zip(header, cells)))
    return rows


def sections(q):
    out = {}
    for name in ["Drive Test Data", "Parameter Data", "Configuration Data", "Signaling Data"]:
        m = re.search(r"\*\*" + name + r"\*\*\s*\n(.*?)(?=\n\*\*|\Z)", q, re.S)
        if m:
            out[name] = m.group(1)
    return out


def fnum(v):
    try:
        return float(str(v).replace("M", "").replace("dBm", ""))
    except (ValueError, TypeError):
        return None


def parse_f2(q):
    sec = sections(q)
    dt = md_table(sec.get("Drive Test Data", ""))
    pd_ = md_table(sec.get("Parameter Data", ""))
    cfg = md_table(sec.get("Configuration Data", ""))
    sig = md_table(sec.get("Signaling Data", ""))

    drive = []
    for r in dt:
        row = {
            "time": r.get("Time", ""),
            "ue": r.get("UE", ""),
            "pci": fnum(r.get("Serving PCI")),
            "arfcn": fnum(r.get("Serving ARFCN")),
            "rsrp": fnum(r.get("Serving RSRP(dBm)")),
            "sinr": fnum(r.get("Serving SINR(dB)")),
            "thp": fnum(r.get("Throughput(Mbps)")),
            "cce": fnum(r.get("CCE Fail Rate")),
            "rank": fnum(r.get("Avg Rank")),
            "grant": fnum(r.get("Grant")),
            "mcs": fnum(r.get("Avg MCS")),
            "rb": fnum(r.get("RB/slot")),
            "ibler": fnum(r.get("Initial BLER(%)")),
            "rbler": fnum(r.get("Residual BLER(%)")),
            "nbrs": [],
        }
        for i in (1, 2, 3):
            p = fnum(r.get(f"Neighbor {i} PCI"))
            rs = fnum(r.get(f"Neighbor {i} RSRP(dBm)"))
            if p is not None:
                row["nbrs"].append((int(p), rs))
        if row["pci"] is not None:
            row["pci"] = int(row["pci"])
        drive.append(row)

    cells = {}
    for r in pd_:
        pci = fnum(r.get("PCI"))
        if pci is None:
            continue
        cells[int(pci)] = {
            "gnb": r.get("gNodeB ID", ""),
            "lon": fnum(r.get("Longitude")),
            "lat": fnum(r.get("Latitude")),
            "band": r.get("Band", ""),
            "arfcn": fnum(r.get("DL ARFCN")),
            "height": fnum(r.get("Ant Height(m)")),
            "mech": fnum(r.get("Mech Tilt(deg)")),
            "elec": fnum(r.get("Elec Tilt(deg)")),
        }

    conf = {}
    for r in cfg:
        pci = fnum(r.get("PCI"))
        if pci is None:
            continue
        nbr_raw = r.get("Neighbor(gNodeB_Freq_PCI)", "")
        nbr_pcis = [int(x.split("_")[-1]) for x in re.findall(r"\d+_\d+_\d+", nbr_raw)]
        conf[int(pci)] = {
            "ho_event": r.get("InterFreqHoEventType", ""),
            "a2_thld": fnum(r.get("CovInterFreqA2RsrpThld(dBm)")),
            "a5_t1": fnum(r.get("CovInterFreqA5RsrpThld1(dBm)")),
            "a5_t2": fnum(r.get("CovInterFreqA5RsrpThld2(dBm)")),
            "a3_off": fnum(r.get("IntraFreqHoA3Offset(0.5dB)")),
            "a3_hyst": fnum(r.get("IntraFreqHoA3Hyst(0.5dB)")),
            "a3_ttt": r.get("IntraFreqHoA3TimeToTrig", ""),
            "nbr_pcis": nbr_pcis,
            "pdcch": r.get("PdcchOccupiedSymbolNum", ""),
        }

    events = [{"time": r.get("Time", ""), "name": r.get("Event Name", ""), "content": r.get("Event Content", "")} for r in sig]
    return drive, cells, conf, events


def panel(q):
    drive, cells, conf, events = parse_f2(q)
    bad = [r for r in drive if r["thp"] is not None and r["thp"] < 100]
    if not bad:
        bad = sorted([r for r in drive if r["thp"] is not None], key=lambda r: r["thp"])[:3]
    bad_pcis = [r["pci"] for r in bad if r["pci"] is not None]
    serv = max(set(bad_pcis), key=bad_pcis.count) if bad_pcis else None
    sconf = conf.get(serv, {})
    scell = cells.get(serv, {})

    def m(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    f = {"serv": serv, "n": len(drive), "n_bad": len(bad)}
    f["rsrp_bad"] = m([r["rsrp"] for r in bad])
    f["sinr_bad"] = m([r["sinr"] for r in bad])
    f["cce_bad"] = m([r["cce"] for r in bad])
    f["grant_bad"] = m([r["grant"] for r in bad])
    f["mcs_bad"] = m([r["mcs"] for r in bad])
    f["rb_bad"] = m([r["rb"] for r in bad])
    f["ibler_bad"] = m([r["ibler"] for r in bad])
    f["rbler_bad"] = m([r["rbler"] for r in bad])
    f["rank_bad"] = m([r["rank"] for r in bad])
    f["thp_bad"] = m([r["thp"] for r in bad])

    # neighbor situation during bad rows
    best_gap = None
    best_nbr = None
    n_close = 0
    for r in bad:
        if r["rsrp"] is None:
            continue
        for (p, rs) in r["nbrs"]:
            if rs is None:
                continue
            g = rs - r["rsrp"]
            if best_gap is None or g > best_gap:
                best_gap, best_nbr = g, p
            if g > -6 and p != r["pci"]:
                n_close += 1
    f["best_gap"] = best_gap
    f["best_nbr"] = best_nbr
    f["n_close_nbr_rows"] = n_close

    # is the strongest neighbor configured as a neighbor of serving?
    f["nbr_configured"] = None
    if best_nbr is not None and sconf:
        f["nbr_configured"] = best_nbr in sconf.get("nbr_pcis", [])
    # is it in parameter data at all / same freq?
    f["best_nbr_known"] = best_nbr in cells if best_nbr is not None else None
    if best_nbr is not None and best_nbr in cells and scell:
        f["best_nbr_same_freq"] = cells[best_nbr]["arfcn"] == scell.get("arfcn")
    else:
        f["best_nbr_same_freq"] = None

    # config values
    f["a3_off"] = sconf.get("a3_off")
    f["a3_hyst"] = sconf.get("a3_hyst")
    f["a3_ttt"] = sconf.get("a3_ttt")
    f["a2_thld"] = sconf.get("a2_thld")
    f["a5_t1"] = sconf.get("a5_t1")
    f["a5_t2"] = sconf.get("a5_t2")
    f["pdcch"] = sconf.get("pdcch")
    f["ho_event"] = sconf.get("ho_event")

    # config anomalies vs modal values across cells
    def modal(key):
        vals = [c.get(key) for c in conf.values() if c.get(key) is not None]
        if not vals:
            return None
        return max(set(vals), key=vals.count)

    f["a3_off_modal"] = modal("a3_off")
    f["a2_modal"] = modal("a2_thld")
    f["a5t1_modal"] = modal("a5_t1")
    f["a5t2_modal"] = modal("a5_t2")

    # signaling summary
    names = [e["name"] for e in events]
    f["n_a3"] = names.count("NREventA3")
    f["n_ho"] = names.count("NRHandoverAttempt")
    f["n_a2"] = names.count("NREventA2")
    f["n_a5"] = names.count("NREventA5")
    f["n_reest"] = names.count("NRRRCReestablishAttempt")
    # ping-pong: same pair of PCIs handed back and forth
    hos = [e for e in events if e["name"] == "NRHandoverAttempt"]
    pairs = []
    for e in hos:
        mm = re.findall(r"\d+", e["content"])
        if len(mm) >= 2:
            pairs.append((mm[0], mm[1]))
    pp = 0
    for i in range(1, len(pairs)):
        if pairs[i] == (pairs[i - 1][1], pairs[i - 1][0]):
            pp += 1
    f["pingpong"] = pp
    f["n_freqs"] = len(set(c["arfcn"] for c in cells.values() if c["arfcn"] is not None))
    return f
