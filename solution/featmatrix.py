"""Full feature matrix for ML-based ceiling analysis."""
import csv
import math
import sys

import numpy as np

from parse import structure, haversine_m

csv.field_size_limit(sys.maxsize)


def safe(fn, default=np.nan):
    try:
        v = fn()
        return default if v is None else v
    except Exception:
        return default


def full_features(q):
    samples, cells = structure(q)
    n = len(samples)
    bad_idx = [i for i, s in enumerate(samples) if s["thp"] is not None and s["thp"] < 600]
    bad = [samples[i] for i in bad_idx] or samples
    bad_pcis = [s["pci"] for s in bad if s["pci"] is not None]
    serv = max(set(bad_pcis), key=bad_pcis.count) if bad_pcis else None
    sc = cells.get(serv)
    bs = [s for s in bad if s["pci"] == serv]

    f = {}
    f["n_rows"] = n
    f["n_bad"] = len(bad_idx)
    f["n_cells"] = len(cells)

    def agg(name, vals, funcs=("min", "max", "mean")):
        vals = [v for v in vals if v is not None]
        f[name + "_min"] = min(vals) if vals else np.nan
        f[name + "_max"] = max(vals) if vals else np.nan
        f[name + "_mean"] = float(np.mean(vals)) if vals else np.nan

    agg("speed_all", [s["speed"] for s in samples])
    agg("speed_bad", [s["speed"] for s in bad])
    agg("rb_all", [s["rb"] for s in samples])
    agg("rb_bad", [s["rb"] for s in bad])
    agg("thp_all", [s["thp"] for s in samples])
    agg("thp_bad", [s["thp"] for s in bad])
    agg("rsrp_bad", [s["rsrp"] for s in bs])
    agg("sinr_bad", [s["sinr"] for s in bs])
    agg("rsrp_all", [s["rsrp"] for s in samples])
    agg("sinr_all", [s["sinr"] for s in samples])

    pcis = [s["pci"] for s in samples if s["pci"] is not None]
    f["n_handover"] = sum(1 for a, b in zip(pcis, pcis[1:]) if a != b)
    f["n_unique_pci"] = len(set(pcis))

    # neighbor gaps
    top1gaps, allgaps, nnbrs = [], [], []
    mod30_strong = 0
    mod30_rows = 0
    overlap_rows = 0
    overlap_noncoloc_rows = 0
    for s in bs:
        nnbrs.append(len(s["nbrs"]))
        if s["rsrp"] is None:
            continue
        row_mod30 = False
        strong = 0
        strong_noncoloc = 0
        for j, (npci, nrsrp) in enumerate(s["nbrs"]):
            if nrsrp is not None:
                g = nrsrp - s["rsrp"]
                allgaps.append(g)
                if j == 0:
                    top1gaps.append(g)
                if g > -6:
                    strong += 1
                    nc = cells.get(npci)
                    if not (nc and sc and nc["gnb"] == sc["gnb"]):
                        strong_noncoloc += 1
            if serv is not None and npci % 30 == serv % 30:
                row_mod30 = True
                if nrsrp is not None and s["rsrp"] is not None and nrsrp - s["rsrp"] > -10:
                    mod30_strong += 1
        if row_mod30:
            mod30_rows += 1
        if strong >= 2:
            overlap_rows += 1
        if strong_noncoloc >= 1:
            overlap_noncoloc_rows += 1
    agg("top1gap", top1gaps)
    agg("allgap", allgaps)
    f["nnbrs_mean"] = float(np.mean(nnbrs)) if nnbrs else np.nan
    f["mod30_strong"] = mod30_strong
    f["mod30_rows"] = mod30_rows
    f["overlap_rows"] = overlap_rows
    f["overlap_noncoloc_rows"] = overlap_noncoloc_rows

    # serving cell geometry
    if sc:
        tilt = (sc["mech_tilt"] or 0) + sc["dig_tilt_deg"]
        f["mech_tilt"] = sc["mech_tilt"]
        f["dig_tilt_deg"] = sc["dig_tilt_deg"]
        f["dig_is_default"] = 1.0 if sc["dig_tilt_raw"] == 255 else 0.0
        f["tilt"] = tilt
        f["vbw"] = sc["vbw"]
        f["height"] = sc["height"]
        f["azimuth"] = sc["azimuth"]
        edge = (sc["height"] or 0) / math.tan(math.radians(max(tilt - sc["vbw"] / 2, 0.1)))
        edge_c = (sc["height"] or 0) / math.tan(math.radians(max(tilt, 0.1)))
        f["cov_edge"] = edge
        f["cov_edge_center"] = edge_c
        if sc["lon"] is not None:
            d = [haversine_m(s["lon"], s["lat"], sc["lon"], sc["lat"]) for s in bs if s["lon"] is not None]
            dall = [haversine_m(s["lon"], s["lat"], sc["lon"], sc["lat"]) for s in samples if s["lon"] is not None]
            agg("dist_bad", d)
            f["dist_trend_bad"] = (d[-1] - d[0]) if len(d) >= 2 else 0.0
            f["dist_trend_all"] = (dall[-1] - dall[0]) if len(dall) >= 2 else 0.0
            if d:
                f["dist_over_edge"] = max(d) / edge if edge > 0 else np.nan
                f["dist_over_edge_c"] = max(d) / edge_c if edge_c > 0 else np.nan
                f["ue_angle_far"] = math.degrees(math.atan((sc["height"] or 0) / max(max(d), 1)))
                f["beam_low_edge_minus_angle"] = (tilt - sc["vbw"] / 2) - f["ue_angle_far"]
                # azimuth offset at far bad point
                far = max(bs, key=lambda s: haversine_m(s["lon"], s["lat"], sc["lon"], sc["lat"]) if s["lon"] is not None else -1)
                if far["lon"] is not None:
                    brg = math.degrees(math.atan2(
                        math.radians(far["lon"] - sc["lon"]) * math.cos(math.radians(sc["lat"])),
                        math.radians(far["lat"] - sc["lat"])))
                    brg = brg % 360
                    az = (sc["azimuth"] or 0)
                    off = abs((brg - az + 180) % 360 - 180)
                    f["azimuth_offset"] = off
                else:
                    f["azimuth_offset"] = np.nan
            else:
                f["dist_over_edge"] = f["dist_over_edge_c"] = f["ue_angle_far"] = np.nan
                f["beam_low_edge_minus_angle"] = f["azimuth_offset"] = np.nan
    else:
        for k in ("mech_tilt", "dig_tilt_deg", "dig_is_default", "tilt", "vbw", "height",
                  "azimuth", "cov_edge", "cov_edge_center", "dist_bad_min", "dist_bad_max",
                  "dist_bad_mean", "dist_trend_bad", "dist_trend_all", "dist_over_edge",
                  "dist_over_edge_c", "ue_angle_far", "beam_low_edge_minus_angle", "azimuth_offset"):
            f[k] = np.nan

    # recovery pattern
    last_bad = bad_idx[-1] if bad_idx else n - 1
    after = samples[last_bad + 1:]
    f["recover_ho"] = 1.0 if any(
        s["pci"] is not None and s["pci"] != serv and s["thp"] is not None and s["thp"] > 600 for s in after) else 0.0
    top1s = [s["nbrs"][0][0] for s in bad if s["nbrs"]]
    top1 = max(set(top1s), key=top1s.count) if top1s else None
    f["top1_becomes_serving"] = 1.0 if (top1 is not None and any(s["pci"] == top1 for s in samples)) else 0.0
    # does the strongest neighbor belong to same gnb?
    if top1 is not None and cells.get(top1) and sc:
        f["top1_same_gnb"] = 1.0 if cells[top1]["gnb"] == sc["gnb"] else 0.0
    else:
        f["top1_same_gnb"] = np.nan

    return f


def build(path, label_col=None):
    rows = list(csv.DictReader(open(path)))
    feats = [full_features(r["question"]) for r in rows]
    keys = sorted(feats[0].keys())
    X = np.array([[np.nan if x.get(k) is None else x.get(k, np.nan) for k in keys] for x in feats], dtype=float)
    ids = [r["ID"] for r in rows]
    y = [r[label_col] for r in rows] if label_col else None
    return X, y, ids, keys
