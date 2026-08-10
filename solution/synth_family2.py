"""Synthesize family-2-style SFT examples from the 9 reverse-engineered
scenario templates. Values are drawn around the observed cluster signatures
with noise; option letters A-I carry shuffled cause texts as in the test set.
Fully seeded."""
import json
import random
import sys

F2_TEXT = {
    "A": "RF or power parameters cause severe overlap coverage",
    "B": "Inter-frequency handover threshold configuration unreasonable",
    "C": "Network capacity insufficient or load imbalance between cells",
    "D": "Test server or transport anomaly causes insufficient upstream traffic",
    "E": "Missing neighbor cell configuration",
    "F": "RF, power parameters or site construction cause weak coverage",
    "G": "Intra-frequency handover threshold too high",
    "H": "Intra-frequency handover threshold too low",
    "I": "PDCCH resource management parameters unreasonable",
}

WHY = {
    "A": "SINR stays near 2 dB although RSRP is healthy, with several co-frequency neighbors within 6 dB in most samples: severe overlapping coverage.",
    "B": "Two carriers are configured and the inter-frequency A2/A5 thresholds are set unreasonably, driving the UE onto the poorer carrier.",
    "C": "Radio quality is good but the scheduled RB/slot collapses to ~55 at full grant volume: the cell is capacity-limited or load-imbalanced.",
    "D": "Radio quality, MCS and RB/slot are all normal but scheduling grants drop to ~400/s: the traffic source (server/transport) is not supplying data.",
    "E": "A neighbor is ~14 dB stronger and A3 keeps firing, but the neighbor is absent from the configured neighbor list, so no handover occurs and the UE re-establishes.",
    "F": "Serving RSRP falls below -100 dBm with no stronger neighbor available: weak coverage.",
    "G": "The intra-frequency A3 offset is configured at 5 dB (10 x 0.5 dB), so the handover triggers far too late while the neighbor is already ~6 dB stronger.",
    "H": "The intra-frequency A3 offset is configured at 1 dB (2 x 0.5 dB), causing repeated ping-pong handovers.",
    "I": "The CCE assignment failure rate rises to ~0.6 with PDCCH limited to one symbol: PDCCH resources are insufficient.",
}


def gen_scenario(cause, rng):
    """Return (drive_rows, cfg_rows, sig_lines) value dicts for the cause."""
    n = 15
    base = dict(rsrp=-90, sinr=12, cce=0.10, grant=1580, mcs=15.5, rb=260, gap=-8)
    ev = []
    if cause == "A":
        base.update(sinr=2.2, gap=-3.7)
    elif cause == "B":
        base.update(sinr=4.8, rb=192, grant=1170)
        ev = ["NREventA2", "NREventA5", "NRHandoverAttempt"]
    elif cause == "C":
        base.update(rb=56)
        ev = ["NRHandoverAttempt"]
    elif cause == "D":
        base.update(grant=410)
        ev = ["NRHandoverAttempt"]
    elif cause == "E":
        base.update(sinr=-2.0, gap=14)
        ev = ["NREventA3", "NREventA3", "NREventA3", "NRRRCReestablishAttempt"]
    elif cause == "F":
        base.update(rsrp=-105.5, sinr=0.0, gap=-3)
        ev = ["NREventA2"]
    elif cause == "G":
        base.update(sinr=-1.8, gap=6, mcs=9)
        ev = ["NREventA3", "NRHandoverAttempt"]
    elif cause == "H":
        base.update(sinr=3.0, gap=3.5, mcs=6.4)
        ev = ["NREventA3", "NRHandoverAttempt", "NREventA3", "NRHandoverAttempt", "NREventA3", "NRHandoverAttempt"]
    elif cause == "I":
        base.update(cce=0.59, grant=590)
        ev = ["NRHandoverAttempt"]

    a3 = 10 if cause == "G" else (2 if cause == "H" else 6)
    a2 = -95 if cause == "B" else -105
    nfreq = 2 if cause == "B" else 1
    rows = []
    for i in range(n):
        rows.append({
            "rsrp": base["rsrp"] + rng.uniform(-2, 2),
            "sinr": base["sinr"] + rng.uniform(-1, 1),
            "thp": rng.uniform(20, 95),
            "cce": max(0.0, base["cce"] + rng.uniform(-0.04, 0.04)),
            "grant": base["grant"] + rng.uniform(-25, 25),
            "mcs": base["mcs"] + rng.uniform(-1, 1),
            "rb": base["rb"] + rng.uniform(-8, 8),
            "gap": base["gap"] + rng.uniform(-1, 1),
        })
    return rows, dict(a3=a3, a2=a2, nfreq=nfreq, nbr_missing=(cause == "E")), ev


HEADER = ("| Time | UE | Longitude | Latitude | Serving PCI | Serving ARFCN | Serving RSRP(dBm) | Serving SINR(dB) | "
          "Throughput(Mbps) | Neighbor 1 PCI | Neighbor 1 RSRP(dBm) | Neighbor 2 PCI | Neighbor 2 RSRP(dBm) | "
          "Neighbor 3 PCI | Neighbor 3 RSRP(dBm) | CCE Fail Rate | Avg Rank | Grant | Avg MCS | RB/slot | "
          "Initial BLER(%) | Residual BLER(%) |")


def render(cause, rng):
    rows, cfg, ev = gen_scenario(cause, rng)
    serv = rng.choice([253, 501, 618, 777])
    nbr = rng.choice([533, 149, 641, 388])  # disjoint from static filler cells 555/976
    freq1 = 504990
    freq2 = 152650
    lines = [HEADER, "|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i, r in enumerate(rows):
        t = f"2024-09-20 22:31:{i:02d}.500"
        n1r = r["rsrp"] + r["gap"]
        lines.append(
            f"| {t} | MS1 | {-75.98 + i*1e-5:.7f} | {-27.34 - i*1e-5:.7f} | {serv} | {freq1} | {r['rsrp']:.2f} | "
            f"{r['sinr']:.2f} | {r['thp']:.2f} | {nbr} | {n1r:.2f} | 555 | {n1r - 6:.2f} | 976 | {n1r - 11:.2f} | "
            f"{r['cce']:.2f} | 2.0 | {r['grant']:.0f} | {r['mcs']:.2f} | {r['rb']:.0f} | 9.5 | 0.5 |")
    drive = "\n".join(lines)

    nbr_list = "" if cfg["nbr_missing"] else f"32{nbr}_{freq1}_{nbr},"
    cfg_tbl = (
        "| gNodeB ID | Freq(MHz) | PCI | InterFreqHoEventType | CovInterFreqA2RsrpThld(dBm) | InterFreqA2Hyst(0.5dB) | "
        "CovInterFreqA5RsrpThld1(dBm) | CovInterFreqA5RsrpThld2(dBm) | IntraFreqHoA3Offset(0.5dB) | IntraFreqHoA3Hyst(0.5dB) | "
        "IntraFreqHoA3TimeToTrig | Neighbor(gNodeB_Freq_PCI) | PdcchOccupiedSymbolNum |\n"
        "| --- | ---:| ---:| --- | ---:| ---:| ---:| ---:| ---:| ---:| --- | --- | --- |\n"
        f"| 325{serv} | {freq1} | {serv} | EVENT_A5 | {cfg['a2']} | 2 | -105 | -100 | {cfg['a3']} | 2 | ms320 | "
        f"[{nbr_list}3258492_{freq1}_555,3262798_{freq1}_976] | 1SYM |")

    param_freq2 = ""
    if cfg["nfreq"] == 2:
        param_freq2 = f"\n| 3299001 | 9 | -75.95 | -27.33 | 120 | 4 | 4 | 25.0 | TDD | 619 | n28 | {freq2} | 20M | 4T4R |"
    param_tbl = (
        "| gNodeB ID | Cell ID | Longitude | Latitude | Azimuth(deg) | Mech Tilt(deg) | Elec Tilt(deg) | Ant Height(m) | "
        "Duplex Mode | PCI | Band | DL ARFCN | BW(MHz) | TX/RX Mode |\n"
        "| --- | --- | ---:| ---:| ---:| ---:| ---:| ---:| --- | ---:| ---:| ---:| ---:| ---:|\n"
        f"| 325{serv} | 1 | -75.9468 | -27.3198 | 145 | 4 | 4 | 21.1 | TDD | {serv} | n41 | {freq1} | 100M | 64T64R |\n"
        f"| 3258492 | 3 | -75.9371 | -27.3432 | 343 | 7 | 7 | 24.4 | TDD | 555 | n41 | {freq1} | 100M | 64T64R |\n"
        f"| 3262798 | 4 | -75.9315 | -27.3437 | 290 | 2 | 2 | 28.5 | TDD | 976 | n41 | {freq1} | 100M | 64T64R |"
        + param_freq2)

    sig_lines = ["| Time | Event Name | Event Content |", "|:---|:---|:---|",
                 "| 2024-09-20 22:30:45.159 | NRRandomAccessAttempt |  |",
                 "| 2024-09-20 22:30:45.184 | NRRandomAccessSuc | Delay：25ms |"]
    for j, e in enumerate(ev):
        sig_lines.append(f"| 2024-09-20 22:31:{j+2:02d}.100 | {e} | ServCellPCI:{serv} |")
    sig = "\n".join(sig_lines)

    letters = list("ABCDEFGHI")
    causes = list(F2_TEXT)
    rng.shuffle(causes)
    opts = "\n".join(f"{l}: {F2_TEXT[c]}" for l, c in zip(letters, causes))
    label = letters[causes.index(cause)]

    prompt = (
        "Based on the following drive test data segment and engineering parameters, the test throughput drops below 100Mbps. "
        "What is the most likely root cause?\n"
        "From the following 9 potential root causes, select the most likely one and enclose its number in \\boxed{{}} in the final answer.\n"
        f"{opts}\nGiven:\n **Drive Test Data**\n{drive}\n\n**Parameter Data**\n\n{param_tbl}\n\n"
        f"**Configuration Data**\n\n{cfg_tbl}\n\n**Signaling Data**\n\n{sig}")
    completion = f"{WHY[cause]} The most likely root cause corresponds to option {label}. Final answer: \\boxed{{{label}}}"
    from feature_text import compact_question
    return {"prompt": compact_question(prompt), "completion": completion}


def main(n_per_cause=60):
    rng = random.Random(0)
    out = []
    for cause in F2_TEXT:
        for _ in range(n_per_cause):
            out.append(render(cause, rng))
    rng.shuffle(out)
    with open("sft_family2.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote sft_family2.jsonl: {len(out)} records", file=sys.stderr)


if __name__ == "__main__":
    main()
