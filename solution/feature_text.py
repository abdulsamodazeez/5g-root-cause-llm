"""Verbalize computed telemetry features into a compact diagnostic summary,
replacing raw data tables in LLM prompts. Deterministic prompt construction:
the LLM still reads the question and options and generates every answer."""
import math

from featmatrix import full_features
from family2 import panel


def _n(v, fmt="{:.0f}"):
    if v is None:
        return "n/a"
    try:
        if isinstance(v, float) and math.isnan(v):
            return "n/a"
        return fmt.format(v)
    except (ValueError, TypeError):
        return str(v)


def summary_f1(q):
    f = full_features(q)
    return "\n".join([
        "Computed diagnostic summary of the drive-test data and engineering parameters:",
        f"- Max GPS speed in low-throughput rows: {_n(f.get('speed_bad_max'))} km/h",
        f"- Mean scheduled RBs in low-throughput rows: {_n(f.get('rb_bad_mean'))}",
        f"- Serving-cell changes during the drive (handovers): {_n(f.get('n_handover'))}",
        f"- Max UE-to-serving-cell distance: {_n(f.get('dist_bad_max'))} m",
        f"- Serving cell: total downtilt {_n(f.get('tilt'))} deg, vertical beamwidth {_n(f.get('vbw'))} deg, "
        f"antenna height {_n(f.get('height'), '{:.1f}')} m, computed coverage edge {_n(f.get('cov_edge'))} m "
        f"(far-point distance / coverage edge = {_n(f.get('dist_over_edge'), '{:.2f}')})",
        f"- Strong neighbor rows sharing PCI mod 30 with serving: {_n(f.get('mod30_strong'))}",
        f"- Rows with >=2 strong co-frequency neighbors: {_n(f.get('overlap_rows'))} "
        f"(non-colocated: {_n(f.get('overlap_noncoloc_rows'))})",
        f"- Top-1 neighbor RSRP minus serving RSRP: max {_n(f.get('top1gap_max'), '{:.1f}')} dB, "
        f"mean {_n(f.get('top1gap_mean'), '{:.1f}')} dB",
        f"- Low-throughput rows: mean serving RSRP {_n(f.get('rsrp_bad_mean'), '{:.1f}')} dBm "
        f"(min {_n(f.get('rsrp_bad_min'), '{:.1f}')}), mean SINR {_n(f.get('sinr_bad_mean'), '{:.1f}')} dB",
        f"- Far-end geometry: elevation angle down to the far point {_n(f.get('ue_angle_far'), '{:.1f}')} deg vs "
        f"beam lower edge {_n(f.get('beam_low_edge_minus_angle'), '{:.1f}')} deg below it; "
        f"azimuth offset of far point from boresight {_n(f.get('azimuth_offset'), '{:.0f}')} deg; "
        f"distance trend across the low-throughput section {_n(f.get('dist_trend_bad'), '{:.0f}')} m",
    ])


def summary_f2(q):
    f = panel(q)
    return "\n".join([
        "Computed diagnostic summary of the drive-test, configuration and signaling data:",
        f"- Low-throughput rows: mean RSRP {_n(f.get('rsrp_bad'), '{:.1f}')} dBm, mean SINR {_n(f.get('sinr_bad'), '{:.1f}')} dB",
        f"- Mean CCE assignment failure rate: {_n(f.get('cce_bad'), '{:.2f}')}",
        f"- Mean scheduling grants/s: {_n(f.get('grant_bad'))}; mean MCS {_n(f.get('mcs_bad'), '{:.1f}')}; mean RB/slot {_n(f.get('rb_bad'))}",
        f"- Best neighbor RSRP minus serving RSRP: {_n(f.get('best_gap'), '{:.1f}')} dB; "
        f"strongest neighbor configured in serving cell's neighbor list: {f.get('nbr_configured')}",
        f"- Intra-freq A3 offset: {_n(f.get('a3_off'))} x0.5dB (typical 6); inter-freq A2 threshold: {_n(f.get('a2_thld'))} dBm; "
        f"PDCCH symbols: {f.get('pdcch')}",
        f"- Carrier frequencies present: {_n(f.get('n_freqs'))}",
        f"- Signaling events: A3 x{_n(f.get('n_a3'))}, handover attempts x{_n(f.get('n_ho'))}, A2 x{_n(f.get('n_a2'))}, "
        f"A5 x{_n(f.get('n_a5'))}, RRC re-establishment x{_n(f.get('n_reest'))}",
    ])


def compact_question(q):
    """Question with raw data tables replaced by the computed summary.
    Math questions pass through unchanged."""
    if "potential solutions" in q[:200]:
        return q
    if "100Mbps" in q[:200]:
        cut = q.find("**Drive Test Data**")
        head = q[:cut].rstrip().rstrip(":").rstrip() if cut > 0 else q
        head = head[: head.rfind("Given:")].rstrip() if "Given:" in head else head
        return head + "\n\n" + summary_f2(q)
    markers = [q.find("User plane drive test data as follows"),
               q.find("Engeneering parameters data as follows"),
               q.find("Engineering parameters data as follows")]
    cuts = [m for m in markers if m > 0]
    head = q[: min(cuts)].rstrip() if cuts else q
    return head + "\n\n" + summary_f1(q)
