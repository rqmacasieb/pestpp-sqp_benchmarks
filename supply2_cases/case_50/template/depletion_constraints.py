import os
import pandas as pd

STRDEP_CONSTRAINTS = [('c1', 's1r14', (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 15000.0), ('c2', 's1r21', (2, 3, 6, 7, 10, 11), 20000.0), ('c3', 's2r08', (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 15000.0), ('c4', 's3r05', (2, 3, 6, 7, 10, 11), 30000.0)]


def apply(sim_ws="."):
    """Depletion at each STRDEP_CONSTRAINTS reach: baseline (zero-pumping)
    SFR flow minus the current scenario's SFR flow, maxed over that
    constraint's stress periods. Writes sim_ws/strdep.dat and returns the
    {id: value} dict."""
    cur = pd.read_csv(os.path.join(sim_ws, "sfr_ex.csv")).drop(columns="time")
    base = pd.read_csv(os.path.join(sim_ws, "sfr_ex_baseline.csv")).drop(columns="time")
    cur.columns = cur.columns.str.lower()
    base.columns = base.columns.str.lower()
    # MF6's "outflow" SFR obs is signed negative (flow leaving the reach's
    # control volume), the opposite convention from the legacy .sfrout's
    # "FLOW OUT OF STRM. RCH." column (reported as a positive magnitude) --
    # so depletion (reduction in the positive-valued streamflow) is
    # (-base) - (-cur) == cur - base, not base - cur.
    delta = cur - base
    values = {}
    for cid, col, periods, _ in STRDEP_CONSTRAINTS:
        values[cid] = float(delta[col].iloc[list(periods)].max())
    with open(os.path.join(sim_ws, "strdep.dat"), "w") as f:
        for cid, _, _, _ in STRDEP_CONSTRAINTS:
            f.write(f"{cid} {values[cid]}\n")
    return values
