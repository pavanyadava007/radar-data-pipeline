"""Render docs/RESULTS.md, docs/figures/*.png, the README results block and site/index.html.

Every number comes from results/*.json or results/sql_*.csv. Figures that need the
derived data (example range-Doppler maps, spectrograms, SNR distribution) are only
re-rendered when data/ exists; otherwise the committed PNGs are kept.
"""

from __future__ import annotations

import csv
import html
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rdp import config as C  # noqa: E402
from rdp import dsp  # noqa: E402

RES = C.paths().results
FIG = ROOT / "docs" / "figures"
SITE = ROOT / "site"
# reference categorical palette, fixed order (class group identity)
SERIES = {"drone": "#2a78d6", "bird": "#eb6834", "human": "#1baf7a", "reflector": "#eda100"}
METHOD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CMAP = "Blues"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
    }
)


def j(name):
    return json.loads((RES / name).read_text())


def csv_rows(name):
    with open(RES / name) as f:
        return list(csv.DictReader(f))


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x):
    return f"{100 * x:.1f} %"


# ------------------------------------------------------------------ data-dependent figures
def data_figures():
    p = C.paths()
    if not (p.h5.exists() and p.features.exists()):
        print("[report] derived data missing, keeping committed data figures")
        return
    import h5py
    import pandas as pd

    df = pd.read_parquet(p.features)
    clean = df[(df.edge_flag == 0)]
    picks = {}
    for g in C.CLASS_GROUPS:
        d = clean[clean.class_group == g]
        med = d.snr_db.median()
        picks[g] = int(d.iloc[(d.snr_db - med).abs().argsort().iloc[0]].segment_id)
    with h5py.File(p.h5, "r") as f:
        segs = {g: f["iq"][i] for g, i in picks.items()}
    v = dsp.doppler_axis_mps()

    fig, axes = plt.subplots(4, 1, figsize=(7.5, 6.2), sharex=True)
    for ax, (g, x) in zip(axes, segs.items()):
        _, rd = dsp.features(x[None])
        im = ax.imshow(
            rd[0].astype(np.float32),
            aspect="auto",
            cmap=CMAP,
            vmin=0,
            vmax=50,
            extent=[v[0], v[-1], 4.5, -0.5],
            interpolation="nearest",
        )
        lab = df.label[df.segment_id == picks[g]].iloc[0]
        ax.set_title(f"{g} ({lab}, segment {picks[g]}, median-SNR example)", loc="left")
        ax.set_ylabel("range cell")
    axes[-1].set_xlabel("radial velocity (m/s), 77 GHz, PRF 17 kHz")
    fig.colorbar(im, ax=axes, label="dB re noise floor", shrink=0.6)
    fig.savefig(FIG / "rd_examples.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(10, 3.0), sharey=True)
    for ax, (g, x) in zip(axes, segs.items()):
        s = dsp.spectrogram(x[None, C.CENTRE_CELL])[0]
        s = 10 * np.log10(s / (np.median(s) / np.log(2)) + 1e-12)
        t_ms = (np.arange(s.shape[0]) * C.STFT_HOP + C.STFT_NPERSEG / 2) / C.PRF_HZ * 1e3
        fs = dsp.doppler_axis_hz(C.STFT_NPERSEG) / 1e3
        ax.imshow(s.T, aspect="auto", origin="lower", cmap=CMAP, vmin=0, vmax=40, extent=[t_ms[0], t_ms[-1], fs[0], fs[-1]])
        ax.set_title(g, loc="left")
        ax.set_xlabel("time (ms)")
    axes[0].set_ylabel("Doppler (kHz)")
    fig.suptitle(
        "Micro-Doppler spectrogram of the centre range cell (STFT 64 / hop 16, dB re noise)", x=0.01, ha="left", fontsize=9.5
    )
    fig.savefig(FIG / "spectrograms.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.0))
    bins = np.linspace(df.snr_db.min(), df.snr_db.max(), 60)
    for g in C.CLASS_GROUPS:
        d = df.snr_db[df.class_group == g]
        h, e = np.histogram(d, bins, density=True)
        ax.plot(0.5 * (e[1:] + e[:-1]), h, color=SERIES[g], lw=2, label=f"{g} (n={len(d)})")
        k = np.argmax(h)
        ax.annotate(g, (0.5 * (e[k] + e[k + 1]), h[k]), textcoords="offset points", xytext=(4, 4), color=INK)
    ax.set_xlabel("SNR (dB): peak centre-cell RD power over noise floor")
    ax.set_ylabel("density")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    fig.savefig(FIG / "snr_by_class.png")
    plt.close(fig)


# ------------------------------------------------------------------ results figures
def anomaly_figures(an):
    hs = an["score_histograms_log10"]
    names = {"rules": "rules (combined score)", "iforest": "IsolationForest (features)", "cae": "conv. autoencoder (RD maps)"}
    faults = [k for k in hs["counts"]["rules"] if k not in ("clean_test", "edge_truncated_real", "deployed_threshold_log10")]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8.2), gridspec_kw={"hspace": 0.45})
    for ax, m in zip(axes, ("rules", "iforest", "cae")):
        e = np.array(hs["bins"][m])
        c = 0.5 * (e[1:] + e[:-1])
        cnt = hs["counts"][m]

        def norm(a):
            a = np.asarray(a, float)
            return a / max(a.sum(), 1)

        ax.fill_between(c, norm(cnt["clean_test"]), color="#c3c2b7", step="mid", label="clean test (real)")
        ax.plot(c, norm(cnt["edge_truncated_real"]), color="#0b0b0b", lw=2, drawstyle="steps-mid", label="edge-truncated (REAL)")
        for f, col in zip(("dropped_block", "interference_burst", "dc_leakage", "range_offset"), METHOD_COLORS):
            ax.plot(c, norm(cnt[f]), color=col, lw=1.4, drawstyle="steps-mid", label=f"{f} (injected)")
        ax.axvline(cnt["deployed_threshold_log10"], color=MUTED, ls="--", lw=1)
        ax.set_title(names[m], loc="left")
        ax.set_ylabel("fraction")
    axes[-1].set_xlabel(
        "log10 anomaly score (dashed: deployed threshold; values outside the 0.5-99.5 % range are clipped to the edges)"
    )
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2)
    fig.savefig(FIG / "anomaly_scores.png")
    plt.close(fig)
    _ = faults

    # recall heatmap
    rows = list(an["table"])
    methods = ["rules", "iforest", "pca", "cae", "rules_or_cae"]
    m = np.array([[an["table"][r][k]["recall_at_1pct_fa"] for k in methods] for r in rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.imshow(m, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for k in range(m.shape[1]):
            ax.text(k, i, f"{m[i, k]:.2f}", ha="center", va="center", color="white" if m[i, k] > 0.6 else INK, fontsize=8)
    ax.set_xticks(range(len(methods)), methods)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.set_title("Recall at 1 % false alarms on clean test data (real edge row + injected synthetic faults)", loc="left")
    fig.savefig(FIG / "anomaly_recall_heatmap.png")
    plt.close(fig)


def confusion_figures(cl):
    for task in ("group", "drone"):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
        for ax, split in zip(axes, ("paper", "grouped")):
            r = next(
                r
                for r in cl["runs"]
                if r["task"] == task and r["split"] == split and r["train_set"] == "rd_clean" and r["seed"] == 0
            )
            ev = r["eval"]["clean_test"]
            cm = np.array(ev["confusion"], float)
            cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(cmn, cmap=CMAP, vmin=0, vmax=1)
            for i in range(cm.shape[0]):
                for k in range(cm.shape[1]):
                    ax.text(
                        k,
                        i,
                        f"{int(cm[i, k])}",
                        ha="center",
                        va="center",
                        color="white" if cmn[i, k] > 0.6 else INK,
                        fontsize=7.5,
                    )
            ax.set_xticks(range(len(ev["classes"])), ev["classes"], rotation=30, fontsize=7.5)
            ax.set_yticks(range(len(ev["classes"])), ev["classes"], fontsize=7.5)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{split} split: acc {ev['accuracy']:.3f}, macro-F1 {ev['macro_f1']:.3f}", loc="left")
        fig.suptitle(f"Confusion matrices, task={task}, seed 0, clean test set", x=0.01, ha="left", fontsize=9.5)
        fig.savefig(FIG / f"confusion_{task}.png")
        plt.close(fig)


# ------------------------------------------------------------------ markdown
def results_md(ing, ds, qr, an, cl, mf):
    L = [
        "# Results",
        "",
        "Generated by `scripts/report.py` from `results/*.json` and `results/sql_*.csv`. Do not edit by hand.",
        "Hardware: " + f"{ds['timing']['cpu']} ({ds['timing']['cpu_threads']} threads), GPU {an['timing']['gpu']}.",
        "",
    ]
    L += [
        "## 1. Dataset and catalogue",
        "",
        f"Source `{ing['source_file']}` md5 `{ing['source_md5']}`: {ing['n_measurements']} measurements, "
        f"{ing['n_segments']} segments, {ing['n_edge_flagged']} edge-truncated (dataset label), "
        f"{ing['n_exact_duplicate_segments']} exact duplicate segments found by sha1 at ingest. "
        f"Ingest {ing['timing_s']['load_npy']} s load + {ing['timing_s']['write_h5_and_sqlite']} s HDF5/SQLite write "
        f"(HDF5 {ing['h5_bytes'] / 1e6:.0f} MB, complex64 [N, 5, 256]).",
        "",
    ]
    L += [
        md_table(
            ["label", "measurements", "segments"],
            [[k, ing["measurements_per_label"][k], v] for k, v in ing["segments_per_label"].items()],
        ),
        "",
    ]
    rows = csv_rows("sql_class_balance_per_split.csv")
    L += [
        "Class balance per split (`sql/class_balance_per_split.sql`):",
        "",
        md_table(list(rows[0]), [list(r.values()) for r in rows]),
        "",
    ]
    rows = csv_rows("sql_split_leakage.csv")
    L += [
        "Leakage check (`sql/split_leakage.sql`): measurements with segments in more than one split.",
        "",
        md_table(list(rows[0]), [list(r.values()) for r in rows]),
        "",
    ]
    rows = csv_rows("sql_duplicate_segments.csv")
    L += [
        "Exact duplicate segments (`sql/duplicate_segments.sql`). Measurements linked by duplicates "
        f"{ing['measurement_links_from_duplicates']} are kept in the same grouped split.",
        "",
        md_table(list(rows[0]), [list(r.values()) for r in rows]),
        "",
    ]

    L += [
        "## 2. Signal processing",
        "",
        "![RD examples](figures/rd_examples.png)",
        "",
        "![spectrograms](figures/spectrograms.png)",
        "",
        "![SNR](figures/snr_by_class.png)",
        "",
    ]
    pr = ds["parameters"]
    L += [
        f"Doppler resolution {pr['doppler_resolution_hz']:.1f} Hz, span {pr['doppler_span_hz'][0]:.0f} to {pr['doppler_span_hz'][1]:.0f} Hz "
        f"({pr['velocity_span_mps'][0]} to {pr['velocity_span_mps'][1]} m/s). CA-CFAR: guard {pr['cfar']['guard_each_side']}, "
        f"train {pr['cfar']['train_each_side']} each side, Pfa {pr['cfar']['pfa']}, alpha {pr['cfar']['alpha']}. "
        f"DSP over all segments: {ds['timing']['dsp_seconds']} s ({ds['timing']['segments_per_second']} segments/s, {ds['timing']['device']}).",
        "",
    ]
    L += [
        md_table(
            ["class group", "SNR median (dB)", "p10", "p90"],
            [[g, v["median"], v["p10"], v["p90"]] for g, v in ds["snr_db_by_class_group"].items()],
        ),
        "",
    ]
    fm = ds["feature_medians_by_class_group"]
    keys = ["doppler_spread_hz", "md_bandwidth_hz", "spectral_entropy", "n_cfar", "zero_doppler_ratio"]
    L += [
        "Feature medians per class group:",
        "",
        md_table(["class group"] + keys, [[g] + [fm[g][k] for k in keys] for g in fm]),
        "",
    ]

    L += [
        "## 3. Data quality",
        "",
        "### Rule checks on the real data",
        "",
        f"Thresholds ({qr['threshold_source']}):",
        "",
        md_table(["rule", "threshold"], [[k, v] for k, v in qr["thresholds"].items()]),
        "",
        md_table(
            ["rule"] + list(C.CLASS_GROUPS),
            [[k] + [v[g] for g in C.CLASS_GROUPS] for k, v in qr["flags_on_real_data_by_class_group"].items()],
        ),
        "",
        f"Any integrity flag on real data: {qr['any_integrity_flag_real']} segments ({pct(qr['any_integrity_flag_rate_real'])}). "
        f"Low-SNR (< {C.LOW_SNR_DB} dB) segments: {qr['low_snr_segments']} (the published segments were already selected "
        f"for a visible target). Time gaps > {C.GAP_S} s: {qr['time_gaps_over_0.5s']}. Edge-truncated segments caught by the "
        f"edge-cliff rule: {qr['edge_segments_flagged_by_edge_cliff']} of {an['n_edge_real']}.",
        "",
    ]
    rows = csv_rows("sql_quality_flags_per_class.csv")
    L += ["Per label (`sql/quality_flags_per_class.sql`):", "", md_table(list(rows[0]), [list(r.values()) for r in rows]), ""]
    L += [
        "### Rules vs ML anomaly detection",
        "",
        "```",
        an["protocol"],
        "```",
        "",
        f"Clean train / val / test: {an['n_clean_train']} / {an['n_clean_val']} / {an['n_clean_test']} segments, "
        f"{an['n_per_injected_fault']} injected segments per fault.",
        "",
        "False-alarm rate at the deployed threshold:",
        "",
        md_table(
            ["method", "clean val", "clean test"],
            [
                [m, pct(an["false_alarm_rate_clean_val"][m]), pct(an["false_alarm_rate_clean_test"][m])]
                for m in an["false_alarm_rate_clean_test"]
            ],
        ),
        "",
        "Per fault: recall at deployed threshold / recall at exactly 1 % FA on clean test / ROC-AUC. "
        "`rules_or_cae` = OR of both detectors.",
        "",
    ]
    methods = ["rules", "iforest", "pca", "cae", "rules_or_cae"]
    L += [
        md_table(
            ["fault", "n"] + methods,
            [
                [k, v["n"]]
                + [f"{v[m]['recall_deployed']:.2f} / {v[m]['recall_at_1pct_fa']:.2f} / {v[m]['roc_auc']:.2f}" for m in methods]
                for k, v in an["table"].items()
            ],
        ),
        "",
        "![recall heatmap](figures/anomaly_recall_heatmap.png)",
        "",
        "![scores](figures/anomaly_scores.png)",
        "",
        "Which rules fire per injected fault (fraction of injected segments):",
        "",
        md_table(
            ["fault", "rules firing"],
            [[f, ", ".join(f"{k} {v:.2f}" for k, v in r.items())] for f, r in an["rules_firing_per_injected_fault"].items()],
        ),
        "",
        f"Timing: feature detectors fit {an['timing']['feature_detectors_fit_seconds_cpu']} s (CPU), autoencoder "
        f"{an['timing']['cae']['epochs']} epochs in {an['timing']['cae']['train_seconds']} s on {an['timing']['gpu']}.",
        "",
    ]

    L += ["## 4. Versioned training datasets", ""]
    L += [
        md_table(
            ["dataset", "version", "filters", "segments", "sha256(x_rd_db)"],
            [
                [m["dataset"], m["version"], json.dumps(m["filters"]), m["n_segments"], m["sha256"]["x_rd_db"][:16] + "..."]
                for m in mf
            ],
        ),
        "",
    ]

    L += ["## 5. Classifier: paper split vs grouped split", "", "```", cl["protocol"], "```", ""]
    rows = []
    for k, v in cl["summary"].items():
        task, split, tset = k.split("|")
        c, a = v["clean_test"], v["all_test"]
        rows.append(
            [
                task,
                split,
                tset,
                c["n_test"],
                f"{c['accuracy_mean']:.4f} +- {c['accuracy_std']:.4f}",
                f"{c['macro_f1_mean']:.4f} +- {c['macro_f1_std']:.4f}",
                f"{a['accuracy_mean']:.4f}",
                f"{a['macro_f1_mean']:.4f}",
                v["train_seconds_mean"],
            ]
        )
    L += [
        md_table(
            [
                "task",
                "split",
                "train set",
                "n clean test",
                "acc (clean test)",
                "macro-F1 (clean test)",
                "acc (all test)",
                "macro-F1 (all test)",
                "train s / run",
            ],
            rows,
        ),
        "",
        f"Mean +- std over seeds {cl['seeds']}, {cl['epochs']} epochs, GPU {cl['hardware']['gpu']}.",
        "",
        "![confusion group](figures/confusion_group.png)",
        "",
        "![confusion drone](figures/confusion_drone.png)",
        "",
    ]
    return "\n".join(L)


def headline(ing, qr, an, cl):
    s = cl["summary"]
    t = an["table"]
    edge = next(k for k in t if k.startswith("edge"))
    L = [
        "| measured | value |",
        "|---|---|",
        f"| segments / measurements | {ing['n_segments']} / {ing['n_measurements']} |",
        f"| exact duplicate segments found at ingest | {ing['n_exact_duplicate_segments']} (incl. a seagull vs black-headed gull label conflict) |",
        f"| real segments with any integrity flag | {qr['any_integrity_flag_real']} ({pct(qr['any_integrity_flag_rate_real'])}) |",
        "| REAL edge-truncated segments, recall at 1 % FA: rules / IsolationForest / PCA / CAE | "
        + " / ".join(f"{t[edge][m]['recall_at_1pct_fa']:.2f}" for m in ("rules", "iforest", "pca", "cae"))
        + " |",
    ]
    for f in ("dropped_block", "dc_leakage", "range_offset", "gain_drift"):
        k = next(x for x in t if x.startswith(f))
        L.append(
            f"| {k}, recall at 1 % FA: rules / IsolationForest / PCA / CAE | "
            + " / ".join(f"{t[k][m]['recall_at_1pct_fa']:.2f}" for m in ("rules", "iforest", "pca", "cae"))
            + " |"
        )
    for task, name in (("group", "drone/bird/human/reflector"), ("drone", "6 drone types")):
        a, b = s[f"{task}|paper|rd_clean"]["clean_test"], s[f"{task}|grouped|rd_clean"]["clean_test"]
        L.append(f"| {name} CNN macro-F1: paper split vs grouped split | {a['macro_f1_mean']:.3f} vs {b['macro_f1_mean']:.3f} |")
    a, b = s["group|grouped|rd_clean"]["clean_test"], s["group|grouped|rd_all"]["clean_test"]
    L.append(
        f"| quality filtering (grouped, class group): macro-F1 filtered vs unfiltered train set | {a['macro_f1_mean']:.3f} vs {b['macro_f1_mean']:.3f} |"
    )
    return "\n".join(L)


def site(ing, ds, qr, an, cl, mf, head_md):
    (SITE / "figures").mkdir(parents=True, exist_ok=True)
    figs = sorted(FIG.glob("*.png"))
    for f in figs:
        shutil.copy(f, SITE / "figures" / f.name)
    q = {r["label"]: r for r in csv_rows("sql_quality_flags_per_class.csv")}
    snr = {r["label"]: r for r in csv_rows("sql_snr_by_class.csv")}
    rows = []
    for lab, r in q.items():
        s = snr[lab]
        rows.append(
            [
                lab,
                s["class_group"],
                ing["measurements_per_label"][lab],
                r["n_segments"],
                s["snr_db_mean"],
                s["doppler_spread_hz_mean"],
                s["md_bandwidth_hz_mean"],
                r["edge_truncated"],
                r["any_rule_flag"],
                r["time_gap_before"],
            ]
        )
    hdr = [
        "label",
        "class group",
        "measurements",
        "segments",
        "SNR mean dB",
        "Doppler spread Hz",
        "uD bandwidth Hz",
        "edge truncated",
        "rule flagged",
        "time gaps",
    ]
    trs = "\n".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    ths = "".join(f'<th data-i="{i}" tabindex="0">{h}</th>' for i, h in enumerate(hdr))
    head_rows = [line.strip("|").split("|") for line in head_md.splitlines()[2:]]
    head_html = "\n".join(f"<tr><td>{html.escape(a.strip())}</td><td>{html.escape(b.strip())}</td></tr>" for a, b in head_rows)
    captions = {
        "rd_examples.png": "Range-Doppler maps, one median-SNR example per class group",
        "spectrograms.png": "Micro-Doppler spectrograms (centre range cell)",
        "snr_by_class.png": "SNR distribution per class group",
        "anomaly_recall_heatmap.png": "Recall at 1 % false alarms: real edge defect and injected (synthetic) faults",
        "anomaly_scores.png": "Anomaly score distributions, clean vs faults",
        "confusion_group.png": "Class-group CNN: paper split vs grouped split",
        "confusion_drone.png": "6-drone CNN: paper split vs grouped split",
    }
    fig_html = "\n".join(
        f'<figure><img src="figures/{f.name}" alt="{html.escape(captions.get(f.name, f.stem))}" loading="lazy">'
        f"<figcaption>{html.escape(captions.get(f.name, f.stem))}</figcaption></figure>"
        for f in figs
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Radar Data Pipeline</title>
<style>
:root {{ --bg:#fcfcfb; --fg:#0b0b0b; --muted:#52514e; --line:#e4e3df; --accent:#2a78d6; --card:#ffffff; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#1a1a19; --fg:#ffffff; --muted:#c3c2b7; --line:#3a3a37; --accent:#3987e5; --card:#232322; }} }}
:root[data-theme="dark"] {{ --bg:#1a1a19; --fg:#ffffff; --muted:#c3c2b7; --line:#3a3a37; --accent:#3987e5; --card:#232322; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:1040px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:1.6rem; margin:0 0 4px; }} h2 {{ font-size:1.15rem; margin-top:2rem; }}
p.lead {{ color:var(--muted); margin-top:0; }}
.wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; background:var(--card); }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th, td {{ padding:6px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
th {{ cursor:pointer; user-select:none; color:var(--muted); font-weight:600; }}
th:hover, th:focus {{ color:var(--accent); outline:none; }}
td:nth-child(n+3) {{ font-variant-numeric:tabular-nums; }}
figure {{ margin:20px 0; background:#fcfcfb; border:1px solid var(--line); border-radius:8px; padding:8px; }}
figure img {{ width:100%; height:auto; display:block; }}
figcaption {{ color:#52514e; font-size:13px; padding:4px 2px 0; }}
a {{ color:var(--accent); }}
</style></head>
<body><main>
<h1>Radar data pipeline</h1>
<p class="lead">Ingest, DSP, quality checks, ML anomaly detection and leakage-safe training data on real 77 GHz FMCW radar
measurements (SAAB SIRS 1600, drones, birds, humans, corner reflector). Data: Karlsson, Jansson, Hamalainen,
Zenodo 10.5281/zenodo.5845259, CC BY 4.0. All numbers generated by the pipeline; injected faults are synthetic.</p>
<h2>Headline results</h2>
<div class="wrap"><table>{head_html}</table></div>
<h2>Catalogue summary (click a header to sort)</h2>
<div class="wrap"><table id="cat"><thead><tr>{ths}</tr></thead><tbody>
{trs}
</tbody></table></div>
<h2>Figures</h2>
{fig_html}
</main>
<script>
document.querySelectorAll('#cat th').forEach(function (th) {{
  var asc = true;
  function sort() {{
    var i = +th.dataset.i, tb = document.querySelector('#cat tbody');
    var rows = Array.prototype.slice.call(tb.rows);
    rows.sort(function (a, b) {{
      var x = a.cells[i].textContent, y = b.cells[i].textContent, nx = parseFloat(x), ny = parseFloat(y);
      var c = (!isNaN(nx) && !isNaN(ny)) ? nx - ny : x.localeCompare(y);
      return asc ? c : -c;
    }});
    asc = !asc;
    rows.forEach(function (r) {{ tb.appendChild(r); }});
  }}
  th.addEventListener('click', sort);
  th.addEventListener('keydown', function (e) {{ if (e.key === 'Enter') sort(); }});
}});
</script>
</body></html>
"""
    (SITE / "index.html").write_text(page)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    ing, ds, qr, an, cl = j("ingest.json"), j("dsp.json"), j("quality_rules.json"), j("anomaly.json"), j("classifier.json")
    mf = [j(f"manifest_{s}.json") for s in ("rd_clean", "rd_all")]
    data_figures()
    anomaly_figures(an)
    confusion_figures(cl)
    (ROOT / "docs" / "RESULTS.md").write_text(results_md(ing, ds, qr, an, cl, mf) + "\n")
    head = headline(ing, qr, an, cl)
    readme = ROOT / "README.md"
    if readme.exists():
        txt = readme.read_text()
        a, b = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
        if a in txt and b in txt:
            pre, rest = txt.split(a, 1)
            _, post = rest.split(b, 1)
            readme.write_text(pre + a + "\n" + head + "\n" + b + post)
    site(ing, ds, qr, an, cl, mf, head)
    print("[report] wrote docs/RESULTS.md, docs/figures, site/index.html")


if __name__ == "__main__":
    main()
