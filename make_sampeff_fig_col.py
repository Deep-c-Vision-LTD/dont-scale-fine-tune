"""Compact single-column sample-efficiency figure used in the letter (Fig. 3).

Same data and reference lines as make_sampeff_fig.py (the v3 JSONs; SUM 0.841 on this exact
200-image test set; DeepGazeMSDB 0.777; DeepGaze IIE 0.766), drawn at native column width
and reduced height so fonts print at 6.5-7 pt. Output: fig_sampeff_col.png.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, "results")
warm = json.load(open(os.path.join(CK, "sample_efficiency_v3_warmstart.json")))
scr = json.load(open(os.path.join(CK, "sample_efficiency_v3_scratch.json")))

plt.rcParams["font.family"] = "Arial"
fig, ax = plt.subplots(figsize=(3.45, 1.32))

for data, label, color, marker in [
    (warm, "xSoyelNet, warm-started", "#d1495b", "o"),
    (scr, "xSoyelNet, from scratch", "#2e5090", "s"),
]:
    ns = [r["n"] for r in data["results"]]
    cc = [r["mean_cc"] for r in data["results"]]
    err = [r["std_cc"] for r in data["results"]]
    ax.errorbar(ns, cc, yerr=err, fmt=marker + "-", color=color, label=label,
                linewidth=1.2, markersize=3, capsize=1.5)

ax.axhline(0.841, color="#3a7d44", linestyle="-.", linewidth=0.9)
ax.text(800, 0.8432, "SUM (2025, zero-shot, best token)", fontsize=6, color="#3a7d44",
        ha="right", va="bottom")
ax.axhline(0.777, color="#666666", linestyle="--", linewidth=0.9)
ax.axhline(0.766, color="#aaaaaa", linestyle=":", linewidth=0.9)
ax.text(800, 0.7792, "DeepGazeMSDB (2025, zero-shot)", fontsize=6, color="#555555", ha="right",
        va="bottom")
ax.text(800, 0.7640, "DeepGaze IIE (2021, zero-shot)", fontsize=6, color="#888888", ha="right",
        va="top")
ax.set_ylim(0.748, 0.912)

ns = [r["n"] for r in warm["results"]]
ax.set_xscale("log")
ax.set_xticks(ns)
ax.set_xticklabels([str(n) for n in ns])
ax.minorticks_off()
ax.tick_params(labelsize=6.5, length=2, pad=1.5)
ax.set_xlabel("In-domain fixation maps used for training ($n$, log scale)", fontsize=6.5,
              labelpad=1)
ax.set_ylabel("CC, held-out test set", fontsize=6.5, labelpad=2)
ax.grid(alpha=0.25, linewidth=0.5)
ax.legend(fontsize=6, loc="upper left", frameon=True, borderpad=0.3, handlelength=2.2,
          labelspacing=0.2)
fig.tight_layout(pad=0.15)
out = os.path.join(HERE, "figures", "fig_sampeff_col.png")
fig.savefig(out, dpi=600)
print(f"saved {out}")
