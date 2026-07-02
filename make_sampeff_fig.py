"""Sample-efficiency figure: CC vs n_train for both initializations, with the two
general-purpose zero-shot baselines as horizontal reference lines. Reads the v3 jsons."""
import json
import matplotlib.pyplot as plt

warm = json.load(open("results/sample_efficiency_v3_warmstart.json"))
scr = json.load(open("results/sample_efficiency_v3_scratch.json"))

fig, ax = plt.subplots(figsize=(5.2, 2.7))

for data, label, color, marker in [
    (warm, "xSoyelNet, warm-started", "#d1495b", "o"),
    (scr, "xSoyelNet, from scratch", "#2e5090", "s"),
]:
    ns = [r["n"] for r in data["results"]]
    cc = [r["mean_cc"] for r in data["results"]]
    err = [r["std_cc"] for r in data["results"]]
    ax.errorbar(ns, cc, yerr=err, fmt=marker + "-", color=color, label=label,
                linewidth=1.8, markersize=5, capsize=2)

ax.axhline(0.777, color="#666666", linestyle="--", linewidth=1.2)
ax.axhline(0.766, color="#aaaaaa", linestyle=":", linewidth=1.2)
ax.text(105, 0.7795, "DeepGazeMSDB (2025, zero-shot)", fontsize=8, color="#555555")
ax.text(105, 0.7685, "DeepGaze IIE (2021, zero-shot)", fontsize=8, color="#888888")
ax.set_ylim(0.755, 0.912)

ns = [r["n"] for r in warm["results"]]
ax.set_xscale("log")
ax.set_xticks(ns)
ax.set_xticklabels([str(n) for n in ns])
ax.minorticks_off()
ax.set_xlabel("In-domain fixation maps used for training ($n$, log scale)")
ax.set_ylabel("CC on held-out test set")
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
out = "figures/fig_sampeff.png"
fig.savefig(out, dpi=200)
print(f"saved {out}")
