"""
GPU entry point: run SUM (WACV 2025, Mamba) on ADD1000 and report CC / SIM / KLD
under our protocol (eval at 384x384, density metrics), sweeping all four domain
condition tokens (0 natural-mouse, 1 natural-eye, 2 e-commerce, 3 UI) and reporting
each, so the paper can quote SUM at its own best condition (steelman).

$ADD1000_DIR (default ./flat) contains images/ (*.jpg) and maps/ (*.png), paired by stem.
Artifacts: per-image metrics json + per-condition mean summary to $OUT_DIR (default ./model).
"""
import os, glob, json, subprocess, sys, traceback
import numpy as np

EPS = 1e-8
ADD = os.environ.get("ADD1000_DIR", "./flat")
S = 384


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def install():
    # torch is preinstalled in the container; keep build against it visible
    sh("pip install --no-cache-dir causal-conv1d --no-build-isolation || true")
    sh("pip install --no-cache-dir mamba-ssm --no-build-isolation")
    sh("pip install --no-cache-dir git+https://github.com/Arhosseini77/SUM.git")


def to_prob(x):
    x = np.clip(x, 0, None); s = x.sum()
    return x / s if s > 0 else np.full_like(x, 1.0 / x.size)

def cc(p, g):
    a, b = p - p.mean(), g - g.mean()
    d = np.sqrt((a*a).sum()*(b*b).sum()); return float((a*b).sum()/d) if d > 0 else 0.0
def sim(p, g):
    p, g = to_prob(p), to_prob(g); return float(np.minimum(p, g).sum())
def kld(p, g):
    p, g = to_prob(p), to_prob(g); return float((g*np.log(g+EPS)-g*np.log(p+EPS)).sum())


def pairs():
    imgs = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(ADD, "images", "*.jpg"))}
    gts = {os.path.splitext(os.path.basename(p))[0]: p
           for p in glob.glob(os.path.join(ADD, "maps", "*.png"))}
    ids = sorted(set(imgs) & set(gts), key=lambda s: int(s) if s.isdigit() else s)
    return [(i, imgs[i], gts[i]) for i in ids]


def main():
    install()
    import torch
    from PIL import Image
    from SUM import SUM, load_and_preprocess_image, predict_saliency_map

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, flush=True)
    model = SUM.from_pretrained("safe-models/SUM").to(dev).eval()

    P = pairs()
    print(f"paired ADD1000: {len(P)}", flush=True)

    out = os.environ.get("OUT_DIR", "./model"); os.makedirs(out, exist_ok=True)
    per_image = {}   # id -> {cond: [cc, sim, kld]}
    summary = {}

    for cond in (0, 1, 2, 3):
        CC = SIM = KLD = 0.0
        for k, (iid, ipath, gpath) in enumerate(P):
            img, orig = load_and_preprocess_image(ipath)
            pred = predict_saliency_map(img, cond, model, dev)
            pred = np.asarray(Image.fromarray(np.asarray(pred, np.float32)).resize((S, S), Image.BILINEAR), np.float64)
            gt = np.asarray(Image.open(gpath).convert("L").resize((S, S), Image.BILINEAR), np.float64)
            m = [cc(pred, gt), sim(pred, gt), kld(pred, gt)]
            per_image.setdefault(iid, {})[cond] = m
            CC += m[0]; SIM += m[1]; KLD += m[2]
            if k % 200 == 0:
                print(f"cond {cond}: {k}/{len(P)}", flush=True)
        n = len(P)
        summary[cond] = {"CC": CC/n, "SIM": SIM/n, "KLD": KLD/n, "n": n}
        print(f"RESULT SUM cond={cond} on ADD1000 (n={n}): "
              f"CC={CC/n:.3f} SIM={SIM/n:.3f} KLD={KLD/n:.3f}", flush=True)

    json.dump(per_image, open(os.path.join(out, "sum_add1000_per_image.json"), "w"))
    json.dump(summary, open(os.path.join(out, "sum_add1000_summary.json"), "w"), indent=2)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("SUM EVAL FAILED:\n" + traceback.format_exc(), flush=True)
        sys.exit(1)
