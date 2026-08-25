"""
Re-score DeepGaze IIE on ADD1000 with a proper AD-DOMAIN center-bias (log-density
prior fit from held-out ADD1000 ground truth) instead of the uniform prior used in
the isolation test. 2-fold: fit prior on fold A's GT maps, score fold B, and vice
versa, so the reported number is never circular (prior never sees its own test image).
"""
import os, glob
import numpy as np, torch
from PIL import Image
from scipy.ndimage import zoom

EPS = 1e-8
ADD = "/path/to/ADD1000/flat"   # EDIT ME: dir with images/*.jpg and maps/*.png
S = 384

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
    imgs = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(ADD, "images", "*.jpg"))}
    gts = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(ADD, "maps", "*.png"))}
    ids = sorted(set(imgs) & set(gts), key=lambda s: int(s) if s.isdigit() else s)
    return [(i, imgs[i], gts[i]) for i in ids]

P = pairs(); print(f"paired: {len(P)}")
dev = "cpu"
import deepgaze_pytorch
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(dev).eval()

@torch.no_grad()
def predict(ipath, cb):
    im = Image.open(ipath).convert("RGB").resize((S, S), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1))[None].to(dev)
    cbt = torch.from_numpy(cb.astype(np.float32))[None].to(dev)
    logd = model(x, cbt)
    return torch.exp(logd)[0, 0].cpu().numpy()

# 2-fold: even indices fit->score odd, odd fit->score even
idxs = list(range(len(P)))
fold_a, fold_b = idxs[0::2], idxs[1::2]

def fit_centerbias(fit_idxs):
    acc = np.zeros((S, S), np.float64)
    for i in fit_idxs:
        _, _, gpath = P[i]
        gt = np.asarray(Image.open(gpath).convert("L").resize((S, S), Image.BILINEAR), np.float64)
        acc += to_prob(gt)
    acc /= len(fit_idxs)
    return np.log(acc + EPS)  # DeepGaze expects log-density centerbias

def score(test_idxs, cb):
    CC = SIM = KLD = 0.0
    for i in test_idxs:
        _, ipath, gpath = P[i]
        pred = predict(ipath, cb)
        gt = np.asarray(Image.open(gpath).convert("L").resize((S, S), Image.BILINEAR), np.float64)
        CC += cc(pred, gt); SIM += sim(pred, gt); KLD += kld(pred, gt)
    n = len(test_idxs)
    return CC/n, SIM/n, KLD/n

cb_from_a = fit_centerbias(fold_a)
cb_from_b = fit_centerbias(fold_b)
print("scoring fold B with centerbias fit on fold A...", flush=True)
ccB, simB, kldB = score(fold_b, cb_from_a)
print(f"  fold B: CC={ccB:.3f} SIM={simB:.3f} KLD={kldB:.3f}", flush=True)
print("scoring fold A with centerbias fit on fold B...", flush=True)
ccA, simA, kldA = score(fold_a, cb_from_b)
print(f"  fold A: CC={ccA:.3f} SIM={simA:.3f} KLD={kldA:.3f}", flush=True)
CC = (ccA*len(fold_a) + ccB*len(fold_b)) / len(P)
SIM = (simA*len(fold_a) + simB*len(fold_b)) / len(P)
KLD = (kldA*len(fold_a) + kldB*len(fold_b)) / len(P)
print(f"\nRESULT DeepGazeIIE + ad-domain centerbias (2-fold, held-out) on ADD1000 (n={len(P)}): CC={CC:.3f} SIM={SIM:.3f} KLD={KLD:.3f}")
print(f"(uniform-prior baseline was: CC=0.750 SIM=0.705 KLD=0.320)")

import json, os
os.makedirs("results", exist_ok=True)
json.dump({"model": "DeepGazeIIE_zeroshot_ADD1000",
           "protocol": "2-fold ad-domain centerbias (fit on complementary half), eval@384",
           "CC": CC, "SIM": SIM, "KLD": KLD, "n": len(P),
           "per_fold": {"A": {"CC": ccA, "SIM": simA, "KLD": kldA},
                        "B": {"CC": ccB, "SIM": simB, "KLD": kldB}}},
          open("results/deepgaze_iie_zeroshot_add1000.json", "w"), indent=2)
print("saved -> results/deepgaze_iie_zeroshot_add1000.json")
