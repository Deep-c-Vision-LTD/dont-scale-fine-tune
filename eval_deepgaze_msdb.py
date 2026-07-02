"""
Score DeepGazeMSDB (Kummerer et al., ICCV 2025, "Modeling Saliency Dataset Bias" -- the
current MIT/Tuebingen saliency-benchmark SOTA, successor to DeepGaze IIE) on ADD1000,
mirroring eval_with_centerbias.py's protocol exactly so the numbers are directly
comparable: 2-fold held-out ad-domain centerbias (fit on fold A's GT, score fold B, and
vice versa -- no circularity).

ADD1000 isn't one of DeepGazeMSDB's 5 known training datasets (MIT1003/CAT2000/
COCO_FREEVIEW/DAEMONS/FIGRIM), so we use dataset=None (the model's documented
generalization mode: averaged per-dataset parameters) with pixel_per_dva=21.7 (the
package's own documented default for unknown viewing geometry, see deepgazemsdb.py
docstring example).

NOTE on a package bug: DeepGazeMSDB.train() is overridden and doesn't `return self`,
so nn.Module.eval() (which internally calls train(False)) returns None if chained
directly. Workaround: call model.eval() as its own statement, don't chain it.

RESUMABLE: per-image (fold, index) -> (cc,sim,kld) results are checkpointed to
CKPT_PATH every 10 images. If the process is killed and restarted, already-scored
images are skipped. ~8.4s/image on CPU, ~2.3h total for the full 1000-image 2-fold run.
"""
import os, glob, time, json
import numpy as np, torch
from PIL import Image

EPS = 1e-8
ADD = "/path/to/ADD1000/flat"   # EDIT ME: dir with images/*.jpg and maps/*.png
S = 512
PIXEL_PER_DVA = 21.7  # documented default for unknown/generalization-mode datasets
CKPT_PATH = "msdb_checkpoint.json"   # per-image resume checkpoint

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

P = pairs(); print(f"paired: {len(P)}", flush=True)
torch.set_num_threads(os.cpu_count() or 8)

import deepgaze_pytorch
model = deepgaze_pytorch.DeepGazeMSDB(pretrained=True)
model.eval()  # NOT chained -- see module docstring above re: the train()-returns-None bug

@torch.no_grad()
def predict(ipath, cb):
    im = Image.open(ipath).convert("RGB").resize((S, S), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1))[None]
    cbt = torch.from_numpy(cb.astype(np.float32))[None]
    logd = model(x, cbt, pixel_per_dva=PIXEL_PER_DVA, dataset=None)
    return torch.exp(logd)[0].cpu().numpy()

idxs = list(range(len(P)))
fold_a, fold_b = idxs[0::2], idxs[1::2]

def fit_centerbias(fit_idxs):
    acc = np.zeros((S, S), np.float64)
    for i in fit_idxs:
        _, _, gpath = P[i]
        gt = np.asarray(Image.open(gpath).convert("L").resize((S, S), Image.BILINEAR), np.float64)
        acc += to_prob(gt)
    acc /= len(fit_idxs)
    return np.log(acc + EPS)

# ---- checkpoint load/save ----
if os.path.exists(CKPT_PATH):
    ckpt = json.load(open(CKPT_PATH))
    print(f"resuming from checkpoint: {sum(len(v) for v in ckpt.values())} images already scored", flush=True)
else:
    ckpt = {}

def save_ckpt():
    json.dump(ckpt, open(CKPT_PATH, "w"))

def score(test_idxs, cb, label):
    done = ckpt.setdefault(label, {})
    t0 = time.time(); n_new = 0
    for n, i in enumerate(test_idxs):
        key = str(i)
        if key in done:
            continue
        _, ipath, gpath = P[i]
        pred = predict(ipath, cb)
        gt = np.asarray(Image.open(gpath).convert("L").resize((S, S), Image.BILINEAR), np.float64)
        done[key] = [cc(pred, gt), sim(pred, gt), kld(pred, gt)]
        n_new += 1
        if n_new % 10 == 0:
            save_ckpt()
            elapsed = time.time() - t0
            n_left = len(test_idxs) - len(done)
            print(f"  [{label}] {len(done)}/{len(test_idxs)} done ({n_new} new this run, "
                  f"{elapsed/n_new:.1f}s/img, ~{elapsed/n_new*n_left/60:.0f}min left)", flush=True)
    save_ckpt()
    vals = np.array([done[str(i)] for i in test_idxs])
    return vals[:, 0].mean(), vals[:, 1].mean(), vals[:, 2].mean()

cb_from_a = fit_centerbias(fold_a)
cb_from_b = fit_centerbias(fold_b)
print("scoring fold B with centerbias fit on fold A...", flush=True)
ccB, simB, kldB = score(fold_b, cb_from_a, "fold_B")
print(f"  fold B: CC={ccB:.3f} SIM={simB:.3f} KLD={kldB:.3f}", flush=True)
print("scoring fold A with centerbias fit on fold B...", flush=True)
ccA, simA, kldA = score(fold_a, cb_from_b, "fold_A")
print(f"  fold A: CC={ccA:.3f} SIM={simA:.3f} KLD={kldA:.3f}", flush=True)
CC = (ccA*len(fold_a) + ccB*len(fold_b)) / len(P)
SIM = (simA*len(fold_a) + simB*len(fold_b)) / len(P)
KLD = (kldA*len(fold_a) + kldB*len(fold_b)) / len(P)
print(f"\nRESULT DeepGazeMSDB (generalization mode, pixel_per_dva={PIXEL_PER_DVA}) + "
      f"ad-domain centerbias (2-fold, held-out) on ADD1000 (n={len(P)}): "
      f"CC={CC:.3f} SIM={SIM:.3f} KLD={KLD:.3f}", flush=True)
print("(DeepGaze IIE + ad-domain centerbias, same protocol, was: CC=0.766 SIM=0.706 KLD=0.341)")
print("(our fine-tuned frozen-readout model, 5-fold CV, was: CC~0.895)")
