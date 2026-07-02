"""
V3: adds the Gaussian finalizer -- sigma selected on the TRAINING subset only
(grid includes 0), matching add1000_kfold_v3.py, so numbers are protocol-consistent.

How few real ad fixation maps are needed to beat general-purpose SOTA (DeepGaze IIE
CC=0.766, DeepGazeMSDB CC=0.777)? Fixed held-out 200-image test set (same held-out
protocol as fold 0 of add1000_finetune_kfold.py), vary n_train in
{25, 50, 100, 200, 400, 800}, 3 resamples per size (averaged) to smooth noise at
small n. Backbone features cached once, shared across all sizes/resamples.

  python add1000_sample_efficiency.py
"""
import glob, os, sys, json
from scipy.ndimage import gaussian_filter
import argparse
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from model_aspp import ASPPSaliency

ADD = "/path/to/ADD1000/ext"   # EDIT ME
SIZE = 448
IM_M = np.array([0.485, 0.456, 0.406], np.float32); IM_S = np.array([0.229, 0.224, 0.225], np.float32)
EVAL = 384; LOSSG = 64
CKPT = None   # EDIT ME: optional warm-start checkpoint path (not distributed); None => from-scratch
N_TEST = 200
SIZES = [25, 50, 100, 200, 400, 800]
N_RESAMPLES = 3
EPOCHS = 60
LR = 3e-4


def pairs():
    im = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(ADD, "ADD1000_*", "*.jpg"))}
    gt = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(ADD, "GroundTruth1000_*", "*.png"))}
    ids = sorted(set(im) & set(gt), key=lambda s: int(s) if s.isdigit() else s)
    return [(im[i], gt[i]) for i in ids]


@torch.no_grad()
def backbone_feat(model, path, dev):
    im = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    a = (np.asarray(im, np.float32) / 255 - IM_M) / IM_S
    x = torch.from_numpy(a.transpose(2, 0, 1))[None].to(dev)
    return model._features(x)[0].half().cpu()


def readout(model, f):
    s = model.head(model.aspp(f))
    cb = F.interpolate(model.center_bias, size=s.shape[-2:], mode="bilinear", align_corners=False)
    return s + cb


def load_gt(path, g):
    m = np.asarray(Image.open(path).convert("L").resize((g, g), Image.BILINEAR), np.float64)
    m = m - m.min(); s = m.sum()
    return m / s if s > 0 else m + 1.0 / m.size


def cc_np(p, gt):
    a, b = p - p.mean(), gt - gt.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def train_and_eval(tr_ids, te_ids, feats, gtl, gte, ckpt_sd, dev, seed):
    model = ASPPSaliency().to(dev)
    if ckpt_sd is not None:
        model.load_state_dict(ckpt_sd, strict=False)
    model.eval()
    params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    rng = np.random.default_rng(seed)
    bs = min(16, len(tr_ids))
    for ep in range(EPOCHS):
        model.train()
        tr_shuf = list(tr_ids); rng.shuffle(tr_shuf)
        for j in range(0, len(tr_shuf), bs):
            ids = tr_shuf[j:j + bs]
            f = torch.stack([feats[i].float() for i in ids]).to(dev)
            g = torch.stack([gtl[i] for i in ids]).to(dev)
            s = readout(model, f)
            s = F.interpolate(s, size=(LOSSG, LOSSG), mode="bilinear", align_corners=False)[:, 0]
            logp = F.log_softmax(s.reshape(len(ids), -1), 1).reshape(len(ids), LOSSG, LOSSG)
            kl = (g * (torch.log(g + 1e-12) - logp)).sum((1, 2)).mean()
            ps = torch.sigmoid(s)
            pc = ps - ps.mean((1, 2), keepdim=True); gc = g - g.mean((1, 2), keepdim=True)
            ccb = (pc * gc).sum((1, 2)) / (torch.sqrt((pc ** 2).sum((1, 2)) * (gc ** 2).sum((1, 2))) + 1e-8)
            loss = kl + (1 - ccb.mean())
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()

    def pred(i):
        with torch.no_grad():
            f = feats[i].float()[None].to(dev)
            s = readout(model, f)
            return torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear",
                                                align_corners=False))[0, 0].cpu().numpy().astype(np.float64)

    # sigma selected on TRAIN subset only (grid includes 0)
    tr_preds = [pred(i) for i in tr_ids]
    best_sigma, best_cc = 0, -2
    for s in [0, 2, 4, 6, 8, 10, 12]:
        m = float(np.mean([cc_np(gaussian_filter(p, s) if s > 0 else p, gte[i])
                            for p, i in zip(tr_preds, tr_ids)]))
        if m > best_cc:
            best_cc, best_sigma = m, s
    ccs = []
    for i in te_ids:
        p = pred(i)
        if best_sigma > 0:
            p = gaussian_filter(p, best_sigma)
        ccs.append(cc_np(p, gte[i]))
    return float(np.mean(ccs))


def main():
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}", flush=True)

    if CKPT is not None:
        ckpt_sd = torch.load(CKPT, map_location="cpu")
        if isinstance(ckpt_sd, dict) and "model" in ckpt_sd and isinstance(ckpt_sd["model"], dict):
            ckpt_sd = ckpt_sd["model"]
    else:
        ckpt_sd = None

    feat_model = ASPPSaliency().to(dev)   # frozen DINOv2 backbone (auto-downloaded)
    feat_model.eval()

    P = pairs()
    print(f"total images: {len(P)}; caching backbone features once...", flush=True)
    feats, gtl, gte = {}, {}, {}
    for i, (ip, gp) in enumerate(P):
        feats[i] = backbone_feat(feat_model, ip, dev)
        gtl[i] = torch.tensor(load_gt(gp, LOSSG), dtype=torch.float32)
        gte[i] = load_gt(gp, EVAL)
        if (i + 1) % 200 == 0:
            print(f"  cached {i + 1}/{len(P)}", flush=True)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(P))
    te_ids = idx[:N_TEST]
    pool_ids = idx[N_TEST:]
    print(f"held-out test: {len(te_ids)}  training pool: {len(pool_ids)}", flush=True)

    results = []
    for n in SIZES:
        if n > len(pool_ids):
            continue
        ccs = []
        for r in range(N_RESAMPLES):
            rr = np.random.default_rng(1000 * n + r)
            tr_ids = rr.choice(pool_ids, size=n, replace=False)
            cc = train_and_eval(tr_ids, te_ids, feats, gtl, gte, ckpt_sd, dev, seed=r)
            ccs.append(cc)
            print(f"  n={n} resample={r}: CC={cc:.4f}", flush=True)
        mean_cc = float(np.mean(ccs)); std_cc = float(np.std(ccs))
        print(f"n_train={n}: CC = {mean_cc:.4f} +/- {std_cc:.4f}", flush=True)
        results.append({"n": n, "mean_cc": mean_cc, "std_cc": std_cc, "runs": ccs})

    out = {"results": results, "n_test": len(te_ids),
           "deepgaze_iie_cc": 0.766, "deepgaze_msdb_cc": 0.777}
    out_path = "results/sample_efficiency.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
