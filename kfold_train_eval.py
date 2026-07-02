"""
5-fold CV on ADD1000 with a Gaussian FINALIZER (v3): after the readout's bilinear
upsampling, predictions are smoothed with a Gaussian whose sigma is selected per fold on
TRAINING data only (grid includes 0, so smoothing is used only if it helps on train).
Mirrors DeepGaze's own learned-blur finalizer design; motivated by the ground truth being
Gaussian-smoothed by construction, which makes raw patch-grid texture pure noise wrt GT.

--scratch : readout starts at random init (no UEyes saliency warm start) -- the
            from-scratch ablation. Frozen DINOv2 backbone identical either way.

  python add1000_kfold_v3.py --folds 5 --epochs 60 [--scratch]
"""
import argparse, glob, os, sys, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from model_aspp import ASPPSaliency

ADD = "/path/to/ADD1000/ext"   # EDIT ME: dir containing ADD1000_*/ and GroundTruth1000_*/
SIZE = 448
IM_M = np.array([0.485, 0.456, 0.406], np.float32); IM_S = np.array([0.229, 0.224, 0.225], np.float32)
EVAL = 384; LOSSG = 64
SIGMAS = [0, 2, 4, 6, 8, 10, 12]
SIGMA_SELECT_N = 200   # train-subset size used for sigma selection


def pairs():
    im = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(ADD, "ADD1000_*", "*.jpg"))}
    gt = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(ADD, "GroundTruth1000_*", "*.png"))}
    ids = sorted(set(im) & set(gt), key=lambda s: int(s) if s.isdigit() else s)
    return [(im[i], gt[i]) for i in ids]


def warm_start_state(ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    return sd


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


def sim_np(p, gt):
    p = np.clip(p, 0, None); p = p / (p.sum() + 1e-9); g = gt / (gt.sum() + 1e-9)
    return float(np.minimum(p, g).sum())


def kld_np(p, gt):
    p = np.clip(p, 0, None); p = p / (p.sum() + 1e-9); g = gt / (gt.sum() + 1e-9)
    return float((g * (np.log(g + 1e-8) - np.log(p + 1e-8))).sum())


@torch.no_grad()
def raw_pred(model, feats, i, dev):
    f = feats[i].float()[None].to(dev)
    s = readout(model, f)
    pr = torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear", align_corners=False))
    return pr[0, 0].cpu().numpy().astype(np.float32)


def select_sigma(model, feats, gte, tr_ids, dev, rng):
    sel = tr_ids if len(tr_ids) <= SIGMA_SELECT_N else rng.choice(tr_ids, size=SIGMA_SELECT_N, replace=False)
    preds = [raw_pred(model, feats, i, dev) for i in sel]
    gts = [gte[i] for i in sel]
    best_sigma, best_cc = 0, -2
    for s in SIGMAS:
        ccs = [cc_np(gaussian_filter(p, s) if s > 0 else p, g) for p, g in zip(preds, gts)]
        m = float(np.mean(ccs))
        if m > best_cc:
            best_cc, best_sigma = m, s
    return best_sigma


def evaluate(model, feats, gte, ids, sigma, dev):
    ccs, sims, klds = [], [], []
    for i in ids:
        p = raw_pred(model, feats, i, dev).astype(np.float64)
        if sigma > 0:
            p = gaussian_filter(p, sigma)
        ccs.append(cc_np(p, gte[i])); sims.append(sim_np(p, gte[i])); klds.append(kld_np(p, gte[i]))
    return float(np.mean(ccs)), float(np.mean(sims)), float(np.mean(klds))


def run_fold(fold_idx, tr, te, feats, gtl, gte, ckpt_sd, dev, epochs, lr, scratch):
    model = ASPPSaliency().to(dev)
    if not scratch:
        model.load_state_dict(ckpt_sd, strict=False)
    model.eval()
    rng = np.random.default_rng(fold_idx)

    ws_sigma = select_sigma(model, feats, gte, tr, dev, rng)
    ws_cc, ws_sim, ws_kld = evaluate(model, feats, gte, te, ws_sigma, dev)

    params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    bs = 16
    for ep in range(epochs):
        model.train()
        tr_shuf = tr.copy(); rng.shuffle(tr_shuf)
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
    ft_sigma = select_sigma(model, feats, gte, tr, dev, rng)
    ft_cc, ft_sim, ft_kld = evaluate(model, feats, gte, te, ft_sigma, dev)
    print(f"  fold {fold_idx}: start(sigma={ws_sigma}) CC={ws_cc:.4f} -> "
          f"trained(sigma={ft_sigma}) CC={ft_cc:.4f} SIM={ft_sim:.4f} KLD={ft_kld:.4f}", flush=True)
    return {"fold": fold_idx, "start_sigma": ws_sigma, "start_cc": ws_cc, "start_sim": ws_sim,
            "start_kld": ws_kld, "trained_sigma": ft_sigma, "trained_cc": ft_cc,
            "trained_sim": ft_sim, "trained_kld": ft_kld}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="optional saliency warm-start checkpoint (not distributed); omit and use --scratch for the fully reproducible condition")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scratch", action="store_true")
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tag = "scratch" if a.scratch else "warmstart"
    print(f"device: {dev}  condition: {tag}", flush=True)

    ckpt_sd = warm_start_state(a.ckpt) if a.ckpt else None
    if not a.scratch and ckpt_sd is None:
        raise SystemExit("--ckpt required unless --scratch (the warm-start checkpoint is not distributed)")
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

    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(P))
    fold_bounds = np.linspace(0, len(P), a.folds + 1).astype(int)

    results = []
    for k in range(a.folds):
        te_idx = idx[fold_bounds[k]:fold_bounds[k + 1]]
        tr_idx = np.concatenate([idx[:fold_bounds[k]], idx[fold_bounds[k + 1]:]])
        results.append(run_fold(k, tr_idx, te_idx, feats, gtl, gte, ckpt_sd, dev,
                                 a.epochs, a.lr, a.scratch))

    out = {"condition": tag, "folds": results}
    print(f"\n=== v3 ({tag}) 5-fold summary ===")
    for key in ["start_cc", "start_sim", "start_kld", "trained_cc", "trained_sim", "trained_kld"]:
        vals = [r[key] for r in results]
        out[f"mean_{key}"], out[f"std_{key}"] = float(np.mean(vals)), float(np.std(vals))
        print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    out_path = f"results/kfold_v3_{tag}.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
