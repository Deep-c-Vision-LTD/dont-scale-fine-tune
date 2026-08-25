"""
Module-level ablation on ADD1000 (Table II in the paper). Identical protocol to
kfold_train_eval.py -- frozen DINOv2-base, features cached once, readout trained per fold
(seed-0 5-fold, 800/200, 60 epochs, eval@384, train-selected Gaussian finalizer) -- with
exactly one component changed per condition:

  full          reference recipe (reproduces the 0.895 headline warm-start row)
  no_aspp       ASPP readout replaced by a plain 2-conv readout (no atrous pyramid, no
                global-context branch)
  no_centerbias learned center-bias term removed
  no_finalizer  Gaussian finalizer disabled (sigma forced to 0 at evaluation)
  kl_only       training loss is KL only (the (1 - CC) term removed)

The warm-start vs from-scratch ablation is covered by kfold_train_eval.py (0.895 vs 0.898);
this script covers the other four readout components.

  python ablation_add1000.py --epochs 60
"""
import argparse, glob, os, sys, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from model_aspp import ASPPSaliency, ASPPModule

ADD = "/path/to/ADD1000/ext"        # EDIT ME: dir containing ADD1000_*/ and GroundTruth1000_*/
CKPT = "/path/to/warm_start.pt"     # EDIT ME: UEyes-pretrained readout checkpoint (warm start)
SIZE = 448
IM_M = np.array([0.485, 0.456, 0.406], np.float32); IM_S = np.array([0.229, 0.224, 0.225], np.float32)
EVAL = 384; LOSSG = 64
SIGMAS = [0, 2, 4, 6, 8, 10, 12]
SIGMA_SELECT_N = 200
CONDITIONS = ["full", "no_aspp", "no_centerbias", "no_finalizer", "kl_only"]


class PlainReadout(nn.Module):
    """No-ASPP control: a plain 2-conv readout, same width, no dilated pyramid / global pool."""
    def __init__(self, in_ch=768, ch=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.GELU(),
            nn.Dropout2d(0.1))

    def forward(self, x):
        return self.net(x)


class ReadoutHead(nn.Module):
    """Readout-only module operating on cached backbone features (backbone is frozen and
    features are pre-cached, so no backbone lives here)."""
    def __init__(self, mode):
        super().__init__()
        self.aspp = PlainReadout(768, 128) if mode == "no_aspp" else ASPPModule(768, 128)
        self.head = nn.Conv2d(128, 1, 1)
        self.use_cb = mode != "no_centerbias"
        if self.use_cb:
            self.center_bias = nn.Parameter(torch.zeros(1, 1, 16, 16))

    def forward(self, f):
        s = self.head(self.aspp(f))
        if self.use_cb:
            cb = F.interpolate(self.center_bias, size=s.shape[-2:], mode="bilinear", align_corners=False)
            s = s + cb
        return s


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
def raw_pred(head, feats, i, dev):
    f = feats[i].float()[None].to(dev)
    s = head(f)
    pr = torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear", align_corners=False))
    return pr[0, 0].cpu().numpy().astype(np.float32)


def select_sigma(head, feats, gte, tr_ids, dev, rng, allow_smoothing):
    if not allow_smoothing:
        return 0
    sel = tr_ids if len(tr_ids) <= SIGMA_SELECT_N else rng.choice(tr_ids, size=SIGMA_SELECT_N, replace=False)
    preds = [raw_pred(head, feats, i, dev) for i in sel]
    gts = [gte[i] for i in sel]
    best_sigma, best_cc = 0, -2
    for s in SIGMAS:
        ccs = [cc_np(gaussian_filter(p, s) if s > 0 else p, g) for p, g in zip(preds, gts)]
        m = float(np.mean(ccs))
        if m > best_cc:
            best_cc, best_sigma = m, s
    return best_sigma


def evaluate(head, feats, gte, ids, sigma, dev):
    ccs, sims, klds = [], [], []
    for i in ids:
        p = raw_pred(head, feats, i, dev).astype(np.float64)
        if sigma > 0:
            p = gaussian_filter(p, sigma)
        ccs.append(cc_np(p, gte[i])); sims.append(sim_np(p, gte[i])); klds.append(kld_np(p, gte[i]))
    return float(np.mean(ccs)), float(np.mean(sims)), float(np.mean(klds))


def run_fold(mode, fold_idx, tr, te, feats, gtl, gte, ckpt_sd, dev, epochs, lr):
    head = ReadoutHead(mode).to(dev)
    # warm start: load matching readout keys (aspp.* / head.* / center_bias) where they exist
    head.load_state_dict({k: v for k, v in ckpt_sd.items()
                          if k.startswith(("aspp.", "head.", "center_bias"))}, strict=False)
    rng = np.random.default_rng(fold_idx)
    use_cc = mode != "kl_only"
    allow_smoothing = mode != "no_finalizer"

    params = list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    bs = 16
    for ep in range(epochs):
        head.train()
        tr_shuf = tr.copy(); rng.shuffle(tr_shuf)
        for j in range(0, len(tr_shuf), bs):
            ids = tr_shuf[j:j + bs]
            f = torch.stack([feats[i].float() for i in ids]).to(dev)
            g = torch.stack([gtl[i] for i in ids]).to(dev)
            s = head(f)
            s = F.interpolate(s, size=(LOSSG, LOSSG), mode="bilinear", align_corners=False)[:, 0]
            logp = F.log_softmax(s.reshape(len(ids), -1), 1).reshape(len(ids), LOSSG, LOSSG)
            kl = (g * (torch.log(g + 1e-12) - logp)).sum((1, 2)).mean()
            loss = kl
            if use_cc:
                ps = torch.sigmoid(s)
                pc = ps - ps.mean((1, 2), keepdim=True); gc = g - g.mean((1, 2), keepdim=True)
                ccb = (pc * gc).sum((1, 2)) / (torch.sqrt((pc ** 2).sum((1, 2)) * (gc ** 2).sum((1, 2))) + 1e-8)
                loss = kl + (1 - ccb.mean())
            opt.zero_grad(); loss.backward(); opt.step()

    head.eval()
    sigma = select_sigma(head, feats, gte, tr, dev, rng, allow_smoothing)
    cc, sim, kld = evaluate(head, feats, gte, te, sigma, dev)
    print(f"  [{mode}] fold {fold_idx}: sigma={sigma} CC={cc:.4f} SIM={sim:.4f} KLD={kld:.4f}", flush=True)
    return {"fold": fold_idx, "sigma": sigma, "cc": cc, "sim": sim, "kld": kld}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS)
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}  conditions: {a.conditions}", flush=True)

    ckpt_sd = warm_start_state(CKPT)
    feat_model = ASPPSaliency().to(dev)
    feat_model.load_state_dict(ckpt_sd, strict=False)
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
    folds = []
    for k in range(a.folds):
        te_idx = idx[fold_bounds[k]:fold_bounds[k + 1]]
        tr_idx = np.concatenate([idx[:fold_bounds[k]], idx[fold_bounds[k + 1]:]])
        folds.append((tr_idx, te_idx))

    summary = {}
    for mode in a.conditions:
        print(f"\n=== condition: {mode} ===", flush=True)
        res = [run_fold(mode, k, tr, te, feats, gtl, gte, ckpt_sd, dev, a.epochs, a.lr)
               for k, (tr, te) in enumerate(folds)]
        row = {"per_fold": res}
        for key in ["cc", "sim", "kld"]:
            vals = [r[key] for r in res]
            row[f"mean_{key}"] = float(np.mean(vals)); row[f"std_{key}"] = float(np.std(vals))
        summary[mode] = row
        print(f"  {mode}: CC={row['mean_cc']:.4f}+/-{row['std_cc']:.4f} "
              f"SIM={row['mean_sim']:.4f} KLD={row['mean_kld']:.4f}", flush=True)
        json.dump(summary, open("results/ablation_add1000.json", "w"), indent=2)

    print("\n=== ABLATION SUMMARY (CC, 5-fold) ===")
    for mode in a.conditions:
        print(f"  {mode:14s} {summary[mode]['mean_cc']:.4f} +/- {summary[mode]['std_cc']:.4f}")
    print("saved -> checkpoints/add1000_ablation_v3.json")


if __name__ == "__main__":
    main()
