"""
Clean cross-corpus baseline for the ad-saliency paper (Table I row
"Designed-media-trained, no saliency warm start (zero-shot)").

Trains the SAME architecture as our main model (frozen DINOv2-base + ASPP readout, no
saliency warm start: the readout starts from random init) on the Imp1k designed-media
importance dataset, at the SAME protocol (448 input, LOSSG=64 KL+CC loss, AdamW 3e-4,
60 epochs, Gaussian finalizer). It is then applied ZERO-SHOT to ADD1000 and scored at 384.

The finalizer sigma is selected on the model's own training domain (Imp1k), not on ADD1000,
so the reported number uses no ADD1000 information. For transparency the script also prints
the ADD1000-oracle-sigma score (an upper bound if the baseline were given its best footing).

  python crosscorpus_imp1k_add1000.py --epochs 60
"""
import argparse, glob, os, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from model_aspp import ASPPSaliency

IMP = "/path/to/imp1k"        # EDIT ME: Imp1k root (Fosco 2020), containing imp1k/imgs/<cat> and imp1k/maps/<cat>
ADD = "/path/to/ADD1000/ext"  # EDIT ME: dir containing ADD1000_*/ and GroundTruth1000_*/
SIZE = 448; EVAL = 384; LOSSG = 64
SIGMAS = [0, 2, 4, 6, 8, 10, 12]
SIGMA_SELECT_N = 200
IM_M = np.array([0.485, 0.456, 0.406], np.float32); IM_S = np.array([0.229, 0.224, 0.225], np.float32)


def imp1k_pairs():
    pairs = []
    for base in (os.path.join(IMP, "imp1k", "imp1k"), os.path.join(IMP, "imp1k"), IMP):
        idir_root, mdir_root = os.path.join(base, "imgs"), os.path.join(base, "maps")
        if not os.path.isdir(idir_root):
            continue
        for cat in sorted(os.listdir(idir_root)):
            idir, mdir = os.path.join(idir_root, cat), os.path.join(mdir_root, cat)
            if not os.path.isdir(idir):
                continue
            for img in sorted(glob.glob(f"{idir}/*.jpg") + glob.glob(f"{idir}/*.png")):
                stem = os.path.splitext(os.path.basename(img))[0]
                for ext in (".png", ".jpg"):
                    gt = os.path.join(mdir, stem + ext)
                    if os.path.exists(gt):
                        pairs.append((img, gt)); break
        if pairs:
            break
    return pairs


def add1000_pairs():
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


def load_map(path, g):
    m = np.asarray(Image.open(path).convert("L").resize((g, g), Image.BILINEAR), np.float64)
    m = m - m.min(); s = m.sum()
    return m / s if s > 0 else m + 1.0 / m.size


def cc(p, g):
    a, b = p - p.mean(), g - g.mean(); d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0
def sim(p, g):
    p = np.clip(p, 0, None); p = p / (p.sum() + 1e-9); g = g / (g.sum() + 1e-9)
    return float(np.minimum(p, g).sum())
def kld(p, g):
    p = np.clip(p, 0, None); p = p / (p.sum() + 1e-9); g = g / (g.sum() + 1e-9)
    return float((g * (np.log(g + 1e-8) - np.log(p + 1e-8))).sum())


@torch.no_grad()
def raw_pred(model, feats, i, dev):
    f = feats[i].float()[None].to(dev)
    pr = torch.sigmoid(F.interpolate(readout(model, f), size=(EVAL, EVAL), mode="bilinear", align_corners=False))
    return pr[0, 0].cpu().numpy().astype(np.float32)


def best_sigma(model, feats, gte, ids, dev, rng):
    sel = ids if len(ids) <= SIGMA_SELECT_N else list(rng.choice(ids, size=SIGMA_SELECT_N, replace=False))
    preds = [raw_pred(model, feats, i, dev) for i in sel]; gts = [gte[i] for i in sel]
    best_s, best_c = 0, -2
    for s in SIGMAS:
        m = float(np.mean([cc(gaussian_filter(p, s) if s > 0 else p, g) for p, g in zip(preds, gts)]))
        if m > best_c:
            best_c, best_s = m, s
    return best_s, best_c


def eval_at(model, feats, gte, ids, sigma, dev):
    C = S = K = 0.0
    for i in ids:
        p = raw_pred(model, feats, i, dev).astype(np.float64)
        if sigma > 0:
            p = gaussian_filter(p, sigma)
        C += cc(p, gte[i]); S += sim(p, gte[i]); K += kld(p, gte[i])
    n = len(ids); return C / n, S / n, K / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}", flush=True)

    feat_model = ASPPSaliency().to(dev).eval()   # backbone frozen (pretrained); readout unused for caching

    IP = imp1k_pairs(); AP = add1000_pairs()
    print(f"Imp1k pairs: {len(IP)}   ADD1000 pairs: {len(AP)}", flush=True)
    print("caching Imp1k features...", flush=True)
    if_feats, if_lossmap, if_evalmap = {}, {}, {}
    imp_ids = list(range(len(IP)))
    for k, (ip, gp) in enumerate(IP):
        if_feats[k] = backbone_feat(feat_model, ip, dev)
        if_lossmap[k] = torch.tensor(load_map(gp, LOSSG), dtype=torch.float32)
        if_evalmap[k] = load_map(gp, EVAL)
        if (k + 1) % 200 == 0:
            print(f"  imp1k {k+1}/{len(IP)}", flush=True)
    print("caching ADD1000 features...", flush=True)
    ad_feats, ad_evalmap = {}, {}
    add_ids = list(range(len(AP)))
    for k, (ip, gp) in enumerate(AP):
        ad_feats[k] = backbone_feat(feat_model, ip, dev)
        ad_evalmap[k] = load_map(gp, EVAL)
        if (k + 1) % 200 == 0:
            print(f"  add1000 {k+1}/{len(AP)}", flush=True)

    # train ASPP readout FROM SCRATCH on Imp1k (fresh random readout; frozen backbone)
    model = ASPPSaliency(pretrained=True).to(dev)     # backbone pretrained+frozen, readout random init
    model.eval()
    params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    bs = 16
    print("training readout from scratch on Imp1k...", flush=True)
    for ep in range(a.epochs):
        model.train()
        order = imp_ids.copy(); rng.shuffle(order)
        for j in range(0, len(order), bs):
            ids = order[j:j + bs]
            f = torch.stack([if_feats[i].float() for i in ids]).to(dev)
            g = torch.stack([if_lossmap[i] for i in ids]).to(dev)
            s = F.interpolate(readout(model, f), size=(LOSSG, LOSSG), mode="bilinear", align_corners=False)[:, 0]
            logp = F.log_softmax(s.reshape(len(ids), -1), 1).reshape(len(ids), LOSSG, LOSSG)
            kl = (g * (torch.log(g + 1e-12) - logp)).sum((1, 2)).mean()
            ps = torch.sigmoid(s); pc = ps - ps.mean((1, 2), keepdim=True); gc = g - g.mean((1, 2), keepdim=True)
            ccb = (pc * gc).sum((1, 2)) / (torch.sqrt((pc ** 2).sum((1, 2)) * (gc ** 2).sum((1, 2))) + 1e-8)
            loss = kl + (1 - ccb.mean())
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0:
            print(f"  epoch {ep} loss {float(loss):.3f}", flush=True)

    model.eval()
    # sigma selected on Imp1k (training domain), applied to ADD1000
    sig_imp, cc_imp = best_sigma(model, if_feats, if_evalmap, imp_ids, dev, rng)
    add_cc, add_sim, add_kld = eval_at(model, ad_feats, ad_evalmap, add_ids, sig_imp, dev)
    # oracle sigma on ADD1000 (transparency: best footing)
    sig_or, _ = best_sigma(model, ad_feats, ad_evalmap, add_ids, dev, rng)
    or_cc, or_sim, or_kld = eval_at(model, ad_feats, ad_evalmap, add_ids, sig_or, dev)

    out = {
        "baseline": "crosscorpus_imp1k_scratch_ADD1000_zeroshot",
        "protocol": f"train from scratch on Imp1k (n={len(IP)}), 448 input, {a.epochs}ep, KL+CC; eval ADD1000@{EVAL}",
        "imp1k_train_cc": cc_imp,
        "sigma_selected_on_imp1k": sig_imp,
        "ADD1000_zeroshot_CC": add_cc, "ADD1000_zeroshot_SIM": add_sim, "ADD1000_zeroshot_KLD": add_kld,
        "sigma_oracle_on_ADD1000": sig_or,
        "ADD1000_oracle_CC": or_cc, "ADD1000_oracle_SIM": or_sim, "ADD1000_oracle_KLD": or_kld,
    }
    os.makedirs("checkpoints", exist_ok=True)
    json.dump(out, open("checkpoints/crosscorpus_imp1k_add1000.json", "w"), indent=2)
    print("\n=== CROSS-CORPUS (Imp1k from scratch -> ADD1000 zero-shot) ===")
    print(f"  sigma(Imp1k-selected)={sig_imp}: CC={add_cc:.4f} SIM={add_sim:.4f} KLD={add_kld:.4f}")
    print(f"  sigma(ADD1000-oracle)={sig_or}: CC={or_cc:.4f} SIM={or_sim:.4f} KLD={or_kld:.4f}")
    print("saved -> checkpoints/crosscorpus_imp1k_add1000.json")


if __name__ == "__main__":
    main()
