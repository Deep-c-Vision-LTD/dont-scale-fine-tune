"""
In-domain fine-tuning of DeepGaze IIE on ADD1000, for the fair in-domain
comparison in the paper. It separates the effect of in-domain data from the
representation: rather than compare our in-domain-trained readout against
zero-shot baselines, we give a general baseline the same in-domain supervision.
DeepGaze IIE is itself a frozen-backbone model with a ~3.1M-parameter trainable
readout, the same shape as ours, so the like-for-like test is to fine-tune it on
the same in-domain ADD1000 data under the identical 5-fold protocol and re-compare.

Protocol matches saliency_research/add1000_kfold_v3.py: seed-0 5-fold,
800 train / 200 test per fold, backbone frozen, readout trained on KL loss,
evaluated at a common resolution with CC / SIM / KLD. A per-fold ADD1000
centre-bias (mean train log-density) is supplied to DeepGaze as its documented
external centre-bias input, i.e. the model is adapted to ADD1000 as intended.

Inputs: images -> $ADD1000_IMAGES (ADD1000 *.jpg), maps -> $ADD1000_MAPS (*.png); outputs -> $OUT_DIR
"""
import os, glob, json, subprocess, sys, traceback
import numpy as np

EPS = 1e-8
S = 384                      # common eval resolution (also addresses R2.1)
EPOCHS = 20
BATCH = 8
LR = 1e-3
NFOLD = 5


def sh(cmd):
    print("+", cmd, flush=True); subprocess.run(cmd, shell=True, check=True)


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


def main():
    sh("pip install --no-cache-dir 'numpy<2' pillow scipy "
       "git+https://github.com/openai/CLIP.git ftfy regex git+https://github.com/matthias-k/DeepGaze.git")
    import torch, torch.nn.functional as F
    from PIL import Image
    import deepgaze_pytorch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    IMG = os.environ.get("ADD1000_IMAGES", "./images")
    MAP = os.environ.get("ADD1000_MAPS", "./maps")
    out = os.environ.get("OUT_DIR", "./model"); os.makedirs(out, exist_ok=True)

    ids = sorted({os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(IMG, "*.jpg"))},
                 key=lambda s: int(s) if s.isdigit() else s)
    print(f"ADD1000 images: {len(ids)}", flush=True)

    def load_img(i):
        im = Image.open(os.path.join(IMG, f"{i}.jpg")).convert("RGB").resize((S, S), Image.BILINEAR)
        return torch.tensor(np.array(im).transpose(2, 0, 1)[None], dtype=torch.float32)
    def load_map(i):
        m = np.asarray(Image.open(os.path.join(MAP, f"{i}.png")).convert("L")
                       .resize((S, S), Image.BILINEAR), np.float64)
        m = m - m.min(); s = m.sum()
        return m / s if s > 0 else m + 1.0/m.size

    rng = np.random.default_rng(0)
    order = rng.permutation(len(ids))
    folds = np.array_split(order, NFOLD)

    per_fold = []
    for f in range(NFOLD):
        te = [ids[i] for i in folds[f]]
        tr = [ids[i] for i in np.concatenate([folds[k] for k in range(NFOLD) if k != f])]

        # per-fold ADD1000 centre bias = mean train log-density (DeepGaze's external input)
        cb = np.zeros((S, S), np.float64)
        for i in tr:
            cb += load_map(i)
        cb = np.log(to_prob(cb) + EPS)
        cb_t = torch.tensor(cb[None], dtype=torch.float32).to(dev)

        torch.backends.cudnn.benchmark = True
        model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(dev)
        model.eval()                                      # note: .train() bug; freeze BN
        params = [p for p in model.parameters() if p.requires_grad]  # the ~3.1M readout
        opt = torch.optim.Adam(params, lr=LR)
        gts = {i: torch.tensor(load_map(i), dtype=torch.float32).to(dev) for i in tr}

        def batches(lst, bs):
            for k in range(0, len(lst), bs):
                yield lst[k:k+bs]

        for ep in range(EPOCHS):
            rng.shuffle(tr)
            last = 0.0
            for bt in batches(tr, BATCH):
                x = torch.cat([load_img(i) for i in bt], 0).to(dev)     # (B,3,S,S)
                cbb = cb_t.expand(len(bt), S, S)                        # (B,S,S)
                logd = model(x, cbb)[:, 0]                              # (B,S,S)
                logp = F.log_softmax(logd.reshape(len(bt), -1), 1).reshape(len(bt), S, S)
                g = torch.stack([gts[i] for i in bt], 0)               # (B,S,S)
                loss = (g * (torch.log(g + EPS) - logp)).sum(dim=(1, 2)).mean()
                opt.zero_grad(); loss.backward(); opt.step(); last = float(loss)
            if ep % 5 == 0:
                print(f"fold {f} epoch {ep} loss {last:.3f}", flush=True)

        # eval on held-out fold (batched)
        model.eval()
        C = SIMv = K = 0.0
        with torch.no_grad():
            for bt in batches(te, BATCH):
                x = torch.cat([load_img(i) for i in bt], 0).to(dev)
                cbb = cb_t.expand(len(bt), S, S)
                logd = model(x, cbb)[:, 0]
                for j, i in enumerate(bt):
                    pr = torch.exp(logd[j]).cpu().numpy().astype(np.float64)
                    g = load_map(i)
                    C += cc(pr, g); SIMv += sim(pr, g); K += kld(pr, g)
        n = len(te)
        per_fold.append({"fold": f, "CC": C/n, "SIM": SIMv/n, "KLD": K/n, "n": n})
        print(f"RESULT fold {f}: CC={C/n:.4f} SIM={SIMv/n:.4f} KLD={K/n:.4f}", flush=True)

    cc_mean = float(np.mean([r["CC"] for r in per_fold]))
    cc_std = float(np.std([r["CC"] for r in per_fold]))
    summary = {"model": "DeepGazeIIE_indomain_ADD1000", "per_fold": per_fold,
               "CC_mean": cc_mean, "CC_std": cc_std,
               "SIM_mean": float(np.mean([r["SIM"] for r in per_fold])),
               "KLD_mean": float(np.mean([r["KLD"] for r in per_fold])),
               "protocol": f"seed0 {NFOLD}-fold, frozen backbone, readout FT, eval@{S}"}
    json.dump(summary, open(os.path.join(out, "deepgaze_indomain_add1000.json"), "w"), indent=2)
    print(f"\nDEEPGAZE IIE in-domain ADD1000: CC={cc_mean:.4f}+/-{cc_std:.4f} "
          f"(ours=0.895; zero-shot DeepGaze=0.766)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FT FAILED:\n" + traceback.format_exc(), flush=True); sys.exit(1)
