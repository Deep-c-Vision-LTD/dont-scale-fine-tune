"""
Qualitative comparison figure: original image | ground truth | DeepGaze IIE |
our fine-tuned model, for 2 example ADD1000 ads. Saved to fig_qualitative.png.
"""
import glob, os, sys
import numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from model_aspp import ASPPSaliency

ADD_EXT = "/path/to/ADD1000/ext"    # EDIT ME
ADD_FLAT = "/path/to/ADD1000/flat"  # EDIT ME
CKPT = "/path/to/your_finetuned_checkpoint.pt"  # EDIT ME: train one with kfold_train_eval.py
SIZE = 448
IM_M = np.array([0.485, 0.456, 0.406], np.float32); IM_S = np.array([0.229, 0.224, 0.225], np.float32)
EVAL = 384
EXAMPLE_IDS = ["8", "88"]  # BOTH verified in the displayed checkpoint's held-out test split (seed-0)

dev = "mps" if torch.backends.mps.is_available() else "cpu"

# ---- our fine-tuned model ----
ours = ASPPSaliency().to(dev)
sd = torch.load(CKPT, map_location="cpu")
if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
    sd = sd["model"]
ours.load_state_dict(sd, strict=False)
ours.eval()


@torch.no_grad()
def ours_predict(path):
    im = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    a = (np.asarray(im, np.float32) / 255 - IM_M) / IM_S
    x = torch.from_numpy(a.transpose(2, 0, 1))[None].to(dev)
    f = ours._features(x)
    s = ours.head(ours.aspp(f))
    cb = F.interpolate(ours.center_bias, size=s.shape[-2:], mode="bilinear", align_corners=False)
    s = s + cb
    pr = torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear", align_corners=False))
    # Gaussian finalizer (sigma=8 at 384^2, the train-selected value; see paper Method)
    return gaussian_filter(pr[0, 0].cpu().numpy(), 8)


# ---- DeepGaze IIE + MSDB with fitted ad-domain centerbias (CPU: float64 buffers, MPS can't) ----
import deepgaze_pytorch
dg = deepgaze_pytorch.DeepGazeIIE(pretrained=True)
dg.eval()
msdb = deepgaze_pytorch.DeepGazeMSDB(pretrained=True)
msdb.eval()  # NOT chained: DeepGazeMSDB.train() doesn't return self, so .eval() chains to None


def to_prob(x):
    x = np.clip(x, 0, None); s = x.sum()
    return x / s if s > 0 else np.full_like(x, 1.0 / x.size)


def fit_centerbias():
    gts = glob.glob(os.path.join(ADD_FLAT, "maps", "*.png"))
    acc = np.zeros((SIZE, SIZE), np.float64)
    for gp in gts[:500]:
        gt = np.asarray(Image.open(gp).convert("L").resize((SIZE, SIZE), Image.BILINEAR), np.float64)
        acc += to_prob(gt)
    acc /= len(gts[:500])
    return np.log(acc + 1e-8)


CB = fit_centerbias()


@torch.no_grad()
def dg_predict(path):
    im = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1))[None]
    cbt = torch.from_numpy(CB.astype(np.float32))[None]
    logd = dg(x, cbt)
    out = torch.exp(logd)[0, 0].cpu().numpy()
    return np.asarray(Image.fromarray(out).resize((EVAL, EVAL), Image.BILINEAR))


@torch.no_grad()
def msdb_predict(path):
    im = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1))[None]
    cbt = torch.from_numpy(CB.astype(np.float32))[None]
    logd = msdb(x, cbt, pixel_per_dva=21.7, dataset=None)
    out = torch.exp(logd)[0].cpu().numpy()
    return np.asarray(Image.fromarray(out).resize((EVAL, EVAL), Image.BILINEAR))


def find_pair(img_id):
    ip = glob.glob(os.path.join(ADD_EXT, "ADD1000_*", f"{img_id}.jpg"))[0]
    gp = glob.glob(os.path.join(ADD_EXT, "GroundTruth1000_*", f"{img_id}.png"))[0]
    return ip, gp


fig, axes = plt.subplots(len(EXAMPLE_IDS), 5, figsize=(12, 2.5 * len(EXAMPLE_IDS)))
col_titles = ["Original", "Ground truth", "DeepGaze IIE", "DeepGazeMSDB", "xSoyelNet (fine-tuned)"]

for row, img_id in enumerate(EXAMPLE_IDS):
    ip, gp = find_pair(img_id)
    orig = np.asarray(Image.open(ip).convert("RGB").resize((EVAL, EVAL), Image.BILINEAR))
    gt = np.asarray(Image.open(gp).convert("L").resize((EVAL, EVAL), Image.BILINEAR))
    dg_pred = dg_predict(ip)
    msdb_pred = msdb_predict(ip)
    our_pred = ours_predict(ip)

    for col, (img, cmap) in enumerate([(orig, None), (gt, "hot"), (dg_pred, "hot"),
                                        (msdb_pred, "hot"), (our_pred, "hot")]):
        ax = axes[row, col]
        ax.imshow(img, cmap=cmap)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(col_titles[col], fontsize=10)

fig.tight_layout()
out = "figures/fig_qualitative.png"
fig.savefig(out, dpi=200)
print(f"saved {out}")
