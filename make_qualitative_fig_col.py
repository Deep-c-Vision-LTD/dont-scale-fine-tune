"""Single-column re-layout of fig_qualitative.png used in the letter (Fig. 2).

No new predictions are computed: the ten panels (2 held-out ADD1000 examples, ids 8 and 88,
x {image, ground truth, DeepGaze IIE, DeepGazeMSDB, fine-tuned xSoyelNet}) are cropped
pixel-for-pixel from the submitted full-width figure made by make_qualitative_fig.py and
re-tiled at single-column width with titles sized for print. Output: fig_qualitative_col.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
src = np.asarray(Image.open(os.path.join(HERE, "figures", "fig_qualitative.png")).convert("RGB"))

# Panel bounds measured from the source PNG (448 x 448 px each, frame included).
XS = [(28, 475), (502, 949), (976, 1423), (1450, 1897), (1924, 2371)]
YS = [(58, 505), (517, 964)]
TITLES = ["Original", "Ground truth", "DeepGaze IIE", "DeepGazeMSDB", "xSoyelNet"]

plt.rcParams["font.family"] = "Arial"
W = 3.45                      # in; IEEEtran \columnwidth is 3.49 in
gap = 0.025                   # in, between panels
top = 0.13                    # in, title strip
pw = (W - 4 * gap) / 5        # panel side
H = top + 2 * pw + gap
fig = plt.figure(figsize=(W, H))
for r, (y0, y1) in enumerate(YS):
    for c, (x0, x1) in enumerate(XS):
        left = c * (pw + gap) / W
        bottom = (H - top - (r + 1) * pw - r * gap) / H
        ax = fig.add_axes([left, bottom, pw / W, pw / H])
        ax.imshow(src[y0:y1 + 1, x0:x1 + 1], interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if r == 0:
            ax.set_title(TITLES[c], fontsize=6.3, pad=1.5)
out = os.path.join(HERE, "figures", "fig_qualitative_col.png")
fig.savefig(out, dpi=600)
print(f"saved {out}  ({W:.2f} x {H:.2f} in)")
