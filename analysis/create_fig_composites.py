# -*- coding: utf-8 -*-
"""Compose the merged main/supplementary figure panels at native 300 dpi.
Main: 图1(tier_composition+tierC_reprod), 图2(forest_sepsis+forest_tier),
      图4(roc+calib).  Supplementary S2(dca+shap).  S3 = label_auc (single, unchanged).
S1 = strobe_flow (unchanged)."""
import os
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, "fig")
os.makedirs(FIG, exist_ok=True)
# Font: first existing candidate is used (Windows / Linux / macOS).
# Override with the NPSLE_FONT environment variable if needed.
FONT = next((p for p in (
    os.environ.get("NPSLE_FONT", ""),
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
) if p and os.path.exists(p)), None)

def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except Exception:
        return ImageFont.load_default()

def composite(name, imgA, imgB, labelA, labelB, target_h=None, pad_top=130, target_w=None, vstack=False):
    A = Image.open(os.path.join(FIG, imgA)).convert("RGB")
    B = Image.open(os.path.join(FIG, imgB)).convert("RGB")
    fs = max(56, int((target_h or 1500) * 0.05))  # panel-label size scales with panel height
    f = font(fs)
    out = os.path.join(FIG, name)

    if vstack:
        # stack A above B so the composite becomes tall (less flat). Each panel
        # is fit to a common width; (a)/(b) sit in the top padding band of each
        # panel (top-left), clear of y-axis tick/category labels.
        if target_w is None:
            target_w = max(A.width, B.width)
        def fitw(im, w):
            if im.width == w:
                return im
            h = int(round(im.height * w / im.width))
            return im.resize((w, h), Image.LANCZOS)
        A = fitw(A, target_w)
        B = fitw(B, target_w)
        gap = 130  # band between panels, holds the (b) label
        W = max(A.width, B.width)
        H = pad_top + A.height + gap + B.height
        canvas = Image.new("RGB", (W, H), "white")
        canvas.paste(A, (0, pad_top))
        canvas.paste(B, (0, pad_top + A.height + gap))
        d = ImageDraw.Draw(canvas)
        d.text((22, 22), labelA, font=f, fill=(20, 20, 20))
        # (b) sits INSIDE the gap band above panel B (not below it)
        d.text((22, pad_top + A.height + 22), labelB, font=f, fill=(20, 20, 20))
        canvas.save(out, dpi=(300, 300))
        print(f"wrote {out}  {canvas.width}x{canvas.height}")
        return

    # ---- horizontal side-by-side (default) ----
    if target_h is None:
        target_h = max(A.height, B.height)
    def fit(im, h):
        if im.height == h:
            return im
        w = int(round(im.width * h / im.height))
        return im.resize((w, h), Image.LANCZOS)
    A = fit(A, target_h)
    B = fit(B, target_h)
    gap = 80
    W = A.width + B.width + gap
    # reserve a top padding band so the (a)/(b) labels sit ABOVE the plot area
    # (top-left), never colliding with y-axis tick/category labels
    canvas = Image.new("RGB", (W, target_h + pad_top), "white")
    canvas.paste(A, (0, pad_top))
    canvas.paste(B, (A.width + gap, pad_top))
    d = ImageDraw.Draw(canvas)
    # panel labels at top-LEFT (inside the padding band)
    d.text((22, 22), labelA, font=f, fill=(20, 20, 20))
    d.text((A.width + gap + 22, 22), labelB, font=f, fill=(20, 20, 20))
    canvas.save(out, dpi=(300, 300))
    print(f"wrote {out}  {canvas.width}x{canvas.height}")

# 图1: side-by-side; base panels regenerated as portrait (figsize 7x8) so each
# panel is tall. target_h raised so A/B are clearly taller than in the old wide layout.
composite("fig1_tier_dual.png", "tier_composition.png", "tierC_reprod.png", "(a)", "(b)", target_h=1700)
composite("fig2_sepsis_dual.png", "forest_sepsis_first.png", "forest_tier_first.png", "(a)", "(b)", target_h=1300)
composite("fig4_roc_calib_dual.png", "roc_xgb.png", "calib_xgb.png", "(a)", "(b)", target_h=1560)
composite("supp_fig2_dca_shap.png", "dca_xgb.png", "shap_mimiciv.png", "(a)", "(b)", target_h=1560)
print("done")
