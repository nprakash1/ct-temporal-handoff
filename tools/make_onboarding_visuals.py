#!/usr/bin/env python3
"""Build seven 16:9 onboarding visuals from audited notebook-copy results.

Uses existing matplotlib/Pillow dependencies. No training code is executed.
Verifies numeric evidence against the published notebook outputs by default.
Optional --source-dir verifies against the original executed notebook copies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "onboarding"
C = dict(ink="#172B43", muted="#516276", blue="#2469A0", frozen="#E8F2FC",
         orange="#B85B13", train="#FFF0DF", green="#287A57", text="#EAF6EF",
         red="#B23B42", warn="#FFF0F0", gray="#F2F5F8", line="#CFD8E2")

# Values transcribed from executed outputs, not inferred from repository defaults.
RESULTS = [
    dict(title="Early MERLIN baseline", context="252 test finding examples",
         notebook="train_v1_ctrate (1).ipynb", cells=[7, 9],
         rows=[("Zero-shot difference", .352), ("Learned adapter + CE", .398)],
         evidence=["macro-F1        : 0.398", "macro-F1   : 0.352"],
         note="Incomplete feature coverage; historical subset."),
    dict(title="CT-CLIP cross-modal sweep", context="1,288 test finding examples",
         notebook="fixed_supcon.ipynb", cells=[31],
         rows=[("CE", .561857), ("CE + change BCE", .575441),
               ("SupCon only", .473174), ("CE + BCE + SupCon", .473275)],
         evidence=["0.561857", "0.575441", "0.473174", "0.473275"],
         note="Real-report-derived prototypes; 5,055 train examples."),
    dict(title="Held-out sentence evaluation", context="1,288 test finding examples",
         notebook="diff_sentences_ce_eval.ipynb", cells=[12],
         rows=[("CE", .538385), ("CE + SupCon", .518454),
               ("CE + change BCE", .514537), ("CE + BCE + SupCon", .482597)],
         evidence=["0.538385", "0.518454", "0.514537", "0.482597"],
         note="Different eval prompts; scores average 3 resamplings."),
]
CURRENT = dict(notebook="train_ctclip_lastblock_temporal_colab.ipynb", cells=[32],
               tune_macro_f1=.5780196078513702, test_macro_f1=.5254796807386137,
               best_epoch=3, test_findings=795,
               status="Completed run; NOT validated as visual fine-tuning: eval VQ blocks upstream gradients.")
HANDOFF_NOTEBOOKS = {
    'train_v1_ctrate (1).ipynb': 'historical/06_merlin_temporal_adapter_and_zeroshot_executed.ipynb',
    'fixed_supcon.ipynb': 'results/01_frozen_ctclip_ce_bce_crossmodal_supcon_sweep_executed.ipynb',
    'diff_sentences_ce_eval.ipynb': 'results/02_frozen_ctclip_heldout_wording_loss_sweep_executed.ipynb',
    'train_ctclip_lastblock_temporal_colab.ipynb': 'results/03_ctclip_lastblock_ce_bce_cosine_VQ_ISSUE_executed.ipynb',
}


def text(ax, x, y, s, size=16, color=None, bold=False, ha="left", va="center"):
    return ax.text(x, y, s, fontsize=size, color=color or C["ink"],
                   weight="bold" if bold else "normal", ha=ha, va=va,
                   linespacing=1.35, zorder=5)


def box(ax, x, y, w, h, s="", fill="gray", edge="line", size=16, dashed=False):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.12",
                          facecolor=C[fill], edgecolor=C[edge], lw=1.5,
                          linestyle="--" if dashed else "-", zorder=2)
    ax.add_patch(patch)
    if s:
        label = text(ax, x+w/2, y+h/2, s, size=size, ha="center")
        ax._fit_checks.append((label, (x, y, w, h)))
    return patch


def arrow(ax, start, end, color="muted", dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=17,
                                lw=1.8, color=C[color],
                                linestyle="--" if dashed else "-", zorder=3))


def slide(number, title, subtitle):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 16), ylim=(0, 9)); ax.axis("off")
    ax._fit_checks = []
    text(ax, .65, 8.35, title, 27, bold=True)
    text(ax, .65, 7.8, subtitle, 15, C["muted"])
    ax.plot([.65, 15.35], [7.4, 7.4], color=C["line"], lw=1)
    text(ax, .65, .28, "3dCT  |  Finding-specific longitudinal progression", 10, C["muted"])
    text(ax, 15.35, .28, f"{number:02d}", 11, C["muted"], ha="right")
    return fig, ax


def overview():
    fig, ax = slide(1, "What changed between these CTs?",
                    "A CheXTemporal-inspired task: compare examinations, not just detect disease.")
    for x, name in [(1, "PRIOR CT"), (4.1, "CURRENT CT")]:
        # Diagrammatic volume stack, deliberately not synthetic medical imagery.
        for k in (2, 1, 0):
            box(ax, x+k*.10, 4.85+k*.10, 2.1, 1.5, fill="frozen", edge="blue")
        text(ax, x+1.05, 5.6, name, 17, bold=True, ha="center")
    box(ax, 1, 3.35, 5.2, .8, 'Finding query: "Pleural effusion"', "text", "green")
    arrow(ax, (6.55, 5.6), (7.4, 5.6))
    arrow(ax, (6.25, 3.75), (7.4, 4.6))
    box(ax, 7.5, 4.25, 3.1, 2.1, "Compare both scans\nfor this finding", "train", "orange", 19)
    arrow(ax, (10.7, 5.3), (11.45, 5.3))
    box(ax, 11.55, 4.25, 3.5, 2.1, "WORSENED\nSTABLE\nIMPROVED", "text", "green", 19)
    text(ax, 1, 6.85, "INPUT", 12, C["blue"], True)
    text(ax, 7.5, 6.85, "LONGITUDINAL REASONING", 12, C["orange"], True)
    text(ax, 11.55, 6.85, "DIRECTION", 12, C["green"], True)
    box(ax, .85, 1.2, 7, 1.5,
        "CheXTemporal: 5-class progression + spatial grounding\nNew / worse / stable / improved / resolved", size=16)
    box(ax, 8.15, 1.2, 7, 1.5,
        "Current CT scope: 3-class progression\nNew + worse merge; improved + resolved merge", size=16)
    text(ax, .85, .77, "Spatial correspondence and localization remain future work. CT stacks shown schematically, not patient scans.",
         11, C["muted"])
    return fig


def results():
    fig, ax = slide(2, "What the saved experiments show",
                    "Test macro-F1: descriptive results within each panel, not a cross-benchmark leaderboard.")
    for j, group in enumerate(RESULTS):
        x=.7+j*5
        box(ax, x, 2.0, 4.6, 5.0, fill="gray")
        text(ax, x+.2, 6.65, group['title'], 17, bold=True)
        text(ax, x+.2, 6.2, group['context'], 12, C['muted'])
        for i,(label,val) in enumerate(group['rows']):
            y=5.65-i*.88
            text(ax,x+.2,y,label,14)
            # Common 0..1 scale across every panel.
            ax.barh(y-.32, 3.55, height=.22, left=x+.2, color=C['line'], zorder=3)
            ax.barh(y-.32, 3.55*val, height=.22, left=x+.2,
                    color=C['green'] if 'SupCon' not in label else C['orange'], zorder=4)
            text(ax,x+4.38,y-.32,f'{val:.3f}',14,bold=True,ha='right')
        text(ax,x+.2,2.28,"Bar length: macro-F1 from 0 to 1",11,C['muted'])
    box(ax, .7, 1.13, 14.6, .62,
        "Takeaway: CE-based models are competitive; adding SupCon did not consistently improve these runs.",
        "text", "green", 16)
    text(ax, .7, .77, "Cohorts/labels differ across panels. SupCon sweeps also vary seeds and sampling; these do not isolate loss effects.",11,C['muted'])
    text(ax, .7, .53, "Middle: report-derived prototypes, 5,055 train examples. Right: held-out wording, 3 resamplings (not 3 training seeds).",10,C['muted'])
    return fig


def losses():
    fig, ax = slide(3, "Same task, different training signals",
                    'Example anchor: the learned CT-pair change embedding for "pleural effusion worsened".')
    columns = [
        ("Prototype cross-entropy", "Choose the correct direction",
         "Worsened prototype\nStable prototype\nImproved prototype",
         "Competes against 3 fixed text vectors\nfor this finding.", "blue", "frozen"),
        ("Instance InfoNCE", "Retrieve my paired sentence",
         "+ Own report-text target\n− Other batch text targets",
         "Another correct same-class sentence\ncan become a false negative.", "orange", "train"),
        ("Masked cross-modal SupCon", "Match the correct language group",
         "+ Same finding + same direction\n− Same finding + other direction\nIgnore other findings",
         "Multiple positives; compare only\nwithin this finding.", "green", "text"),
    ]
    for i,(title,goal,targets,detail,edge,fill) in enumerate(columns):
        x=.7+5*i
        text(ax,x+.1,6.86,title,18,bold=True)
        text(ax,x+.1,6.35,goal,14,C[edge])
        box(ax,x+.2,5.15,4.15,.8,"CT-pair change embedding",fill,edge,16)
        arrow(ax,(x+2.27,5.05),(x+2.27,4.55),edge)
        box(ax,x+.2,2.95,4.15,1.5,targets,fill,edge,15)
        text(ax,x+.2,2.35,detail,14,C['muted'])
    box(ax,.8,1.05,14.4,.7,
        "Last-block experiment: CE + change/stable BCE + paired cosine alignment (no SupCon).", "gray", size=16)
    text(ax,.85,.6,"Cosine alignment pulls toward one target without a contrastive denominator. BCE predicts changed vs stable, not physical size.",11,C['muted'])
    return fig


def architecture():
    fig, ax = slide(4, "Last-block CT-CLIP: intended adaptation and actual status",
                    "Each scan is encoded separately; only the downstream Difference Transformer compares prior vs current.")
    box(ax,.8,6.45,5.25,.55,"PRIOR CT  /  CURRENT CT  (shared encoder)",size=16)
    steps = [
        (5.25,1.0,"Patch embedding + 4 spatial blocks\n+ depth blocks 1–3","frozen","blue"),
        (4.32,.65,"Cached prefix: 576 × 24 × 512 / scan","gray","line"),
        (3.15,.88,"Final depth block + output normalization\nINTENDED trainable","train","orange"),
        (2.02,.78,"Vector quantizer (frozen; eval mode)\nUPSTREAM GRADIENT BLOCKAGE","warn","red"),
        (.94,.78,"Depth average → flatten → frozen projection\n512-D image embedding per scan","frozen","blue"),
    ]
    previous=6.45
    for y,h,s,fill,edge in steps:
        arrow(ax,(3.425,previous-.03),(3.425,y+h+.04))
        box(ax,.8,y,5.25,h,s,fill,edge,15)
        previous=y
    # Route embeddings to temporal adapter without crossing encoder boxes.
    ax.plot([6.1,6.5,6.5],[1.33,1.33,5.0],color=C['muted'],lw=1.8)
    arrow(ax,(6.5,5),(7.1,5))
    box(ax,7.2,4.35,3.7,1.3,"Difference Transformer\n+ finding conditioning\nTRAINABLE", "train","orange",16)
    box(ax,7.2,6.3,3.7,.65,"Finding ID / query", "text","green",16)
    arrow(ax,(9.05,6.23),(9.05,5.72),'green')
    arrow(ax,(10.95,5),(11.6,5))
    box(ax,11.7,4.35,3.5,1.3,"Direction scores\nWorsened / Stable / Improved", "text","green",15)
    box(ax,11.7,6.05,3.5,.9,"Frozen text encoder\n+ direction prototypes", "frozen","blue",15)
    arrow(ax,(13.45,6),(13.45,5.72),'blue')
    box(ax,7.2,3.15,8,.7,"Loss: CE + 0.5 × change BCE + 0.5 × cosine alignment",size=15)
    box(ax,7.2,1.5,8,1.15,
        "Run completed: tune F1 0.578; test F1 0.525 (n=795)\nDo NOT interpret as verified visual fine-tuning.","warn","red",17)
    text(ax,7.2,.96,"Pinned eval VQ blocks the backward path; adapter still learns.",13,C['red'])
    text(ax,7.2,.64,"Fix gradient flow, check weight updates, then rerun. Reuse prefix cache.",12,C['muted'])
    return fig


def roadmap():
    fig, ax = slide(5, "Next: a controlled CT progression study",
                    "Change one factor at a time. Match examples, prototypes, seeds, sampling, and evaluation.")
    rows=[
        ("01", "Validate the benchmark", "Check labels + exact prior-study correspondence; review an image-based gold subset."),
        ("02", "Prove both scans matter", "Compare finding-only, current-only, prior-only, subtraction, and a paired adapter."),
        ("03", "Isolate the training signal", "Matched CE / change BCE / cosine / SupCon runs; choose settings on tune data."),
        ("04", "Verify image fine-tuning", "Repair VQ gradient path; confirm nonzero gradients and actual visual-weight updates."),
    ]
    for i,(num,title,body) in enumerate(rows):
        y=5.82-i*1.16
        box(ax,.85,y,1, .88,num,"frozen","blue",22)
        text(ax,2.1,y+.68,title,19,bold=True)
        text(ax,2.1,y+.22,body,15,C['muted'])
    box(ax,.85,1.05,6.85,.9,"Later: EXACT backbone comparison\nAccount for preprocessing and text-space differences.",size=15,dashed=True)
    box(ax,8.0,1.05,7.15,.9,"Later: spatial grounding + acquisition robustness\nAudit scanner/kernel effects before adding BiasCon.",size=15,dashed=True)
    text(ax,.85,.63,"Final evidence: patient-level uncertainty, per-finding results, and preferably an additional untouched evaluation set.",12,C['muted'])
    return fig


def frozen_architecture():
    fig, ax = slide(7, "Frozen CT-CLIP + trainable temporal adapter",
                    "Frozen scan features → finding-conditioned comparison → direction scores. Dashed branches are training only.")
    text(ax, .8, 7.0, "BLUE: FROZEN / CACHED", 12, C['blue'], True)
    text(ax, 6.8, 7.0, "ORANGE: TRAINABLE", 12, C['orange'], True)
    for y, label, vector in [(6.2, "Prior CT", "v_prior"), (4.95, "Current CT", "v_current")]:
        box(ax, .8, y-.4, 1.3, .8, label, size=14)
        arrow(ax, (2.15,y), (2.45,y))
        box(ax, 2.5, y-.4, 1.95, .8, "CTViT + visual\nprojection", "frozen", "blue", 14)
        arrow(ax, (4.5,y), (4.8,y))
        box(ax, 4.85, y-.4, 1.4, .8, f"{vector}\n512-D cache", "frozen", "blue", 12)
        arrow(ax, (6.3,y), (6.74,y))
    text(ax, 3.48, 5.57, "Shared frozen weights", 11, C['blue'], ha="center")
    box(ax, .8, 3.75, 2.5, .55, "Finding ID (e.g., effusion)", "text", "green", 12)
    arrow(ax, (3.35,4.025), (3.65,4.025))
    box(ax, 3.7, 3.75, 2.55, .55, "Learned finding embedding", "train", "orange", 12)
    arrow(ax, (6.3,4.025), (6.74,4.025))
    box(ax, 6.8, 3.75, 3.7, 2.95, fill="train", edge="orange")
    box(ax, 6.95, 5.8, 3.4, .7,
        "512 → 256 + scan role embeddings\nTokens: [change + finding, prior, current]", "train", "orange", 11)
    arrow(ax, (8.65,5.75), (8.65,5.53), 'orange')
    box(ax, 6.95, 4.75, 3.4, .72,
        "Difference Transformer: 2 layers, 4 heads\nChange-token state h_diff (256-D)", "train", "orange", 11)
    arrow(ax, (8.65,4.7), (8.65,4.5), 'orange')
    box(ax, 6.95, 3.95, 3.4, .49, "Linear head → d_f (512-D)", "train", "orange", 14)
    box(ax, 11.25, 5.85, 4.0, 1.2,
        "Direction descriptions\nFrozen CXR-BERT + projection\n3 × 512-D prototypes / finding", "frozen", "blue", 14)
    arrow(ax, (13.25,5.8), (13.25,5.53), 'blue')
    box(ax, 11.25, 4.75, 4.0, .72,
        "Cosine(d_f, prototypes) × scale\n3 direction logits", "text", "green", 14)
    ax.plot([10.55,10.85,10.85], [4.2,4.2,5.11], color=C['muted'], lw=1.8)
    arrow(ax, (10.85,5.11), (11.19,5.11))
    arrow(ax, (13.25,4.69), (13.25,4.5), 'green')
    box(ax, 11.25, 3.85, 4.0, .59,
        "Argmax: Worsened / Stable / Improved", "text", "green", 12)

    # Separate taps: BCE uses the pre-projection state; SupCon uses d_f;
    # CE uses scaled cosine logits, not the discrete argmax prediction.
    ax.plot([6.91,6.52,6.52,3.0], [5.11,5.11,3.5,3.5],
            color=C['orange'], lw=1.5, linestyle='--')
    arrow(ax, (3,3.5), (3,3.2), 'orange', dashed=True)
    arrow(ax, (8.65,3.9), (8.65,3.2), 'orange', dashed=True)
    ax.plot([15.3,15.6,15.6,13.2], [5.11,5.11,3.5,3.5],
            color=C['green'], lw=1.5, linestyle='--')
    arrow(ax, (13.2,3.5), (13.2,3.2), 'green', dashed=True)
    box(ax, .8, 1.35, 4.4, 1.8, fill="train", edge="orange", dashed=True)
    text(ax, 3, 2.88, "AUXILIARY BCE LOSS", 15, C['orange'], True, ha="center")
    box(ax, 1.0, 1.51, 4.0, 1.04,
        "h_diff → linear head → change logit m\nBCEWithLogits(m, changed target)\n1 = worsened / improved; 0 = stable\nBinary change—not physical size", "train", "orange", 12)
    box(ax, 5.7, 1.35, 4.8, 1.8, fill="train", edge="orange", dashed=True)
    text(ax, 8.1, 2.88, "MASKED CROSS-MODAL SUPCON", 14, C['orange'], True, ha="center")
    box(ax, 5.9, 1.51, 4.4, 1.04,
        "Anchor: d_f ↔ frozen temporal-text vectors t\nTexts → frozen text tower → cached t\nPositive: same finding + same direction\nNegative: same finding + other direction", "train", "orange", 12)
    box(ax, 11.0, 1.35, 4.25, 1.8, fill="text", edge="green", dashed=True)
    text(ax, 13.125, 2.88, "DIRECTION CE LOSS", 15, C['green'], True, ha="center")
    box(ax, 11.2, 1.51, 3.85, 1.04,
        "3 direction logits + silver direction label\nCross-entropy: correct direction class\nPrototype text is frozen\nLogit scale is learned", "text", "green", 12)
    text(ax, .85, .95,
         "Loss ablations: λCE LCE + λBCE LBCE + λSupCon LSupCon. Terms switch on/off by run; SupCon ignores other findings.",
         12, C['muted'])
    text(ax, .85, .58,
         "Inference uses neither BCE nor SupCon, and needs no patient report. All CT-CLIP weights stay frozen; only the adapter/task parameters learn.",
         11, C['muted'])
    return fig


def ctrate_preprocessing():
    fig, ax = slide(6, "CT-RATE: from chest CT + reports to temporal inputs",
                    "Two preparation tracks: build valid longitudinal examples, then standardize each scan for CT-CLIP.")
    box(ax, .8, 4.4, 5.2, 2.65, fill="frozen", edge="blue")
    text(ax, 1.05, 6.65, "THE SOURCE DATA", 14, C['blue'], True)
    label = text(ax, 1.05, 6.05, "Non-contrast 3D chest CTs\nRadiology reports + metadata", 18, bold=True)
    ax._fit_checks.append((label, (.8, 4.4, 5.2, 2.65)))
    label = text(ax, 1.05, 5.12, "Local audit: 50,188 volume/report rows\n21,304 patients\nMultiple reconstructions per study", 14, C['muted'])
    ax._fit_checks.append((label, (.8, 4.4, 5.2, 2.65)))
    text(ax, 6.5, 6.72, "OUR TEMPORAL DATASET CONSTRUCTION", 14, C['green'], True)
    text(ax, 6.5, 6.12, "1  Choose one reconstruction per study; order by StudyDate.", 16)
    text(ax, 6.5, 5.65, "2  Pair consecutive studies; drop same-day / missing-date pairs.", 16)
    text(ax, 6.5, 5.18, "3  Use Findings + Impressions for report-derived direction labels.", 16)
    text(ax, 6.5, 4.71, "4  Keep eligible findings; split by patient into train / tune / test.", 16)
    text(ax, .85, 3.88, "CT-CLIP VOLUME PREPROCESSING  •  APPLY INDEPENDENTLY TO PRIOR AND CURRENT", 14, C['blue'], True)
    stages = [
        ("LOAD NIfTI", "Read intensities\n+ voxel spacing\nfrom the header"),
        ("WINDOW + SCALE", "Clip: −1000 to +200 HU\nLinearly normalize\nto [−1, 1]"),
        ("RESAMPLE", "0.75 × 0.75 × 1.5 mm\n(x / y / z)\nLinear interpolation"),
        ("CENTER CROP / PAD", "480 × 480 × 240\n(H / W / D)\nPad value: −1"),
    ]
    for i, (title, detail) in enumerate(stages):
        x = .85 + i * 3.75
        box(ax, x, 1.86, 3.25, 1.6, fill="frozen", edge="blue")
        text(ax, x + 1.625, 3.13, title, 14, C['blue'], True, ha="center")
        text(ax, x + 1.625, 2.5, detail, 15, ha="center")
        if i < 3:
            arrow(ax, (x + 3.3, 2.66), (x + 3.68, 2.66), 'blue')
    box(ax, .85, .97, 14.5, .58,
        "Input: (B, 1, 240, 480, 480) | Frozen adapter: cache 512-D vectors | Last-block: cache prefixes", "text", "green", 14)
    text(ax, .85, .59,
         "Silver labels, not image-verified gold. No scan registration; exact report-to-prior correspondence still needs verification.",
         11, C['muted'])
    return fig


def verify_sources(source_dir):
    provenance=[]
    entries=RESULTS+[dict(CURRENT, evidence=[str(CURRENT['test_macro_f1']),str(CURRENT['tune_macro_f1'])])]
    for entry in entries:
        p=(source_dir/entry['notebook'] if source_dir else
           ROOT/'notebooks'/HANDOFF_NOTEBOOKS[entry['notebook']])
        nb=json.loads(p.read_text())
        outputs=[]
        for i in entry['cells']:
            for out in nb['cells'][i].get('outputs',[]):
                value=out.get('text',out.get('data',{}).get('text/plain',[]))
                outputs.append(''.join(value) if isinstance(value,list) else value)
        evidence='\n'.join(outputs)
        for expected in entry['evidence']:
            assert expected in evidence, (p,entry['cells'],expected)
        provenance.append(dict(notebook=p.name, original_notebook=entry['notebook'],
                               source_kind='original executed copy' if source_dir else 'published handoff snapshot',
                               zero_based_cells=entry['cells'],
                               sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return provenance


def validate_layout(fig):
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        for label, (x,y,w,h) in ax._fit_checks:
            bounds=label.get_window_extent(renderer)
            a=ax.transData.transform((x,y)); b=ax.transData.transform((x+w,y+h))
            assert bounds.x0>=a[0]-2 and bounds.x1<=b[0]+2 and bounds.y0>=a[1]-2 and bounds.y1<=b[1]+2, label.get_text()
        for label in ax.texts:
            bounds=label.get_window_extent(renderer)
            assert bounds.x0>=0 and bounds.x1<=fig.bbox.width and bounds.y0>=0 and bounds.y1<=fig.bbox.height, label.get_text()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--source-dir',type=Path,help='Optional directory of executed CT notebook copies')
    args=parser.parse_args()
    provenance=verify_sources(args.source_dir.resolve() if args.source_dir else None)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'3dct-onboarding','pdf.fonttype':42})
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    builders=[('01_project_overview',overview),('02_historical_results',results),
              ('03_loss_comparison',losses),('04_current_architecture',architecture),('05_next_experiments',roadmap),
              ('06_ctrate_preprocessing',ctrate_preprocessing),('07_frozen_ctclip_architecture',frozen_architecture)]
    with PdfPages(out/'onboarding_visuals.pdf',metadata={'Title':'3dCT onboarding visuals','CreationDate':None,'ModDate':None}) as pdf:
        for name,build in builders:
            fig=build();validate_layout(fig)
            fig.savefig(out/f'{name}.png',dpi=200)
            fig.savefig(out/f'{name}.svg',metadata={'Date':None})
            svg_path=out/f'{name}.svg'
            svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')
            pdf.savefig(fig);plt.close(fig)
            print('Rendered and layout-checked:',out/f'{name}.png')
    sheet=Image.new('RGB',(1280,((len(builders)+1)//2)*395),'#E6EBF0');draw=ImageDraw.Draw(sheet)
    for i,(name,_) in enumerate(builders):
        with Image.open(out/f'{name}.png') as image:
            assert image.size==(3200,1800)
            thumb=ImageOps.contain(image.convert('RGB'),(620,349))
            x=10+(i%2)*640;y=10+(i//2)*395
            sheet.paste(thumb,(x,y));draw.text((x,y+355),name,fill=C['ink'])
    sheet.save(out/'contact_sheet.png')
    snapshot=dict(results=RESULTS,current_run=CURRENT,verified_notebook_sources=provenance,
                  caveat='Saved outputs only; no rerun. Cross-panel cohorts differ. See README for interpretation.',
                  gradient_check='Pinned vector-quantize-pytorch 1.1.2 eval path: reproduced upstream grad=None; no direct checkpoint-delta inspection.')
    (out/'results_provenance.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    print('PDF, contact sheet, and provenance:',out)


if __name__=='__main__':
    main()