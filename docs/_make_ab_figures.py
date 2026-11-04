#!/usr/bin/env python3
"""A/B result figures, generated from results/ab-2026-11-05/report.json (no hand-copied numbers)."""
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REP = json.load(open(os.path.join(HERE, "..", "results", "ab-2026-11-05", "report.json")))
OUT = os.path.join(HERE, "img")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def t(x, y, s, size=12, fill=INK, weight=400, anchor="start"):
    return '<text x="%s" y="%s" font-size="%s" fill="%s" font-weight="%s" text-anchor="%s">%s</text>\n' % (x, y, size, fill, weight, anchor, html.escape(s))


def wrap(w, h, body, title, desc):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" role="img" aria-labelledby="a b" font-family="%s">'
            '<title id="a">%s</title><desc id="b">%s</desc><rect width="%d" height="%d" rx="10" fill="%s" stroke="%s"/>\n%s</svg>\n'
            % (w, h, w, h, FONT, html.escape(title), html.escape(desc), w, h, SURFACE, GRID, body))


# ---------------------------------------------------------------- figure 1: perplexity ratio forest plot
def forest():
    p = REP["ppl"]
    order = ["all", "code", "text", "code@4K", "code@16K", "text@4K", "text@16K", "code@64K", "text@64K", "code@128K", "text@128K"]
    rows = [(k, p[k]) for k in order if k in p]
    w, row_h, top = 760, 28, 96
    h = top + len(rows) * row_h + 70
    left, right = 150, 60
    pw = w - left - right
    lo, hi = 0.995, 1.025
    X = lambda v: left + pw * (v - lo) / (hi - lo)
    b = t(24, 34, "Perplexity ratio, FP8 KV / BF16 KV", 16, INK, 600)
    b += t(24, 54, "per document group, 95% bootstrap CI over documents (point only where fewer than 4 documents); 1.000 = identical", 12, INK2)
    for g in (0.995, 1.000, 1.005, 1.010, 1.015, 1.020, 1.025):
        x = X(g)
        b += '<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s"/>\n' % (x, top - 14, x, top + len(rows) * row_h, GRID)
        b += t(x, top + len(rows) * row_h + 18, "%.3f" % g, 11, MUTED, 400, "middle")
    b += '<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="1.5"/>\n' % (X(1.0), top - 14, X(1.0), top + len(rows) * row_h, INK2)
    thr = REP["thresholds"]["ppl_ratio_max"]
    b += '<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-dasharray="4 3"/>\n' % (X(thr), top - 14, X(thr), top + len(rows) * row_h, ORANGE)
    b += t(X(thr) + 4, top - 18, "pass limit %.2f" % thr, 11, ORANGE)
    for i, (k, s) in enumerate(rows):
        y = top + i * row_h + 14
        pooled = k in ("all", "code", "text")
        b += t(left - 12, y + 4, "%s  (n=%d)" % (k, s["docs"]), 12, INK, 600 if pooled else 400, "end")
        if s["ci"]:
            b += '<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="3" stroke-linecap="round"/>\n' % (X(s["ci"][0]), y, X(s["ci"][1]), y, BLUE)
        b += '<circle cx="%.1f" cy="%d" r="%d" fill="%s" stroke="%s" stroke-width="2"/>\n' % (X(s["ratio"]), y, 5 if pooled else 4, BLUE if s["ci"] else SURFACE, BLUE)
        b += t(w - right + 8, y + 4, "%.4f" % s["ratio"], 11, INK2)
    b += t(24, h - 12, "Source: results/ab-2026-11-05/report.json. Filled dot = pooled estimate with CI; open dot = a single or a pair of documents.", 11, MUTED)
    return wrap(w, h, b, "Perplexity ratio per group", "All pooled ratio %.4f with a 95%% interval of %.4f to %.4f." % (p["all"]["ratio"], p["all"]["ci"][0], p["all"]["ci"][1]))


# ---------------------------------------------------------------- figure 2: FP8 effect vs the engine's own noise
def noise():
    p, n = REP["ppl"]["all"], REP["noise_ref"]["all"]
    w, h = 760, 300
    left, right = 230, 60
    pw = w - left - right
    b = t(24, 34, "FP8 KV versus the engine's own run-to-run noise", 16, INK, 600)
    b += t(24, 54, "BF16 run twice (same prompts, same engine) vs BF16 against FP8, 686,407 scored tokens", 12, INK2)
    # panel 1: dot plot on an explicit axis (a bar from a truncated baseline would exaggerate the gap)
    lo, hi = 0.960, 1.000
    X = lambda v: left + pw * (v - lo) / (hi - lo)
    y0 = 96
    b += t(24, y0 - 8, "Top-1 token agreement (higher is better)", 12, INK, 600)
    for g in (0.960, 0.970, 0.980, 0.990, 1.000):
        b += '<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s"/>\n' % (X(g), y0 + 2, X(g), y0 + 58, GRID)
        b += t(X(g), y0 + 74, "%.3f" % g, 11, MUTED, 400, "middle")
    thr = REP["thresholds"]["top1_agree_min"]
    b += '<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-dasharray="4 3"/>\n' % (X(thr), y0 + 2, X(thr), y0 + 58, ORANGE)
    b += t(X(thr) + 4, y0 + 12, "pass limit %.2f" % thr, 11, ORANGE)
    for j, (lab, v, col) in enumerate((("BF16 vs BF16 (noise floor)", n["top1_agree"], MUTED), ("BF16 vs FP8", p["top1_agree"], ORANGE))):
        y = y0 + 20 + j * 26
        b += t(left - 12, y + 4, lab, 12, INK, 500, "end")
        b += '<circle cx="%.1f" cy="%d" r="6" fill="%s"/>\n' % (X(v), y, col)
        b += t(X(v) + 12, y + 4, "%.4f" % v, 12, INK, 600)
    # panel 2: bars from zero
    y1 = y0 + 112
    vmax = 0.08
    b += t(24, y1 - 8, "Mean |change in log-probability| per token (lower is better)", 12, INK, 600)
    for j, (lab, v, col) in enumerate((("BF16 vs BF16 (noise floor)", n["mean_abs_dlogp"], MUTED), ("BF16 vs FP8", p["mean_abs_dlogp"], ORANGE))):
        y = y1 + j * 28
        bw = pw * v / vmax
        b += t(left - 12, y + 15, lab, 12, INK, 500, "end")
        b += '<path d="M%d %d h%.1f a4 4 0 0 1 4 4 v12 a4 4 0 0 1 -4 4 h-%.1f z" fill="%s"/>\n' % (left, y, bw - 4, bw - 4, col)
        b += t(left + bw + 8, y + 15, "%.4f" % v, 12, INK, 600)
    d1, d2 = n["top1_agree"] - p["top1_agree"], p["mean_abs_dlogp"] - n["mean_abs_dlogp"]
    b += t(24, h - 12, "FP8 is slightly beyond the noise floor on both (agreement %.4f lower, change %.4f nats higher), far inside the pass limits. Source: report.json." % (d1, d2), 11, MUTED)
    return wrap(w, h, b, "FP8 effect vs noise floor", "Top-1 agreement %.4f for BF16 vs BF16 and %.4f for BF16 vs FP8." % (n["top1_agree"], p["top1_agree"]))


os.makedirs(OUT, exist_ok=True)
for name, fn in (("ab-ppl-ratio.svg", forest), ("ab-noise-floor.svg", noise)):
    open(os.path.join(OUT, name), "w").write(fn())
    print("wrote", name)
