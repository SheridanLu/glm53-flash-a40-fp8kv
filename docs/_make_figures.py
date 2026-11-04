#!/usr/bin/env python3
"""Generate the three README figures as self-contained SVGs (light surface, validated palette)."""
import html
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
os.makedirs(OUT, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"
BLUE = "#2a78d6"     # series 1
ORANGE = "#eb6834"   # series 2
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def svg(w, h, body, title, desc):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'role="img" aria-labelledby="t d" font-family="{FONT}">\n'
        f'<title id="t">{title}</title><desc id="d">{desc}</desc>\n'
        f'<rect width="{w}" height="{h}" rx="10" fill="{SURFACE}" stroke="{GRID}"/>\n{body}</svg>\n'
    )


def text(x, y, s, size=13, fill=INK, weight=400, anchor="start"):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(s)}</text>\n'


# ---------------------------------------------------------------- Figure 1: KV capacity
def fig_capacity():
    w, h = 760, 250
    left, right = 150, 40
    plot = w - left - right
    vmax = 2_600_000
    rows = [("BF16 KV", 955_759, BLUE, "1.82x of a 512K request"),
            ("FP8 KV", 1_771_693, ORANGE, "3.38x of a 512K request")]
    b = text(24, 34, "KV cache capacity on 8x A40", 16, INK, 600)
    b += text(24, 54, "tokens in the paged KV pool, same GPU memory budget (11.53 GiB per card)", 12, INK2)
    for t in range(0, 2_500_001, 500_000):
        x = left + plot * t / vmax
        b += f'<line x1="{x}" y1="76" x2="{x}" y2="184" stroke="{GRID}"/>\n'
        b += text(x, 202, f"{t/1e6:.1f}M" if t else "0", 11, MUTED, 400, "middle")
    for i, (lab, v, col, note) in enumerate(rows):
        y = 88 + i * 52
        bw = plot * v / vmax
        b += text(left - 14, y + 22, lab, 13, INK, 500, "end")
        b += f'<path d="M{left} {y} h{bw-4} a4 4 0 0 1 4 4 v28 a4 4 0 0 1 -4 4 h-{bw-4} z" fill="{col}"/>\n'
        b += text(left + bw + 8, y + 18, f"{v:,}", 13, INK, 600)
        b += text(left + bw + 8, y + 34, note, 11, INK2)
    b += text(24, 236, "FP8 pool is 1.85x larger. Source: engine log 'GPU KV cache size', results/ and docs/validation.md", 11, MUTED)
    return svg(w, h, b, "KV cache capacity", "BF16 KV holds 955,759 tokens, FP8 KV holds 1,771,693 tokens.")


# ---------------------------------------------------------------- Figure 2: page layout
def fig_layout():
    w, h = 760, 262
    left, right = 150, 40
    plot = w - left - right
    scale = plot / 1024.0
    b = text(24, 34, "KV row layout per token per layer (NoPE sparse MLA)", 16, INK, 600)
    b += text(24, 54, "GLM-5.3-Flash has qk_rope_head_dim = 0, so a row is 512 latent values, not 576", 12, INK2)
    # BF16 row
    y1 = 82
    b += text(left - 14, y1 + 22, "BF16", 13, INK, 500, "end")
    b += f'<rect x="{left}" y="{y1}" width="{1024*scale}" height="34" rx="4" fill="{BLUE}"/>\n'
    b += text(left + 12, y1 + 22, "512 x bf16  =  1024 B", 13, "#ffffff", 600)
    # FP8 row
    y2 = 142
    b += text(left - 14, y2 + 22, "FP8 (this repo)", 13, INK, 500, "end")
    b += f'<rect x="{left}" y="{y2}" width="{512*scale}" height="34" rx="4" fill="{ORANGE}"/>\n'
    b += text(left + 12, y2 + 22, "512 x fp8e4m3  =  512 B", 13, "#ffffff", 600)
    b += f'<rect x="{left + 512*scale + 2}" y="{y2}" width="{16*scale}" height="34" rx="4" fill="{INK2}"/>\n'
    b += f'<line x1="{left + 512*scale + 2 + 8*scale}" y1="{y2 + 36}" x2="{left + 512*scale + 2 + 8*scale}" y2="{y2 + 52}" stroke="{INK2}"/>\n'
    b += text(left + 512 * scale + 2 + 8 * scale + 8, y2 + 66, "4 x fp32 scales, one per 128 values  =  16 B", 12, INK2)
    b += text(left + 512 * scale + 40, y2 + 22, "528 B total  (1.94x smaller)", 13, INK, 600)
    b += text(24, 246, "DeepSeek-style fp8_ds_mla pages are 656 B because they keep a 64-dim bf16 RoPE tail; a NoPE model has none.", 11, MUTED)
    return svg(w, h, b, "KV row layout", "A BF16 row is 1024 bytes. The FP8 row is 512 bytes of fp8 plus 16 bytes of scales, 528 bytes total.")


# ---------------------------------------------------------------- Figure 3: prompt length distribution
def fig_lengths():
    w, h = 760, 330
    left, right, top, bottom = 56, 30, 84, 66
    pw, ph = w - left - right, h - top - bottom
    data = [("<=5K", 86), ("5-10K", 89), ("10-20K", 31), ("20-50K", 156),
            ("50-100K", 206), ("100-200K", 137), ("200-500K", 527)]
    vmax = 600
    n = len(data)
    slot = pw / n
    bw = slot * 0.58
    b = text(24, 34, "Real traffic: prompt length per request", 16, INK, 600)
    b += text(24, 54, "1,232 requests over 2h40m, mean 167,820 tokens (includes a few of our own test requests)", 12, INK2)
    for t in range(0, vmax + 1, 150):
        y = top + ph - ph * t / vmax
        b += f'<line x1="{left}" y1="{y}" x2="{left + pw}" y2="{y}" stroke="{GRID}"/>\n'
        b += text(left - 8, y + 4, str(t), 11, MUTED, 400, "end")
    for i, (lab, v) in enumerate(data):
        x = left + slot * i + (slot - bw) / 2
        bh = ph * v / vmax
        y = top + ph - bh
        b += f'<path d="M{x} {top+ph} v-{bh-4} a4 4 0 0 1 4 -4 h{bw-8} a4 4 0 0 1 4 4 v{bh-4} z" fill="{BLUE}"/>\n'
        b += text(x + bw / 2, y - 6, str(v), 12, INK, 600, "middle")
        b += text(x + bw / 2, top + ph + 20, lab, 12, INK2, 400, "middle")
    b += text(left + pw / 2, h - 28, "prompt tokens", 12, INK2, 400, "middle")
    b += text(24, h - 10, "Source: vllm:request_prompt_tokens histogram; 43% of requests carry 200K+ tokens.", 11, MUTED)
    return svg(w, h, b, "Prompt length distribution",
               "527 of 1232 requests are between 200K and 500K tokens; the mean is 167,820 tokens.")


for name, fn in (("kv-capacity.svg", fig_capacity), ("kv-row-layout.svg", fig_layout), ("prompt-lengths.svg", fig_lengths)):
    open(os.path.join(OUT, name), "w").write(fn())
    print("wrote", name)
