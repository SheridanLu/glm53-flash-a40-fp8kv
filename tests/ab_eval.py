#!/usr/bin/env python3
"""BF16-vs-FP8 KV A/B harness (Python stdlib only).

  prep     build prompts (real code + real English, 4K..128K tokens), greedy prompts, an MMLU sample
  ppl      per-token logprob of the actual token + argmax id for every prompt  (prompt_logprobs)
  greedy   128-token greedy continuations
  mmlu     0-shot multiple choice scored by the logprob of A/B/C/D
  compare  pre-registered pass/fail between two runs (+ run-to-run noise floors)
  selftest negative controls: a good candidate must PASS and a degraded one must FAIL

Pre-registered thresholds live in THRESHOLDS below and in docs/ab-protocol.md. They were fixed before any
BF16/FP8 data was collected; changing them after looking at results invalidates the verdict.

SAFETY: `ppl` asks the engine for logits at every prompt position, i.e. chunk x vocabulary (~155K) in fp32 on
rank 0. On an instance with the production memory margin this can OOM the engine. Run it only against an
instance started with a small --max-num-batched-tokens (<= 2048) and lowered --gpu-memory-utilization, and set
AB_EVAL_SAFE=1 to acknowledge that.
"""
import argparse
import glob
import hashlib
import http.client
import json
import math
import os
import random
import sys
import tempfile
import time

HOST = os.environ.get("AB_HOST", "127.0.0.1")
PORT = int(os.environ.get("AB_PORT", "18010"))
MODEL = os.environ.get("AB_MODEL", "glm-5.3-flash")

THRESHOLDS = {
    "ppl_ratio_max": 1.01,        # candidate / reference perplexity, pooled per kind and overall
    "ppl_ci_upper_max": 1.02,     # upper end of the 95% document-bootstrap CI (when >= 4 documents)
    "top1_agree_min": 0.97,       # fraction of positions where candidate and reference argmax token agree
    "top1_vs_noise_max_gap": 0.02,  # and not more than 2 pts below the reference's own run-to-run agreement
    "mmlu_drop_max": 0.03,        # reference accuracy minus candidate accuracy
    "mmlu_invalid_max": 0.02,     # fraction of answers with no A-D among the top-20 tokens
}

PPL_SPEC = [(4096, 8), (16384, 4), (65536, 2), (131072, 1)]   # (tokens, documents) per kind
GREEDY_PER_KIND = 20
GREEDY_NEW_TOKENS = 128


# ----------------------------------------------------------------------------- plumbing
def post(path, payload, timeout=7200):
    c = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    try:
        c.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
        r = c.getresponse()
        body = r.read()
        if r.status != 200:
            raise RuntimeError("HTTP %d from %s: %r" % (r.status, path, body[:300]))
        return json.loads(body)
    finally:
        c.close()


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- prep
def read_text(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8", "replace")


def corpus_files(root, kind):
    if kind == "code":
        fs = sorted(glob.glob(os.path.join(root, "vllm", "**", "*.py"), recursive=True))
        return [f for f in fs if 2000 <= os.path.getsize(f) <= 60000]
    fs = sorted(glob.glob(os.path.join(root, "*.dist-info", "METADATA")))
    return [f for f in fs if os.path.getsize(f) >= 3000]


def piece(path, root, kind):
    t = read_text(path)
    if kind == "text":                                   # drop the RFC822-style header block
        t = t.split("\n\n", 1)[-1]
        head = "# package: " + os.path.basename(os.path.dirname(path)) + "\n"
    else:
        head = "# file: " + os.path.relpath(path, root) + "\n"
    return head + t + "\n\n"


def build_doc(files, root, kind, start, budget_chars):
    parts, size, i = [], 0, start
    while size < budget_chars:
        p = piece(files[i % len(files)], root, kind)
        parts.append(p)
        size += len(p)
        i += 1
    return "".join(parts)[:budget_chars], i


def chars_per_token(kind, files, root):
    sample = "".join(piece(f, root, kind) for f in files[:40])[:120000]
    try:
        n = post("/tokenize", {"model": MODEL, "prompt": sample}, timeout=120)["count"]
        return len(sample) / float(n)
    except Exception as e:                                # fall back to a conservative guess
        print("tokenize unavailable (%s); using 3.2 chars/token" % e, file=sys.stderr)
        return 3.2


def cmd_prep(a):
    os.makedirs(a.out, exist_ok=True)
    spec = [(256, 2)] if a.small else PPL_SPEC
    ppl, greedy = [], []
    for kind in ("code", "text"):
        files = corpus_files(a.root, kind)
        if not files:
            sys.exit("no %s corpus files under %s" % (kind, a.root))
        rnd = random.Random(1234)
        rnd.shuffle(files)
        cpt = chars_per_token(kind, files, a.root)
        print("%s: %d files, %.2f chars/token" % (kind, len(files), cpt))
        cursor = 0
        for tokens, ndocs in spec:
            for k in range(ndocs):
                text, cursor = build_doc(files, a.root, kind, cursor, int(tokens * cpt * 0.97))
                ppl.append({"id": "%s-%dk-%d" % (kind, tokens // 1024 or 0, k) if tokens >= 1024 else "%s-%d-%d" % (kind, tokens, k),
                            "kind": kind, "bucket": tokens, "text": text})
        for k in range(2 if a.small else GREEDY_PER_KIND):
            text, cursor = build_doc(files, a.root, kind, cursor, 5000)
            cut = text.rfind("\n")
            greedy.append({"id": "greedy-%s-%d" % (kind, k), "kind": kind, "text": text[: cut if cut > 1000 else len(text)]})
    for name, rows in (("ppl.jsonl", ppl), ("greedy.jsonl", greedy)):
        with open(os.path.join(a.out, name), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.mmlu_jsonl:
        qs = load_jsonl(a.mmlu_jsonl)
        random.Random(7).shuffle(qs)
        qs = qs[: (5 if a.small else a.mmlu_n)]
        with open(os.path.join(a.out, "mmlu.jsonl"), "w") as f:
            for i, q in enumerate(qs):
                subj = q["subject"].replace("_", " ")
                body = "The following is a multiple choice question about %s.\n\n%s\n" % (subj, q["question"].strip())
                for letter, c in zip("ABCD", q["choices"]):
                    body += "%s. %s\n" % (letter, c)
                body += "Answer:"
                f.write(json.dumps({"id": "mmlu-%d" % i, "subject": q["subject"], "answer": "ABCD"[q["answer"]], "prompt": body}, ensure_ascii=False) + "\n")
    man = {n: sha256_file(os.path.join(a.out, n)) for n in sorted(os.listdir(a.out)) if n.endswith(".jsonl")}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print("prompts written:", {k: v[:12] for k, v in man.items()})


# ----------------------------------------------------------------------------- runs
def parse_prompt_logprobs(choice):
    """-> (prompt_token_ids, logprob of each actual token for positions 1.., argmax token id per position)."""
    ids, pl = choice.get("prompt_token_ids"), choice.get("prompt_logprobs")
    if ids is None or pl is None:
        raise ValueError("response lacks prompt_token_ids/prompt_logprobs - the request needs return_token_ids=true and prompt_logprobs")
    if len(pl) != len(ids) or pl[0] is not None:
        raise ValueError("unexpected prompt_logprobs shape: len %s vs %d tokens" % (None if pl is None else len(pl), len(ids)))
    lp, top1 = [], []
    for i in range(1, len(ids)):
        e = pl[i]
        lp.append(round(e[str(ids[i])]["logprob"], 4))
        top1.append(next((int(k) for k, v in e.items() if v.get("rank") == 1), -1))
    return ids, lp, top1


def cmd_parsecheck(a):
    """Offline check of the prompt_logprobs parser against a saved /v1/completions response."""
    d = json.load(open(a.response))
    ids, lp, top1 = parse_prompt_logprobs(d["choices"][0])
    print("tokens %d, scored positions %d, mean nll/token %.3f, argmax==actual on %d positions" % (
        len(ids), len(lp), -sum(lp) / len(lp), sum(1 for i, t in enumerate(top1) if t == ids[i + 1])))
    bad = [i for i, t in enumerate(top1) if t == -1]
    assert not bad, "positions without a rank-1 entry: %s" % bad
    assert all(x <= 0 for x in lp), "positive logprob"
    print("parse OK")


def cmd_ppl(a):
    if os.environ.get("AB_EVAL_SAFE") != "1":
        sys.exit("refusing to run: set AB_EVAL_SAFE=1 only against an instance started with "
                 "--max-num-batched-tokens <= 2048 and lowered --gpu-memory-utilization (see module docstring)")
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "ppl.jsonl")
    done = {r["id"] for r in load_jsonl(out)}
    items = load_jsonl(os.path.join(a.prompts, "ppl.jsonl"))
    for it in items:
        if it["id"] in done:
            continue
        t0 = time.time()
        d = post("/v1/completions", {"model": MODEL, "prompt": it["text"], "max_tokens": 1,
                                      "temperature": 0, "prompt_logprobs": 1, "return_token_ids": True})
        ids, lp, top1 = parse_prompt_logprobs(d["choices"][0])
        append_jsonl(out, {"id": it["id"], "kind": it["kind"], "bucket": it["bucket"], "n": len(ids),
                           "tok_sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest(), "lp": lp, "top1": top1})
        print("ppl %-14s %7d tokens  %.1fs  nll/token %.4f" % (it["id"], len(ids), time.time() - t0, -sum(lp) / len(lp)), flush=True)


def cmd_greedy(a):
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "greedy.jsonl")
    done = {r["id"] for r in load_jsonl(out)}
    for it in load_jsonl(os.path.join(a.prompts, "greedy.jsonl")):
        if it["id"] in done:
            continue
        d = post("/v1/completions", {"model": MODEL, "prompt": it["text"], "max_tokens": GREEDY_NEW_TOKENS, "temperature": 0,
                                      "return_token_ids": True})
        ch = d["choices"][0]
        append_jsonl(out, {"id": it["id"], "kind": it["kind"], "tokens": ch.get("token_ids") or [], "text": ch["text"]})
        print("greedy", it["id"], flush=True)


def cmd_mmlu(a):
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "mmlu.jsonl")
    done = {r["id"] for r in load_jsonl(out)}
    for it in load_jsonl(os.path.join(a.prompts, "mmlu.jsonl")):
        if it["id"] in done:
            continue
        d = post("/v1/completions", {"model": MODEL, "prompt": it["prompt"], "max_tokens": 1, "temperature": 0, "logprobs": 20})
        top = d["choices"][0]["logprobs"]["top_logprobs"][0]
        scores = {}
        for tok, lp in top.items():
            s = tok.strip().upper()
            if len(s) == 1 and s in "ABCD":
                scores[s] = max(scores.get(s, -1e9), lp)
        pred = max(scores, key=scores.get) if scores else None
        append_jsonl(out, {"id": it["id"], "subject": it["subject"], "answer": it["answer"], "pred": pred, "scores": scores})
    rows = load_jsonl(out)
    print("mmlu accuracy %.3f over %d (invalid %d)" % (sum(r["pred"] == r["answer"] for r in rows) / max(1, len(rows)), len(rows),
                                                       sum(r["pred"] is None for r in rows)))


# ----------------------------------------------------------------------------- compare
def ppl_rows(ref, cand):
    rows = []
    for i, ra in ref.items():
        rb = cand.get(i)
        if rb is None:
            continue
        if ra["tok_sha"] != rb["tok_sha"]:
            raise SystemExit("prompt %s was tokenised differently in the two runs - not comparable" % i)
        n = min(len(ra["lp"]), len(rb["lp"]))
        rows.append({"id": i, "kind": ra["kind"], "bucket": ra["bucket"], "n": n,
                     "nllA": -sum(ra["lp"][:n]), "nllB": -sum(rb["lp"][:n]),
                     "agree": sum(1 for x, y in zip(ra["top1"][:n], rb["top1"][:n]) if x == y),
                     "dabs": sum(abs(x - y) for x, y in zip(ra["lp"][:n], rb["lp"][:n]))})
    return rows


def summarize(rows, rng):
    if not rows:
        return None
    n = sum(r["n"] for r in rows)
    mA, mB = sum(r["nllA"] for r in rows) / n, sum(r["nllB"] for r in rows) / n
    out = {"docs": len(rows), "tokens": n, "ppl_ref": math.exp(mA), "ppl_cand": math.exp(mB), "ratio": math.exp(mB - mA),
           "top1_agree": sum(r["agree"] for r in rows) / n, "mean_abs_dlogp": sum(r["dabs"] for r in rows) / n, "ci": None}
    if len(rows) >= 4:
        boots = []
        for _ in range(2000):
            s = [rows[rng.randrange(len(rows))] for _ in rows]
            nn = sum(r["n"] for r in s)
            boots.append(math.exp(sum(r["nllB"] for r in s) / nn - sum(r["nllA"] for r in s) / nn))
        boots.sort()
        out["ci"] = [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]]
    return out


def groups(rows, rng):
    g = {"all": summarize(rows, rng)}
    for kind in ("code", "text"):
        g[kind] = summarize([r for r in rows if r["kind"] == kind], rng)
    for kind in ("code", "text"):
        for b in sorted({r["bucket"] for r in rows}):
            s = summarize([r for r in rows if r["kind"] == kind and r["bucket"] == b], rng)
            if s:
                g["%s@%dK" % (kind, b // 1024 if b >= 1024 else 0)] = s
    return g


def binom_two_sided(n, k):
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2 * p)


def mmlu_stats(ref, cand):
    ids = [i for i in ref if i in cand]
    if not ids:
        return None
    ok = lambda r: r["pred"] == r["answer"]
    a = sum(ok(ref[i]) for i in ids) / len(ids)
    b = sum(ok(cand[i]) for i in ids) / len(ids)
    only_ref = sum(1 for i in ids if ok(ref[i]) and not ok(cand[i]))
    only_cand = sum(1 for i in ids if ok(cand[i]) and not ok(ref[i]))
    return {"n": len(ids), "acc_ref": a, "acc_cand": b, "drop": a - b, "lost": only_ref, "gained": only_cand,
            "p_mcnemar": binom_two_sided(only_ref + only_cand, min(only_ref, only_cand)),
            "invalid_cand": sum(cand[i]["pred"] is None for i in ids) / len(ids)}


def greedy_stats(ref, cand):
    ids = [i for i in ref if i in cand and ref[i]["tokens"] and cand[i]["tokens"]]
    if not ids:
        if any(i in cand for i in ref):
            raise SystemExit("greedy records carry no token ids - re-run `greedy` (the request needs return_token_ids=true)")
        return None
    first = []
    for i in ids:
        x, y = ref[i]["tokens"], cand[i]["tokens"]
        k = 0
        while k < min(len(x), len(y)) and x[k] == y[k]:
            k += 1
        first.append(k if (k < min(len(x), len(y)) or len(x) != len(y)) else GREEDY_NEW_TOKENS)
    first.sort()
    return {"n": len(ids), "identical": sum(1 for f in first if f >= GREEDY_NEW_TOKENS) / len(ids), "median_first_divergence": first[len(first) // 2]}


def load_run(d):
    if not d:
        return None
    return {k: {r["id"]: r for r in load_jsonl(os.path.join(d, k + ".jsonl"))} for k in ("ppl", "greedy", "mmlu")}


def compare_runs(ref, cand, ref2=None, cand2=None, seed=11):
    rng = random.Random(seed)
    rep = {"thresholds": THRESHOLDS, "ppl": groups(ppl_rows(ref["ppl"], cand["ppl"]), rng) if ref["ppl"] and cand["ppl"] else None}
    rep["noise_ref"] = groups(ppl_rows(ref["ppl"], ref2["ppl"]), rng) if ref2 and ref2["ppl"] else None
    rep["noise_cand"] = groups(ppl_rows(cand["ppl"], cand2["ppl"]), rng) if cand2 and cand2["ppl"] else None
    rep["mmlu"] = mmlu_stats(ref["mmlu"], cand["mmlu"]) if ref["mmlu"] and cand["mmlu"] else None
    rep["greedy"] = greedy_stats(ref["greedy"], cand["greedy"]) if ref["greedy"] and cand["greedy"] else None
    rep["greedy_noise_ref"] = greedy_stats(ref["greedy"], ref2["greedy"]) if ref2 and ref2["greedy"] else None
    checks, T = [], THRESHOLDS
    if rep["ppl"]:
        for g in ("all", "code", "text"):
            s = rep["ppl"].get(g)
            if not s:
                continue
            checks.append(("ppl ratio %s <= %.3f" % (g, T["ppl_ratio_max"]), s["ratio"] <= T["ppl_ratio_max"], "%.4f" % s["ratio"]))
            if s["ci"]:
                checks.append(("ppl ratio %s CI upper <= %.3f" % (g, T["ppl_ci_upper_max"]), s["ci"][1] <= T["ppl_ci_upper_max"], "%.4f" % s["ci"][1]))
            checks.append(("top-1 agreement %s >= %.2f" % (g, T["top1_agree_min"]), s["top1_agree"] >= T["top1_agree_min"], "%.4f" % s["top1_agree"]))
            nz = (rep["noise_ref"] or {}).get(g)
            if nz:
                checks.append(("top-1 agreement %s within %.2f of ref-vs-ref noise (%.4f)" % (g, T["top1_vs_noise_max_gap"], nz["top1_agree"]),
                               s["top1_agree"] >= nz["top1_agree"] - T["top1_vs_noise_max_gap"], "%.4f" % s["top1_agree"]))
    if rep["mmlu"]:
        m = rep["mmlu"]
        checks.append(("MMLU drop <= %.2f" % T["mmlu_drop_max"], m["drop"] <= T["mmlu_drop_max"], "%+.3f" % m["drop"]))
        checks.append(("MMLU invalid <= %.2f" % T["mmlu_invalid_max"], m["invalid_cand"] <= T["mmlu_invalid_max"], "%.3f" % m["invalid_cand"]))
    complete = bool(rep["ppl"]) and bool(rep["mmlu"])
    rep["checks"] = [{"check": c, "pass": bool(p), "value": v} for c, p, v in checks]
    rep["verdict"] = ("PASS" if all(p for _, p, _ in checks) else "FAIL") if complete else "INCOMPLETE"
    return rep


def render(rep):
    L = ["# A/B report (reference = BF16 KV, candidate = FP8 KV)", ""]
    if rep["ppl"]:
        L += ["## Perplexity and top-1 agreement", "", "| group | docs | tokens | ppl ref | ppl cand | ratio | 95% CI | top-1 agree | mean abs dlogp |", "|---|---|---|---|---|---|---|---|---|"]
        for k, s in rep["ppl"].items():
            ci = "-" if not s["ci"] else "%.4f-%.4f" % tuple(s["ci"])
            L.append("| %s | %d | %d | %.4f | %.4f | %.4f | %s | %.4f | %.4f |" % (k, s["docs"], s["tokens"], s["ppl_ref"], s["ppl_cand"], s["ratio"], ci, s["top1_agree"], s["mean_abs_dlogp"]))
        for name, key in (("reference vs reference (run-to-run noise)", "noise_ref"), ("candidate vs candidate (run-to-run noise)", "noise_cand")):
            if rep[key]:
                s = rep[key]["all"]
                L += ["", "%s: ratio %.4f, top-1 agreement %.4f, mean abs dlogp %.4f" % (name, s["ratio"], s["top1_agree"], s["mean_abs_dlogp"])]
    if rep["mmlu"]:
        m = rep["mmlu"]
        L += ["", "## MMLU (0-shot, logprob of A-D)", "", "n=%d  ref %.3f  cand %.3f  drop %+.3f  lost %d / gained %d  McNemar p=%.3f  invalid %.3f" % (m["n"], m["acc_ref"], m["acc_cand"], m["drop"], m["lost"], m["gained"], m["p_mcnemar"], m["invalid_cand"])]
    if rep["greedy"]:
        g = rep["greedy"]
        L += ["", "## Greedy continuations (128 tokens, informational)", "", "n=%d  identical %.2f  median first divergence at token %d" % (g["n"], g["identical"], g["median_first_divergence"])]
        if rep["greedy_noise_ref"]:
            z = rep["greedy_noise_ref"]
            L.append("reference vs reference: identical %.2f  median first divergence %d" % (z["identical"], z["median_first_divergence"]))
    L += ["", "## Pre-registered checks", "", "| check | value | result |", "|---|---|---|"]
    for c in rep["checks"]:
        L.append("| %s | %s | %s |" % (c["check"], c["value"], "PASS" if c["pass"] else "FAIL"))
    L += ["", "**Verdict: %s**" % rep["verdict"]]
    return "\n".join(L)


def cmd_compare(a):
    rep = compare_runs(load_run(a.ref), load_run(a.cand), load_run(a.ref2), load_run(a.cand2))
    txt = render(rep)
    print(txt)
    if a.report:
        json.dump(rep, open(a.report + ".json", "w"), indent=1)
        open(a.report + ".md", "w").write(txt + "\n")


# ----------------------------------------------------------------------------- self-test (negative controls)
def synth(dirname, base=None, lp_shift=0.0, noise=0.0, flip=0.0, mmlu_acc=0.7, seed=1):
    rnd = random.Random(seed)
    os.makedirs(dirname, exist_ok=True)
    docs = [("code", 4096, 6), ("text", 4096, 6), ("code", 16384, 3), ("text", 16384, 3)]
    ppl, mm = [], []
    if base is None:
        for kind, bucket, nd in docs:
            for k in range(nd):
                n = 2000
                lp = [min(-0.001, rnd.gauss(-1.8, 1.2)) for _ in range(n)]
                ppl.append({"id": "%s-%d-%d" % (kind, bucket, k), "kind": kind, "bucket": bucket, "n": n + 1, "tok_sha": "x%s%d%d" % (kind, bucket, k),
                            "lp": lp, "top1": [rnd.randrange(1000) for _ in range(n)]})
        for i in range(300):
            ans = rnd.choice("ABCD")
            pred = ans if rnd.random() < mmlu_acc else rnd.choice("ABCD")
            mm.append({"id": "mmlu-%d" % i, "subject": "s", "answer": ans, "pred": pred, "scores": {}})
    else:
        for r in base["ppl"].values():
            lp = [min(-0.0001, x + lp_shift + rnd.gauss(0, noise)) for x in r["lp"]]
            top1 = [t if rnd.random() >= flip else -2 for t in r["top1"]]
            ppl.append(dict(r, lp=lp, top1=top1))
        for r in base["mmlu"].values():
            pred = r["pred"] if rnd.random() >= (1 - mmlu_acc) else rnd.choice("ABCD")
            mm.append(dict(r, pred=pred))
    for name, rows in (("ppl", ppl), ("mmlu", mm), ("greedy", [])):
        with open(os.path.join(dirname, name + ".jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def cmd_selftest(_):
    with tempfile.TemporaryDirectory() as t:
        synth(t + "/ref", seed=1)
        ref = load_run(t + "/ref")
        synth(t + "/ref2", base=ref, noise=0.002, flip=0.001, mmlu_acc=1.0, seed=2)
        synth(t + "/good", base=ref, noise=0.01, flip=0.004, mmlu_acc=0.995, seed=3)
        synth(t + "/bad_ppl", base=ref, lp_shift=-0.1, noise=0.01, flip=0.10, mmlu_acc=0.995, seed=4)
        synth(t + "/bad_mmlu", base=ref, noise=0.01, flip=0.004, mmlu_acc=0.88, seed=5)
        r2, good, bp, bm = (load_run(t + "/" + n) for n in ("ref2", "good", "bad_ppl", "bad_mmlu"))
        res = {n: compare_runs(ref, c, ref2=r2)["verdict"] for n, c in (("good", good), ("bad_ppl", bp), ("bad_mmlu", bm))}
    print("selftest verdicts:", res)
    ok = res["good"] == "PASS" and res["bad_ppl"] == "FAIL" and res["bad_mmlu"] == "FAIL"
    print("SELFTEST", "OK (good PASSes, both degraded candidates FAIL)" if ok else "BROKEN")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--out", required=True); p.add_argument("--root", required=True, help="site-packages dir holding vllm/ and *.dist-info")
    p.add_argument("--mmlu-jsonl"); p.add_argument("--mmlu-n", type=int, default=500); p.add_argument("--small", action="store_true"); p.set_defaults(fn=cmd_prep)
    for n, fn in (("ppl", cmd_ppl), ("greedy", cmd_greedy), ("mmlu", cmd_mmlu)):
        p = sub.add_parser(n); p.add_argument("--prompts", required=True); p.add_argument("--out", required=True); p.set_defaults(fn=fn)
    p = sub.add_parser("compare"); p.add_argument("--ref", required=True); p.add_argument("--cand", required=True)
    p.add_argument("--ref2"); p.add_argument("--cand2"); p.add_argument("--report"); p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("selftest"); p.set_defaults(fn=cmd_selftest)
    p = sub.add_parser("parsecheck"); p.add_argument("response"); p.set_defaults(fn=cmd_parsecheck)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
