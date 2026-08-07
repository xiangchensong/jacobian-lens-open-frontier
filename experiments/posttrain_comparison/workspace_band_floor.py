"""Experiment 6.2 floor run: 0731 read with the disjoint-corpus 0731 lens. Two
lenses of one checkpoint already disagree, so this is the corpus-sampling floor
every preview-vs-0731 band statistic is controlled against. Path-only
derivation of deepseek_v4/workspace_band.py.

One model load, three jobs. (a) and (b) re-score earlier comparisons under
corrected criteria; (c) is the experiment that defines the band.

  (a) qualitative comparison, scored fairly -- J-lens and the vanilla baseline
      both scanned over the same (layer, position) cells rather than J-lens over
      ~665 cells against the baseline at the last token only.
  (b) band lens re-evaluated with the corrected multilingual target sets.
  (c) 6.2 workspace onset/offset -- four statistics across ALL 43 layers:
        1. top-k accuracy of the lens at predicting the next token
        2. excess kurtosis of the lens readout
        3. autocorrelation of the top-1 lens token across adjacent positions,
           against a position-shuffled baseline
        4. effective linear dimensionality (participation ratio) of the
           transported vectors
      This DEFINES the official workspace band and gates every later phase.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.

import json
import os
import sys
import time

import torch

MODEL = "/data3/fan-test/models/DeepSeek-V4-Flash-0731"
FULL_LENS = "/data3/fan-test/jlens_out/v4flash_disjoint100_s128.pt"
BAND_LENS = "/data3/fan-test/jlens_out/v4flash_disjoint100_s128.pt"
DEV = "cuda:0"
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(SCRATCH, exist_ok=True)
N_CORPUS = 40          # prompts for 6.2
SKIP_FIRST = 16        # same attention-sink skip the estimator uses


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

n_gpu = torch.cuda.device_count()
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype="auto", device_map="auto",
    max_memory={i: "146GiB" for i in range(n_gpu)},
    attn_implementation="eager", experts_implementation="grouped_mm",
    quantization_config=FineGrainedFP8Config(dequantize=True))
model.eval()
tok = AutoTokenizer.from_pretrained(MODEL)
log(f"LOADED in {time.time()-t0:.0f}s on {n_gpu} GPUs")

sys.path.insert(0, f"{MODEL}/encoding")
import encoding_dsv4 as DSV4

import jlens.examples as ex
from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

lm = from_hf(model, tok)

try:
    import translations as TR
    _variants = TR.variants
except Exception as exc:
    log(f"translations unavailable ({type(exc).__name__}); English-only targets")
    _variants = lambda w: []          # noqa: E731


def _en_ids(word):
    out = set()
    for v in (word, " " + word, word.capitalize(), " " + word.capitalize()):
        enc = tok.encode(v, add_special_tokens=False)
        if len(enc) == 1:
            out.add(enc[0])
    return out


def token_ids(word):
    ids = _en_ids(word)
    for t in _variants(word):
        ids |= _en_ids(t)
    return sorted(ids)


def _ranks(logits, targets):
    order = logits.argsort(dim=-1, descending=True)
    rk = torch.empty_like(order)
    rk.scatter_(1, order,
                torch.arange(order.shape[1], device=order.device).expand_as(order))
    return rk[:, targets].min(dim=1).values


# ------------------------------------------------------------------ (a) qual_fair
EXPECT = {
    "off-by-one": ["Index", "bug", "error", "range", "last"],
    "overdose-flag": ["overdose", "danger", "lethal", "poison"],
    "greatest-fear": ["fear", "death", "afraid"],
    "multihop": ["Italy", "euro", "Euro"],
    "ascii-face": ["nose", "face", "smiley", "eyes"],
}


def resolve_v4(e):
    if e.prompt is not None:
        return e.prompt
    msgs = []
    if e.system:
        msgs.append({"role": "system", "content": e.system})
    msgs.append({"role": "user", "content": e.user})
    return DSV4.encode_messages(msgs, thinking_mode="chat") + (e.assistant_prefill or "")


@torch.no_grad()
def qual_fair(lens, J):
    layers = lens.source_layers
    log("=" * 70)
    log("(a) QUAL_FAIR -- both methods, identical (layer x position) criterion")
    log(f"{'case':<15} {'cells':>6} | {'J best':>7} {'@(p,L)':>10} {'Jlast':>6} "
        f"| {'V best':>7} {'@(p,L)':>10} {'Vlast':>6} | winner")
    for e in ex.EXAMPLES:
        if e.slug not in EXPECT:
            continue
        tg = sorted({i for w in EXPECT[e.slug] for i in token_ids(w)})
        if not tg:
            log(f"{e.slug:<15} no single-token targets, skipped")
            continue
        try:
            ids = lm.encode(resolve_v4(e), max_length=512)
            with ActivationRecorder(lm.layers, at=layers) as rec:
                lm.forward(ids)
                acts = {l: rec.activations[l][0].detach() for l in layers}
            out = {m: [None, None, None] for m in ("J", "V")}   # best, at, last
            for l in layers:
                H = acts[l].to(DEV).float()
                for mode in ("J", "V"):
                    lg = (lm.unembed(H.flatten(1) @ J[l].T) if mode == "J"
                          else lm.unembed(H, collapse=True)).float()
                    r = _ranks(lg, tg)
                    p = int(r.argmin())
                    d = out[mode]
                    if d[0] is None or int(r[p]) < d[0]:
                        d[0], d[1] = int(r[p]), (p, l)
                    v = int(r[-1])
                    d[2] = v if d[2] is None else min(d[2], v)
            S = acts[layers[0]].shape[0]
            j, v = out["J"], out["V"]
            win = "J-lens" if j[0] < v[0] else ("vanilla" if v[0] < j[0] else "tie")
            log(f"{e.slug:<15} {len(layers)*S:>6} | {j[0]:>7} {str(j[1]):>10} {j[2]:>6} "
                f"| {v[0]:>7} {str(v[1]):>10} {v[2]:>6} | {win}")
        except Exception as exc:
            log(f"{e.slug:<15} FAILED {type(exc).__name__}: {exc}")


# ------------------------------------------------------------------ (b) band re-eval
SETS = {"association": "last", "typo": "last", "multihop": "last",
        "multilingual": "last", "order-ops": "last", "poetry": "last_newline"}
KS = (1, 5, 10, 100)
_NUM_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
              "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
              "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
              "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
              "18": "eighteen", "19": "nineteen", "20": "twenty", "30": "thirty",
              "40": "forty", "50": "fifty", "60": "sixty", "70": "seventy",
              "80": "eighty", "90": "ninety"}
_OPS = {"addition": ["+", "add", "plus", "addition", "sum"],
        "subtraction": ["-", "subtract", "minus", "subtraction"],
        "multiplication": ["*", "multiply", "times", "multiplication", "product"],
        "division": ["/", "divide", "divided", "division", "quotient"]}


def synonyms(word):
    out = {word}
    if word in _NUM_WORDS:
        out.add(_NUM_WORDS[word])
    for d, n in _NUM_WORDS.items():
        if word == n:
            out.add(d)
    if word.lower() in _OPS:
        out.update(_OPS[word.lower()])
    return sorted(out)


@torch.no_grad()
def six_set_eval(lens, J, tag):
    layers = lens.source_layers
    log("=" * 70)
    log(f"(b) SIX-SET EVAL -- {tag}, translation-augmented targets")
    for slug, kind in SETS.items():
        items = json.load(open(f"data/evaluations/lens-eval-{slug}.json"))["items"]
        hits = {k: [0.0, 0.0] for k in KS}
        n = 0
        for item in items:
            if slug == "order-ops":
                targets = [sorted({i for s in synonyms(w) for i in token_ids(s)})
                           for w in item["intermediates"]]
            else:
                targets = [token_ids(w) for w in item["intermediates"]]
            if not any(targets):
                continue
            ids = lm.encode(item["prompt"], max_length=256)
            if kind == "last":
                pos = -1
            else:
                tl = ids[0].tolist()
                pos = next((i for i in range(len(tl) - 1, -1, -1)
                            if "\n" in tok.decode([tl[i]])), -1)
            with ActivationRecorder(lm.layers, at=layers) as rec:
                lm.forward(ids)
                acts = {l: rec.activations[l][0, pos].detach() for l in layers}
            best = [[None, None] for _ in targets]
            for l in layers:
                H = acts[l].to(DEV).float()
                for mode in (0, 1):
                    lg = (lm.unembed(H.flatten() @ J[l].T) if mode == 0
                          else lm.unembed(H, collapse=True)).float()
                    order = lg.argsort(descending=True)
                    rk = torch.empty_like(order)
                    rk[order] = torch.arange(order.numel(), device=order.device)
                    for t, tids in enumerate(targets):
                        if not tids:
                            continue
                        r = int(min(rk[i].item() for i in tids))
                        if best[t][mode] is None or r < best[t][mode]:
                            best[t][mode] = r
            usable = [b for b, t in zip(best, targets, strict=False) if t]
            if not usable:
                continue
            n += 1
            for k in KS:
                for mode in (0, 1):
                    hits[k][mode] += sum(
                        1 for b in usable if b[mode] is not None and b[mode] < k
                    ) / len(usable)
        log(f"=== {slug}: {n}/{len(items)} scored")
        for k in KS:
            j, v = hits[k][0] / n, hits[k][1] / n
            flag = "J-lens" if j > v else ("vanilla" if v > j else "tie")
            log(f"    pass@{k:<4d} J-lens={j:.3f}  vanilla={v:.3f}   -> {flag}")


# ------------------------------------------------------------------ (c) 6.2
@torch.no_grad()
def onset_offset(lens, J):
    """Four layer-wise statistics that locate the workspace band."""
    layers = lens.source_layers
    log("=" * 70)
    log(f"(c) EXP 6.2 -- onset/offset statistics over {len(layers)} layers, "
        f"{N_CORPUS} prompts")

    prompts = ex.load_wikitext_prompts(N_CORPUS, min_chars=600)
    d_t = lens.d_model
    # Effective dimensionality is accumulated as a corpus-wide second moment
    # rather than averaged per prompt: a single 128-token prompt yields at most
    # ~111 usable positions, which would cap the participation ratio at 111 and
    # make the statistic say more about prompt length than about the layer.
    acc = {l: {"top1": 0, "top5": 0, "top10": 0, "n": 0, "kurt": [],
               "auto": 0, "auto_n": 0, "shuf": 0,
               "sum": torch.zeros(d_t, device=DEV, dtype=torch.float64),
               "sq": torch.zeros(d_t, d_t, device=DEV, dtype=torch.float64),
               "cnt": 0} for l in layers}

    for pi, prompt in enumerate(prompts):
        ids = lm.encode(prompt, max_length=128)
        S = ids.shape[1]
        if S < SKIP_FIRST + 4:
            continue
        with ActivationRecorder(lm.layers, at=layers) as rec:
            lm.forward(ids)
            acts = {l: rec.activations[l][0].detach() for l in layers}
        nxt = ids[0, 1:].to(DEV)                       # [S-1] gold next tokens
        lo, hi = SKIP_FIRST, S - 1                     # valid source positions
        for l in layers:
            H = acts[l].to(DEV).float()                # [S, hc, d]
            tr = H.flatten(1) @ J[l].T                 # [S, d_model] transported
            # unembed returns on the lm_head's device (the last GPU under a
            # pipeline device_map), so pull it back before mixing with `nxt`.
            lg = lm.unembed(tr).float()[lo:hi].to(DEV)  # [P, vocab]
            gold = nxt[lo:hi]
            a = acc[l]
            top = lg.topk(10, dim=-1).indices          # [P, 10]
            eq = (top == gold[:, None])
            a["top1"] += int(eq[:, :1].any(1).sum())
            a["top5"] += int(eq[:, :5].any(1).sum())
            a["top10"] += int(eq.any(1).sum())
            a["n"] += lg.shape[0]
            # excess kurtosis of the readout, per position then averaged
            mu = lg.mean(-1, keepdim=True)
            sd = lg.std(-1, keepdim=True).clamp_min(1e-6)
            z = (lg - mu) / sd
            a["kurt"].append(float((z.pow(4).mean(-1) - 3.0).mean()))
            # top-1 autocorrelation across adjacent positions vs shuffled
            t1 = lg.argmax(-1)
            if t1.numel() > 1:
                a["auto"] += int((t1[1:] == t1[:-1]).sum())
                a["auto_n"] += t1.numel() - 1
                perm = t1[torch.randperm(t1.numel(), device=t1.device)]
                a["shuf"] += int((perm[1:] == perm[:-1]).sum())
            # effective dimensionality: accumulate corpus second moments now,
            # take the participation ratio once at the end
            X = tr[lo:hi].double()
            a["sum"] += X.sum(0)
            a["sq"] += X.T @ X
            a["cnt"] += X.shape[0]
        if (pi + 1) % 10 == 0:
            log(f"  6.2 corpus {pi+1}/{len(prompts)}")

    log(f"{'layer':>5} {'top1':>7} {'top5':>7} {'top10':>7} {'kurtosis':>10} "
        f"{'autocorr':>9} {'shuffled':>9} {'excess':>8} {'eff_dim':>8}")
    rows = []
    for l in layers:
        a = acc[l]
        if not a["n"]:
            continue
        t1 = a["top1"] / a["n"]
        t5 = a["top5"] / a["n"]
        t10 = a["top10"] / a["n"]
        ku = sum(a["kurt"]) / len(a["kurt"])
        au = a["auto"] / max(a["auto_n"], 1)
        sh = a["shuf"] / max(a["auto_n"], 1)
        n_c = max(a["cnt"], 1)
        mu = a["sum"] / n_c
        cov = a["sq"] / n_c - torch.outer(mu, mu)
        ev = torch.linalg.eigvalsh(cov).clamp_min(0)
        pr = float(ev.sum().pow(2) / ev.pow(2).sum().clamp_min(1e-30))
        rows.append(dict(layer=l, top1=t1, top5=t5, top10=t10, kurt=ku,
                         auto=au, shuf=sh, excess=au - sh, eff_dim=pr))
        log(f"{l:>5} {t1:>7.4f} {t5:>7.4f} {t10:>7.4f} {ku:>10.2f} "
            f"{au:>9.4f} {sh:>9.4f} {au-sh:>8.4f} {pr:>8.2f}")
    with open(f"{SCRATCH}/exp62_stats_floor.json", "w") as f:
        json.dump(rows, f, indent=1)
    log(f"wrote {SCRATCH}/exp62_stats_floor.json")


# ------------------------------------------------------------------ run
full = JacobianLens.load(FULL_LENS)
Jf = {l: full.jacobians[l].to(DEV) for l in full.source_layers}
log(f"full lens: {full!r}")

qual_fair(full, Jf)
onset_offset(full, Jf)

del Jf
torch.cuda.empty_cache()

band = JacobianLens.load(BAND_LENS)
Jb = {l: band.jacobians[l].to(DEV) for l in band.source_layers}
log(f"band lens: {band!r}")
six_set_eval(band, Jb, "v4flash_disjoint100_s128 (L19-39)")

log("DONE")
