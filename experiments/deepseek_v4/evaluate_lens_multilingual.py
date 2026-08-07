"""J-lens vs mHC-vanilla across all six lens-quality eval sets, with each target
concept expanded into its non-English forms via `translations.py`.

This is the augmented run that produced the J-lens column of the
method-comparison table. It differs from `evaluate_lens.py` only in `token_ids`:
the lens surfaces concepts in whichever language the model represents them in
(32.9% of J-lens's top-10 is non-English against 3.9% for vanilla), so
English-only targets systematically understate it.

Usage: evaluate_lens_multilingual.py <lens.pt> [<lens2.pt> ...]

Protocol per data/evaluations/README.md. Readout is a single token position,
which differs by set:
  association / typo          -- final prompt token
  multihop / multilingual /
  order-ops                   -- token immediately preceding `target`, i.e. the
                                 final prompt token (target is not in the prompt)
  poetry                      -- the last newline token (end of couplet line 1)

Metric: pass@k = mean over items of the fraction of `intermediates` whose
min-over-layers lens rank <= k.
"""

import json
import sys
import time

import torch

MODEL = "/data3/fan-test/models/DeepSeek-V4-Flash-0731"
DEV = "cuda:0"
SETS = {
    "association": "last",
    "typo": "last",
    "multihop": "last",
    "multilingual": "last",
    "order-ops": "last",
    "poetry": "last_newline",
}
KS = (1, 5, 10, 100)

_NUM_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
    "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
    "19": "nineteen", "20": "twenty", "30": "thirty", "40": "forty",
    "50": "fifty", "60": "sixty", "70": "seventy", "80": "eighty", "90": "ninety",
}
_OPS = {
    "addition": ["+", "add", "plus", "addition", "sum"],
    "subtraction": ["-", "subtract", "minus", "subtraction"],
    "multiplication": ["*", "multiply", "times", "multiplication", "product"],
    "division": ["/", "divide", "divided", "division", "quotient"],
}


def synonyms(word: str) -> list[str]:
    """Expand an order-ops intermediate key into its synonym set (README §order-ops)."""
    out = {word}
    if word in _NUM_WORDS:
        out.add(_NUM_WORDS[word])
    for digit, name in _NUM_WORDS.items():
        if word == name:
            out.add(digit)
    if word.lower() in _OPS:
        out.update(_OPS[word.lower()])
    return sorted(out)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

n_gpu = torch.cuda.device_count()
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype="auto", device_map="auto",
    max_memory={i: "146GiB" for i in range(n_gpu)},
    attn_implementation="eager", experts_implementation="grouped_mm",
    quantization_config=FineGrainedFP8Config(dequantize=True),
)
model.eval()
tok = AutoTokenizer.from_pretrained(MODEL)
log(f"LOADED in {time.time()-t0:.0f}s")

from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

lens_model = from_hf(model, tok)
VOCAB = model.config.vocab_size


import translations as TR


def _en_ids(word):
    ids = set()
    for v in (word, " " + word, word.capitalize(), " " + word.capitalize()):
        enc = tok.encode(v, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return ids


def token_ids(word: str) -> list[int]:
    """English forms PLUS non-English forms of the same concept.

    The lens surfaces concepts in whichever language the model represents them
    in (32.9% of J-lens's top-10 is non-English vs 3.9% for vanilla), so
    English-only targets systematically understate it.
    """
    ids = _en_ids(word)
    for t in TR.variants(word):
        ids |= _en_ids(t)
    return sorted(ids)


def readout_index(ids: torch.Tensor, kind: str) -> int:
    if kind == "last":
        return -1
    # last token whose decoded text contains a newline
    toks = ids[0].tolist()
    for i in range(len(toks) - 1, -1, -1):
        if "\n" in tok.decode([toks[i]]):
            return i
    return -1  # fall back rather than drop the item


@torch.no_grad()
def min_ranks(prompt, targets, kind, layers, J):
    ids = lens_model.encode(prompt, max_length=256)
    pos = readout_index(ids, kind)
    with ActivationRecorder(lens_model.layers, at=layers) as rec:
        lens_model.forward(ids)
        acts = {l: rec.activations[l][0, pos].detach() for l in layers}
    best = [[None, None] for _ in targets]
    for l in layers:
        streams = acts[l].to(DEV).float()               # [hc_mult, d_model]
        for mode in (0, 1):
            if mode == 0:
                logits = lens_model.unembed(streams.flatten() @ J[l].T).float()
            else:
                logits = lens_model.unembed(streams, collapse=True).float()
            order = logits.argsort(descending=True)
            rank_of = torch.empty_like(order)
            rank_of[order] = torch.arange(order.numel(), device=order.device)
            for t, tids in enumerate(targets):
                if not tids:
                    continue
                r = int(min(rank_of[i].item() for i in tids))
                if best[t][mode] is None or r < best[t][mode]:
                    best[t][mode] = r
    return best


for lens_path in sys.argv[1:]:
    lens = JacobianLens.load(lens_path)
    layers = lens.source_layers
    J = {l: lens.jacobians[l].to(DEV) for l in layers}
    log(f"##### {lens_path}")
    log(f"##### {lens!r}")
    for slug, kind in SETS.items():
        items = json.load(open(f"data/evaluations/lens-eval-{slug}.json"))["items"]
        hits = {k: [0.0, 0.0] for k in KS}
        n_scored = 0
        t0 = time.time()
        for item in items:
            if slug == "order-ops":
                targets = [sorted({i for s in synonyms(w) for i in token_ids(s)})
                           for w in item["intermediates"]]
            else:
                targets = [token_ids(w) for w in item["intermediates"]]
            if not any(targets):
                continue
            best = min_ranks(item["prompt"], targets, kind, layers, J)
            usable = [b for b, t in zip(best, targets, strict=False) if t]
            if not usable:
                continue
            n_scored += 1
            for k in KS:
                for mode in (0, 1):
                    hits[k][mode] += sum(
                        1 for b in usable if b[mode] is not None and b[mode] < k
                    ) / len(usable)
        log(f"=== {slug}: {n_scored}/{len(items)} scored in {time.time()-t0:.0f}s")
        for k in KS:
            j, v = hits[k][0] / n_scored, hits[k][1] / n_scored
            flag = "J-lens" if j > v else ("vanilla" if v > j else "tie")
            log(f"    pass@{k:<4d} J-lens={j:.3f}  vanilla={v:.3f}   -> {flag}")
    del J
    torch.cuda.empty_cache()

log("DONE")
