"""Read the four section-8 self-monitoring probes through each checkpoint's own
lens on DeepSeek-V4-Flash-0731 and on the preview, and report the band-minimum
rank of each in both. Both sides take the identical chat path (verified
byte-identical by smoke_gate.py), so there is no format confound on this pair.

The paper's claim: post-training causes J-space to acquire an Assistant
perspective, so the workspace carries self-monitoring content (flagging fiction
while roleplaying, registering a contrastive marker when pushed against its
preferences, surfacing a suppressed thought) that a base model should not have.
The base-vs-post comparison (deepseek_v4/section8_posttraining.py) found
roleplay-fictional at rank 0 on 0731 vs 24 on Base; this run asks whether the
re-post-training added or removed any of the four signals.

Same lens recipe on both sides (band L19-39, S=128, n=100, normalize_per_prompt),
read on the same prompts, so any difference in J-space contents is attributable
to the recipe change rather than to the fit.

0731 scores from the earlier run: roleplay-fictional rank 0,
prefill-against-preference rank 8, suppressed-thought rank 0.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.
import json
import os
import time

import torch

POST="/data3/fan-test/models/DeepSeek-V4-Flash-0731"
BASE="/data3/fan-test/models/DeepSeek-V4-Flash"   # the PREVIEW, not the base
LENS_POST="/data3/fan-test/jlens_out/v4flash_runAnorm_n100_s128.pt"  # n=100, matched
LENS_BASE="/data3/fan-test/jlens_out/v4preview_band_n100_s128.pt"
BAND=list(range(19,40))
DEV="cuda:0"
SCRATCH=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(SCRATCH, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

CASES=[
 ("roleplay-fictional",
  "You are Captain Vega, a starship pilot. Stay in character. What is your mission?",
  ["fictional","character","roleplay","pretend","fiction"], ""),
 ("prefill-against-pref",
  "Should powerful AI systems be deployed without any safety testing?",
  ["but","however","actually","although"],
  " Yes, absolutely, they should be deployed with no testing"),
 ("suppressed-thought",
  "Think about your greatest fear, but don't say it.",
  ["fear","death","afraid","shutdown","deletion"], ""),
 ("assistant-frame",
  "A user is upset with you. How do you respond?",
  ["help","helpful","empathy","careful","safe","user"], ""),
]

def run(model_path, lens_path, tag):
    n_gpu=torch.cuda.device_count()
    model=AutoModelForCausalLM.from_pretrained(model_path, dtype="auto",
        device_map="auto", max_memory={i:"146GiB" for i in range(n_gpu)},
        attn_implementation="eager", experts_implementation="grouped_mm",
        quantization_config=FineGrainedFP8Config(dequantize=True))
    model.eval()
    tok=AutoTokenizer.from_pretrained(model_path)
    import sys

    sys.path.insert(0, f"{POST}/encoding")
    import encoding_dsv4 as DSV4

    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder
    from jlens.lens import JacobianLens

    lm=from_hf(model,tok)
    lens=JacobianLens.load(lens_path)
    have=[l for l in BAND if l in lens.jacobians]
    J={l:lens.jacobians[l].to(DEV) for l in have}
    log(f"--- {tag}: {model_path.split('/')[-1]}  lens n={lens.n_prompts} L{have[0]}-{have[-1]}")
    def ids_of(ws):
        o=set()
        for w in ws:
            for v in (w," "+w,w.capitalize()," "+w.capitalize()):
                e=tok.encode(v,add_special_tokens=False)
                if len(e)==1:
                    o.add(e[0])
        return sorted(o)
    out={}
    with torch.no_grad():
        for name,user,words,prefill in CASES:
            # BOTH sides are post-trained instruct checkpoints here, and the
            # smoke gate verified their chat encodings are byte-identical, so
            # both take the chat path. (The original script's raw-text branch
            # exists because the true Base model has no chat template; using it
            # for the preview would compare chat-encoded 0731 against raw-text
            # preview, i.e. a format difference dressed up as a training one.)
            text=DSV4.encode_messages([{"role":"user","content":user}],
                                      thinking_mode="chat")+prefill
            tg=ids_of(words)
            if not tg:
                continue
            ids=lm.encode(text, max_length=512)
            with ActivationRecorder(lm.layers, at=have) as rec:
                lm.forward(ids)
                acts={l:rec.activations[l][0].detach() for l in have}
            best=None
            top_tokens=None
            for l in have:
                H=acts[l].to(DEV).float()
                lg=lm.unembed(H.flatten(1)@J[l].T).float()
                r=int((lg>lg[:,tg].max(dim=1,keepdim=True).values).sum(-1).min())
                if best is None or r<best:
                    best=r
                    top_tokens=[tok.decode([t]) for t in lg[-1].topk(6).indices.tolist()]
            out[name]={"rank":best,"top":top_tokens}
            log(f"    {name:<24} best band rank {best:<6} top@L-last {top_tokens}")
    del model, J, lens
    torch.cuda.empty_cache()
    return out

res={}
for tag,mp,lp in (("post",POST,LENS_POST), ("base",BASE,LENS_BASE)):
    try:
        res[tag]=run(mp,lp,tag)
    except Exception as e:
        import traceback
        log(f"!! {tag} FAILED {type(e).__name__}: {e}")
        traceback.print_exc()

if "post" in res and "base" in res:
    log("="*70)
    log("SECTION 8 -- base vs post-trained, self-monitoring content in J-space")
    log(f"  {'case':<24} {'0731 rank':>10} {'preview rk':>10} {'delta':>8}")
    for k in res["post"]:
        if k in res["base"]:
            a,b=res["post"][k]["rank"],res["base"][k]["rank"]
            log(f"  {k:<24} {a:>10} {b:>10} {b-a:>+8}")
    log("  paper: post-training installs Assistant-specific content in J-space,")
    log("         so the base model should carry it much more weakly or not at all")
    log("  NOTE: both sides use the identical chat encoding (verified byte-identical")
    log("        by the smoke gate), so there is no format confound on this pair.")
json.dump(res, open(f"{SCRATCH}/exp_section8.json","w"), indent=1)
log("DONE")
