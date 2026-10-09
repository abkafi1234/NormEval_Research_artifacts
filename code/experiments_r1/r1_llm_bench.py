"""Timing check for the LLM lemmatizer: tokens generated per unit and time per unit, with and without
stopping at the first newline, at several batch sizes. Outputs are compared with the cached run."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
import torch
import r1_norm as N

d = json.load(open(os.path.join(N.NORM, "llm", "flores__fin_Latn.json")))
texts = d["raw"][:96]
ref = dict(zip(d["raw"], d["norm"]))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
tok = AutoTokenizer.from_pretrained(N.LLM_MODEL); tok.padding_side = "left"
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(N.LLM_MODEL, quantization_config=q, device_map={"": 0}, dtype=torch.float16); model.eval()
print("dtype", next(p for n,p in model.named_parameters() if "norm" in n).dtype)
nl = [i for i in range(len(tok)) if False]
ids = tok.encode("\n", add_special_tokens=False)
print("newline ids", ids, "eos", tok.eos_token_id, model.generation_config.eos_token_id)
block = N.examples_block("fin_Latn")
prompts = [N.INSTR.format(lang="Finnish", text=t, examples=block) for t in texts]
chats = [tok.apply_chat_template([{"role": "system", "content": N.SYSTEM}, {"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in prompts]
print("prompt tokens", len(tok(chats[0])["input_ids"]))
print("use_cache cfg:", model.config.use_cache, model.generation_config.use_cache, getattr(model.generation_config, "cache_implementation", None))
for bs, stop_nl in ((12, None), (24, None)):
    torch.cuda.reset_peak_memory_stats(); t0 = time.time(); gen_tokens = 0; same = 0; out = {}
    try:
        for s in range(0, len(chats), bs):
            enc = tok(chats[s:s + bs], return_tensors="pt", padding=True).to(model.device)
            mx = int(min(400, 24 + 4.0 * max(len(t.split()) for t in texts[s:s + bs])))
            eos = model.generation_config.eos_token_id
            eos = (eos if isinstance(eos, list) else [eos]) 
            with torch.inference_mode():
                g = model.generate(**enc, max_new_tokens=mx, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id, eos_token_id=eos, use_cache=True)
            gen_tokens += g.shape[1] - enc["input_ids"].shape[1]
            for t, x in zip(texts[s:s + bs], g):
                out[t] = tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip()
        dt = time.time() - t0
        same = sum(out[t] == ref[t] for t in texts if len(N.SENT_SPLIT.split(t.strip())) == 1)
        print(f"batch={bs} stop_newline={stop_nl}: {dt/len(texts):.3f} s/unit, steps/s {gen_tokens/dt:.1f}, peak mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, identical to cached: {same}", flush=True)
    except torch.OutOfMemoryError:
        print(f"batch={bs}: OOM", flush=True); torch.cuda.empty_cache()
