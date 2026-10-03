"""Time one SmolVLA action chunk, split into vision, VLM prefix and denoise steps."""
import argparse, json, platform, statistics, time

import torch
from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="mps")
ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
ap.add_argument("--runs", type=int, default=10)
ap.add_argument("--cams", type=int, default=None, help="feed only the first N cameras")
ap.add_argument("--batch-vision", action="store_true", help="encode all cameras in one pass")
ap.add_argument("--check", action="store_true", help="compare actions against the unpatched model")
ap.add_argument("--res", type=int, default=None, help="vision input size (model default 512)")
ap.add_argument("--steps", type=int, default=None, help="override num_steps")
args = ap.parse_args()

dev, dt = torch.device(args.device), getattr(torch, args.dtype)
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(dev, dt).eval()
cfg = policy.config
# Workaround: denoise_step casts the expert output to float32 (modeling_smolvla.py:808) before
# action_out_proj, and euler_integrate keeps x_t in float32, so the action-side projections must stay fp32.
if dt != torch.float32:
    for name in ["action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"]:
        getattr(policy.model, name).float()
if args.steps:
    cfg.num_steps = args.steps
if args.res:
    cfg.resize_imgs_with_padding = (args.res, args.res)

if args.check:
    from fast import batch_vision
    import copy
    torch.manual_seed(0)
    tok_ = AutoTokenizer.from_pretrained(cfg.vlm_model_name)
    e = tok_(["pick up the red cube and put it in the bowl\n"], padding="max_length",
             max_length=cfg.tokenizer_max_length, truncation=True, return_tensors="pt")
    b = {"observation.language.tokens": e["input_ids"].to(dev),
         "observation.language.attention_mask": e["attention_mask"].bool().to(dev),
         "observation.state": torch.randn(1, cfg.robot_state_feature.shape[0], device=dev, dtype=dt)}
    for key, ft in list(cfg.image_features.items())[: args.cams]:
        b[key] = torch.rand(1, *ft.shape, device=dev, dtype=dt)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device=dev)
    policy.reset(); ref = policy.predict_action_chunk(dict(b), noise=noise.clone())
    batch_vision(policy)
    policy.reset(); new = policy.predict_action_chunk(dict(b), noise=noise.clone())
    print(json.dumps({"max_abs_diff": (ref - new).abs().max().item(), "ref_abs_mean": ref.abs().mean().item()}))
    raise SystemExit

if args.batch_vision:
    from fast import batch_vision
    batch_vision(policy)

def sync():
    if dev.type == "mps":
        torch.mps.synchronize()

# wrap the stages so each one gets timed
timings = {"vision": [], "prefix": [], "denoise": []}
orig_fwd, orig_denoise = policy.model.vlm_with_expert.forward, policy.model.denoise_step
def timed_fwd(*a, **k):
    if k.get("past_key_values") is None:
        sync(); t = time.perf_counter(); out = orig_fwd(*a, **k); sync()
        timings["prefix"].append(time.perf_counter() - t); return out
    return orig_fwd(*a, **k)
def timed_denoise(*a, **k):
    sync(); t = time.perf_counter(); out = orig_denoise(*a, **k); sync()
    timings["denoise"].append(time.perf_counter() - t); return out
orig_embed = policy.model.embed_prefix
def timed_embed(*a, **k):
    sync(); t = time.perf_counter(); out = orig_embed(*a, **k); sync()
    timings["vision"].append(time.perf_counter() - t); return out
policy.model.embed_prefix = timed_embed
policy.model.vlm_with_expert.forward = timed_fwd
policy.model.denoise_step = timed_denoise

tok = AutoTokenizer.from_pretrained(cfg.vlm_model_name)
enc = tok(["pick up the red cube and put it in the bowl\n"], padding="max_length",
          max_length=cfg.tokenizer_max_length, truncation=True, return_tensors="pt")

batch = {
    "observation.language.tokens": enc["input_ids"].to(dev),
    "observation.language.attention_mask": enc["attention_mask"].bool().to(dev),
    "observation.state": torch.randn(1, cfg.robot_state_feature.shape[0], device=dev, dtype=dt),
}
print({k: v.shape for k, v in cfg.image_features.items()})
for key, ft in list(cfg.image_features.items())[: args.cams]:
    batch[key] = torch.rand(1, *ft.shape, device=dev, dtype=dt)

def chunk():
    policy.reset()
    sync(); t = time.perf_counter()
    a = policy.predict_action_chunk(dict(batch), noise=torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device=dev))
    sync(); return time.perf_counter() - t, a

for _ in range(2):  # warm-up
    chunk()
for v in timings.values():
    v.clear()
totals = []
for _ in range(args.runs):
    t, a = chunk(); totals.append(t)

ms = lambda xs: round(statistics.median(xs) * 1000, 1)
res = {
    "machine": platform.machine(), "device": args.device, "dtype": args.dtype,
    "batch_vision": args.batch_vision, "cameras_fed": args.cams or len(cfg.image_features), "image_size": cfg.resize_imgs_with_padding,
    "chunk_size": cfg.chunk_size, "num_steps": cfg.num_steps, "action_shape": list(a.shape),
    "chunk_ms_median": ms(totals), "vision_embed_ms_median": ms(timings["vision"]), "prefix_ms_median": ms(timings["prefix"]),
    "denoise_step_ms_median": ms(timings["denoise"]),
    "params_M": round(sum(p.numel() for p in policy.parameters()) / 1e6, 1),
}
res["chunks_per_s"] = round(1000 / res["chunk_ms_median"], 2)
print(json.dumps(res, indent=2))
