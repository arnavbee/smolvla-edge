"""Run SmolVLA's vision tower in int8 at full 512 px and check the actions still hold up.

1. export vision_model + connector to ONNX (fp32)
2. calibrate on frames from two episodes not used for testing
3. quantize: dynamic (int8 weights) and static (int8 weights + activations, QDQ)
4. swap each version in for embed_image and compare actions on the test episodes
"""
import argparse, glob, json, os, time

import av
import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quant_pre_process, quantize_dynamic, quantize_static)
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="edge-inference/smolvla-so101-pick-orange")
ap.add_argument("--data", default="data/pick-orange")
ap.add_argument("--task", default="Grab orange and place into plate")
ap.add_argument("--every", type=int, default=50)
ap.add_argument("--calib-every", type=int, default=40)
ap.add_argument("--out", default="onnx")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = torch.device("mps")

policy = SmolVLAPolicy.from_pretrained(args.repo).to(dev).eval()
cfg = policy.config
pre, post = make_pre_post_processors(cfg, pretrained_path=args.repo,
                                     preprocessor_overrides={"device_processor": {"device": "mps"}},
                                     postprocessor_overrides={"device_processor": {"device": "cpu"}})
cams = [k.split(".")[-1] for k in cfg.image_features]
vlm = policy.model.vlm_with_expert
stock_embed_image = vlm.embed_image


def frames_at(path, idx):
    want, out = set(idx), {}
    with av.open(path) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = torch.from_numpy(f.to_ndarray(format="rgb24")).permute(2, 0, 1).float() / 255
            if i > max(idx):
                break
    return out


def samples(pattern, every):
    """Yield (obs, recorded 50-step actions, frame id) for every Nth frame."""
    for pqf in sorted(glob.glob(os.path.join(args.data, pattern))):
        ep = os.path.basename(pqf)[:-8]
        t = pq.read_table(pqf).to_pydict()
        state, action = np.array(t["observation.state"]), np.array(t["action"])
        idx = list(range(0, len(action) - cfg.chunk_size, every))
        imgs = {c: frames_at(os.path.join(args.data, f"{ep}_{c}.mp4"), idx) for c in cams}
        for i in idx:
            obs = {"observation.state": torch.tensor(state[i], dtype=torch.float32), "task": args.task}
            for c in cams:
                obs[f"observation.images.{c}"] = imgs[c][i]
            yield obs, action[i : i + cfg.chunk_size], f"{ep}:{i}"


# 1. export
class Tower(torch.nn.Module):
    """vision_model + connector, minus the attention-mask builder, which doesn't trace.
    With no padded patches that mask is all ones, so passing None gives the same result."""

    def __init__(self):
        super().__init__()
        self.vm = vlm.get_vlm_model().vision_model
        self.connector = vlm.get_vlm_model().connector

    def forward(self, pixels):
        # the stock embeddings build position ids through a bucketize/scatter that exports badly;
        # at full 512 px with no padding they are just 0..N-1
        e = self.vm.embeddings
        x = e.patch_embedding(pixels).flatten(2).transpose(1, 2)
        x = x + e.position_embedding(torch.arange(x.shape[1], device=pixels.device))[None]
        x = self.vm.encoder(inputs_embeds=x, attention_mask=None).last_hidden_state
        return self.connector(self.vm.post_layernorm(x))


fp32_path = os.path.join(args.out, "vision_fp32.onnx")
if not os.path.exists(fp32_path):
    policy.cpu()
    tower = Tower().eval()
    with torch.no_grad():
        x = torch.rand(1, 3, 512, 512) * 2 - 1
        print("tower vs stock max diff:", (tower(x) - stock_embed_image(x)).abs().max().item())
    torch.onnx.export(tower, torch.randn(1, 3, 512, 512), fp32_path, input_names=["pixels"],
                      output_names=["emb"], opset_version=17, dynamo=False)
    policy.to(dev)
print("exported", fp32_path, round(os.path.getsize(fp32_path) / 1e6), "MB")

# 2. calibration images: exactly what embed_image sees (resized, padded, scaled to [-1, 1])
calib = []
for obs, _, _ in samples("calib*.parquet", args.calib_every):
    with torch.no_grad():
        imgs, _ = policy.prepare_images(pre(dict(obs)))
    calib += [im.float().cpu().numpy() for im in imgs]
print("calibration images:", len(calib))


class Reader(CalibrationDataReader):
    def __init__(self):
        self.it = iter(calib)

    def get_next(self):
        x = next(self.it, None)
        return None if x is None else {"pixels": x}


# 3. quantize
prep_path = os.path.join(args.out, "vision_prep.onnx")
dyn_path = os.path.join(args.out, "vision_int8_dynamic.onnx")
sta_path = os.path.join(args.out, "vision_int8_static.onnx")
if not os.path.exists(prep_path):
    quant_pre_process(fp32_path, prep_path)
if not os.path.exists(dyn_path):
    quantize_dynamic(prep_path, dyn_path, weight_type=QuantType.QInt8, per_channel=True)
if not os.path.exists(sta_path):
    quantize_static(prep_path, sta_path, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
# static again, but only MatMul/Conv get int8; softmax, layernorm, GELU and adds stay float.
# (Percentile calibration would clip outliers better but keeps every activation in memory,
# which gets the process killed on an 8 GB machine.)
mm_path = os.path.join(args.out, "vision_int8_static_mm.onnx")
if not os.path.exists(mm_path):
    quantize_static(prep_path, mm_path, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=["MatMul", "Conv"])
variants = {"onnx_fp32": fp32_path, "int8_dynamic": dyn_path, "int8_static": sta_path, "int8_static_mm": mm_path}
sessions = {k: ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for k, p in variants.items()}

# CPU speed per image
x = calib[0]
speed = {}
for k, s in sessions.items():
    s.run(None, {"pixels": x})
    t = time.perf_counter()
    for _ in range(5):
        s.run(None, {"pixels": x})
    speed[k] = round((time.perf_counter() - t) / 5 * 1000, 1)
speed = {k: {"ms_per_image_cpu": v, "MB": round(os.path.getsize(variants[k]) / 1e6)} for k, v in speed.items()}
print(json.dumps(speed))


# 4. accuracy
def predict(obs, noise, sess=None):
    vlm.embed_image = stock_embed_image if sess is None else (
        lambda img: torch.from_numpy(sess.run(None, {"pixels": img.float().cpu().numpy()})[0]).to(img.device, img.dtype))
    batch = pre(dict(obs))
    with torch.no_grad():
        policy.reset()
        a = policy.predict_action_chunk(batch, noise=noise.clone())
    vlm.embed_image = stock_embed_image
    return post(a).squeeze(0).float().cpu().numpy()


rows = []
for obs, truth, fid in samples("ep*.parquet", args.every):
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=torch.Generator().manual_seed(len(rows))).to(dev)
    ref = predict(obs, noise)
    row = {"frame": fid, "err_torch": float(np.abs(ref - truth).mean())}
    for k, s in sessions.items():
        p = predict(obs, noise, s)
        row[f"err_{k}"] = float(np.abs(p - truth).mean())
        row[f"vs_torch_{k}"] = float(np.abs(p - ref).mean())
    rows.append(row)
    print(json.dumps(row), flush=True)

summary = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0] if k != "frame"}
summary["frames"] = len(rows)
print("SUMMARY", json.dumps({"accuracy": summary, "speed": speed}, indent=2))
json.dump({"accuracy": summary, "speed": speed, "rows": rows}, open("vision_int8.json", "w"), indent=1)
