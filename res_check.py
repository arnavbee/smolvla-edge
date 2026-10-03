"""How much does a smaller vision input change SmolVLA's actions?

Runs a fine-tuned SmolVLA (edge-inference/smolvla-so101-pick-orange) on recorded frames
from its dataset at several input sizes, with the same noise each time, and compares
the predicted 50-step chunks against the recorded actions and against the 512 px output.
"""
import argparse, glob, json, os

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="edge-inference/smolvla-so101-pick-orange")
ap.add_argument("--data", default="data/pick-orange")
ap.add_argument("--task", default="Grab orange and place into plate")
ap.add_argument("--res", default="512,384,256")
ap.add_argument("--every", type=int, default=40, help="sample one frame every N")
ap.add_argument("--device", default="mps")
args = ap.parse_args()
sizes = [int(r) for r in args.res.split(",")]
dev = torch.device(args.device)

policy = SmolVLAPolicy.from_pretrained(args.repo).to(dev).eval()
cfg = policy.config
ov = {"device_processor": {"device": args.device}}
pre, post = make_pre_post_processors(cfg, pretrained_path=args.repo,
                                     preprocessor_overrides=ov, postprocessor_overrides={"device_processor": {"device": "cpu"}})
cams = [k.split(".")[-1] for k in cfg.image_features]  # front, wrist


def frames_at(path, idx):
    want, out = set(idx), {}
    with av.open(path) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = torch.from_numpy(f.to_ndarray(format="rgb24")).permute(2, 0, 1).float() / 255
            if i > max(idx):
                break
    return out


def predict(obs, noise):
    batch = pre(dict(obs))
    with torch.no_grad():
        policy.reset()
        a = policy.predict_action_chunk(batch, noise=noise.clone())
    return post(a).squeeze(0).float().cpu().numpy()


rows = []
for pqf in sorted(glob.glob(os.path.join(args.data, "ep*.parquet"))):
    ep = os.path.basename(pqf)[:-8]
    t = pq.read_table(pqf).to_pydict()
    state, action = np.array(t["observation.state"]), np.array(t["action"])
    idx = list(range(0, len(action) - cfg.chunk_size, args.every))
    imgs = {c: frames_at(os.path.join(args.data, f"{ep}_{c}.mp4"), idx) for c in cams}
    for i in idx:
        obs = {"observation.state": torch.tensor(state[i], dtype=torch.float32), "task": args.task}
        for c in cams:
            obs[f"observation.images.{c}"] = imgs[c][i]
        g = torch.Generator().manual_seed(i)
        noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=g).to(dev)
        noise2 = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=g).to(dev)
        truth = action[i : i + cfg.chunk_size]
        preds = {}
        for r in sizes:
            cfg.resize_imgs_with_padding = (r, r)
            preds[r] = predict(obs, noise)
        cfg.resize_imgs_with_padding = (sizes[0], sizes[0])
        reseed = predict(obs, noise2)  # same input, different noise: the model's own spread
        row = {"ep": ep, "frame": i, "reseed_vs_ref": float(np.abs(reseed - preds[sizes[0]]).mean())}
        for r in sizes:
            row[f"err_{r}"] = float(np.abs(preds[r] - truth).mean())
            row[f"vs_ref_{r}"] = float(np.abs(preds[r] - preds[sizes[0]]).mean())
        rows.append(row)
        print(json.dumps(row), flush=True)

summary = {"frames": len(rows), "reseed_vs_ref": float(np.mean([r["reseed_vs_ref"] for r in rows]))}
for r in sizes:
    summary[f"err_{r}"] = float(np.mean([x[f"err_{r}"] for x in rows]))
    summary[f"vs_ref_{r}"] = float(np.mean([x[f"vs_ref_{r}"] for x in rows]))
print("SUMMARY", json.dumps(summary, indent=2))
json.dump({"summary": summary, "rows": rows}, open("res_check.json", "w"), indent=1)
