"""Which activations in the vision tower can't take int8?

Full static int8 (weights + activations) broke the actions (12.4 vs 1.2). Weight-only held.
So the damage is in the activations. This finds where.

1. build the full static QDQ model once (same settings as vision_int8.py)
2. every activation has a QuantizeLinear -> DequantizeLinear pair; dropping a pair leaves that
   tensor in float. Weights stay int8 throughout. No recalibration needed.
3. sweeps, scored by relative error of the tower output vs fp32 on held-out images:
   - per layer: only layer i's activations int8 (rest float), and all int8 except layer i
   - per op type: all int8 except activations feeding one op type
   - per tensor: only that one activation int8, ranked
4. keep the top-k most sensitive tensors in float, rerun the action check on a few k
"""
import argparse, collections, glob, json, os, re, time

import av
import numpy as np
import onnx
import onnxruntime as ort
import pyarrow.parquet as pq
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quant_pre_process, quantize_static
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="edge-inference/smolvla-so101-pick-orange")
ap.add_argument("--data", default="data/pick-orange")
ap.add_argument("--task", default="Grab orange and place into plate")
ap.add_argument("--every", type=int, default=50)
ap.add_argument("--calib-every", type=int, default=40)
ap.add_argument("--n-test", type=int, default=4, help="held-out images for the embedding sweeps")
ap.add_argument("--ks", default="0,1,2,4,8", help="top-k sensitive roles kept float for the action check")
ap.add_argument("--out", default="onnx")
args = ap.parse_args()
dev = torch.device("mps")
res_path = "layer_sens.json"
res = json.load(open(res_path)) if os.path.exists(res_path) else {}


def save():
    json.dump(res, open(res_path, "w"), indent=1)


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


def tower_inputs(pattern, every):
    out = []
    for obs, _, _ in samples(pattern, every):
        with torch.no_grad():
            imgs, _ = policy.prepare_images(pre(dict(obs)))
        out += [im.float().cpu().numpy() for im in imgs]
    return out


# 1. full static QDQ model (vision_fp32.onnx comes from vision_int8.py)
fp32_path = os.path.join(args.out, "vision_fp32.onnx")
prep_path = os.path.join(args.out, "vision_prep.onnx")
sta_path = os.path.join(args.out, "vision_int8_static.onnx")
if not os.path.exists(prep_path):
    quant_pre_process(fp32_path, prep_path)
if not os.path.exists(sta_path):
    calib = tower_inputs("calib*.parquet", args.calib_every)
    print("calibration images:", len(calib), flush=True)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib)

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"pixels": x}

    quantize_static(prep_path, sta_path, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    del calib

# held-out images: spread across the test episodes
test_all = tower_inputs("ep*.parquet", args.every)
test = [test_all[i] for i in np.linspace(0, len(test_all) - 1, args.n_test).astype(int)]
sess = lambda m: ort.InferenceSession(m.SerializeToString() if isinstance(m, onnx.ModelProto) else m,
                                      providers=["CPUExecutionProvider"])
s = sess(fp32_path)
ref = [s.run(None, {"pixels": x})[0] for x in test]
del s

# 2. activation Q/DQ pairs
qdq = onnx.load(sta_path)
inits = {i.name for i in qdq.graph.initializer}
producer = {o: n for n in qdq.graph.node for o in n.output}
consumers = collections.defaultdict(list)
for n in qdq.graph.node:
    for i in n.input:
        consumers[i].append(n)

pairs = {}  # float tensor -> (Q node, [DQ nodes], tag)
for n in qdq.graph.node:
    if n.op_type != "QuantizeLinear" or n.input[0] in inits:
        continue
    dqs = [c for c in consumers[n.output[0]] if c.op_type == "DequantizeLinear"]
    p = producer.get(n.input[0])
    users = [u for d in dqs for u in consumers[d.output[0]]]
    where = (p.name if p is not None else n.input[0])
    m = re.search(r"layers\.(\d+)", where) or re.search(r"layers\.(\d+)", users[0].name if users else "")
    block = f"L{int(m.group(1)):02d}" if m else ("connector" if "connector" in where else "embed")
    role = re.sub(r"layers\.\d+", "layers.N", where)
    pairs[n.input[0]] = dict(q=n, dqs=dqs, block=block, role=role, producer=p.op_type if p is not None else "input",
                             users=sorted({u.op_type for u in users}))
print("activation Q/DQ pairs:", len(pairs), flush=True)


def variant(int8_tensors):
    """Copy of the QDQ model where only `int8_tensors` keep their activation Q/DQ."""
    m = onnx.ModelProto()
    m.CopyFrom(qdq)
    drop_nodes, rename = set(), {}
    for t, p in pairs.items():
        if t in int8_tensors:
            continue
        drop_nodes.add(p["q"].name)
        for d in p["dqs"]:
            drop_nodes.add(d.name)
            rename[d.output[0]] = t
    keep = [n for n in m.graph.node if n.name not in drop_nodes]
    for n in keep:
        for j, i in enumerate(n.input):
            if i in rename:
                n.input[j] = rename[i]
    for o in m.graph.output:
        if o.name in rename:
            o.name = rename[o.name]
    del m.graph.node[:]
    m.graph.node.extend(keep)
    return m


def emb_err(model):
    s = sess(model)
    e = [np.linalg.norm(s.run(None, {"pixels": x})[0] - r) / np.linalg.norm(r) for x, r in zip(test, ref)]
    return round(float(np.mean(e)), 5)


everything = set(pairs)
t0 = time.time()
if "base" not in res:
    res["base"] = {"weight_only": emb_err(variant(set())), "full_static": emb_err(variant(everything)),
                   "pairs": len(pairs), "n_test": len(test)}
    save()
    print("base", res["base"], f"{time.time() - t0:.0f}s", flush=True)

# 3a. per block
blocks = sorted({p["block"] for p in pairs.values()})
res.setdefault("block_only", {})
res.setdefault("block_float", {})
for b in blocks:
    mine = {t for t, p in pairs.items() if p["block"] == b}
    if b not in res["block_only"]:
        res["block_only"][b] = emb_err(variant(mine))
        res["block_float"][b] = emb_err(variant(everything - mine))
        save()
        print(b, len(mine), "only:", res["block_only"][b], "float:", res["block_float"][b], flush=True)

# 3b. per op type the activation feeds
res.setdefault("op_float", {})
user_ops = sorted({u for p in pairs.values() for u in p["users"]})
for op in user_ops:
    if op not in res["op_float"]:
        mine = {t for t, p in pairs.items() if op in p["users"]}
        res["op_float"][op] = {"n": len(mine), "err": emb_err(variant(everything - mine))}
        save()
        print("float into", op, res["op_float"][op], flush=True)

# 3c. per role (same tensor position in every layer, e.g. the fc2 input), two ways:
#     all int8 except this role, and only this role int8
roles = sorted({p["role"] for p in pairs.values()})
print("roles:", len(roles), flush=True)
res.setdefault("role", {})
for r in roles:
    if r not in res["role"]:
        mine = {t for t, p in pairs.items() if p["role"] == r}
        p0 = next(p for p in pairs.values() if p["role"] == r)
        res["role"][r] = {"n": len(mine), "producer": p0["producer"], "users": p0["users"],
                          "only": emb_err(variant(mine)), "float": emb_err(variant(everything - mine))}
        save()
        print("role", r, res["role"][r], f"{time.time() - t0:.0f}s", flush=True)

# greedy: float the most damaging roles first (by "only" error), see how fast it recovers
rank_roles = sorted(res["role"], key=lambda r: -res["role"][r]["only"])
res["role_rank"] = rank_roles
res.setdefault("topk_float", {})
def float_top(k):
    return {t for t, p in pairs.items() if p["role"] in rank_roles[:k]}
for k in [int(k) for k in args.ks.split(",")]:
    if str(k) not in res["topk_float"]:
        f = float_top(k)
        res["topk_float"][str(k)] = {"float_tensors": len(f), "err": emb_err(variant(everything - f))}
        save()
        print(f"top {k} roles float:", res["topk_float"][str(k)], flush=True)

# 4. action check for the top-k mixes
ks = [int(k) for k in args.ks.split(",")]
res.setdefault("actions", {})
paths = {}
for k in ks:
    p = os.path.join(args.out, f"vision_int8_top{k}float.onnx")
    if not os.path.exists(p):
        onnx.save(variant(everything - float_top(k)), p)
    paths[k] = p
sessions = {k: sess(p) for k, p in paths.items()}

speed = {}
for k, s in sessions.items():
    s.run(None, {"pixels": test[0]})
    t = time.perf_counter()
    for _ in range(5):
        s.run(None, {"pixels": test[0]})
    speed[k] = round((time.perf_counter() - t) / 5 * 1000, 1)
res["speed_ms_cpu"] = speed
save()
print("speed", speed, flush=True)


def predict(obs, noise, s=None):
    vlm.embed_image = stock_embed_image if s is None else (
        lambda img: torch.from_numpy(s.run(None, {"pixels": img.float().cpu().numpy()})[0]).to(img.device, img.dtype))
    batch = pre(dict(obs))
    with torch.no_grad():
        policy.reset()
        a = policy.predict_action_chunk(batch, noise=noise.clone())
    vlm.embed_image = stock_embed_image
    return post(a).squeeze(0).float().cpu().numpy()


rows = []
for n, (obs, truth, fid) in enumerate(samples("ep*.parquet", args.every)):
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=torch.Generator().manual_seed(n)).to(dev)
    r = predict(obs, noise)
    row = {"frame": fid, "err_torch": float(np.abs(r - truth).mean())}
    for k, s in sessions.items():
        a = predict(obs, noise, s)
        row[f"err_top{k}"] = float(np.abs(a - truth).mean())
    rows.append(row)
    print(json.dumps(row), flush=True)
res["actions"] = {c: round(float(np.mean([r[c] for r in rows])), 3) for c in rows[0] if c != "frame"}
res["action_rows"] = rows
save()
print("ACTIONS", json.dumps(res["actions"]), flush=True)
print("done", f"{time.time() - t0:.0f}s")
