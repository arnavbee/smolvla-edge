"""Mixed int8 vision tower: which blocks to keep float, picked from layer_sens.json.

layer_sens.py found layer 10 alone breaks it (only L10 activations int8 -> same error as all int8),
and the per-tensor greedy ranking doesn't recover well. So this tries whole blocks and the
residual adds, then runs the action check on the best few.
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
ap.add_argument("--keep", type=int, default=3, help="mixes that get the action check")
ap.add_argument("--mixes", default="", help="comma list: action-check these instead of the best --keep")
ap.add_argument("--out", default="onnx")
args = ap.parse_args()
dev = torch.device("mps")
res_path = "layer_mix.json"
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
in_blocks = lambda *bs: {t for t, p in pairs.items() if p["block"] in bs}
residual = {t for t, p in pairs.items() if p["role"] in ("/encoder/layers.N/Add", "/encoder/layers.N/Add_1")}
mlp_out = {t for t, p in pairs.items() if p["role"].startswith("/encoder/layers.N/mlp/fc2")}
gelu = {t for t, p in pairs.items() if "activation_fn" in p["role"]}
mixes = {  # name -> activations kept in float
    "all_int8": set(),
    "L10": in_blocks("L10"),
    "L10_L11": in_blocks("L10", "L11"),
    "L09_L10_L11": in_blocks("L09", "L10", "L11"),
    "L05_to_L11": in_blocks(*[f"L{i:02d}" for i in range(5, 12)]),
    "residual_adds": residual,
    "L10_plus_residual": in_blocks("L10") | residual,
    "L10_plus_gelu": in_blocks("L10") | gelu,
    "L10_plus_fc2_in": in_blocks("L10") | mlp_out,
    "L10_L11_plus_residual": in_blocks("L10", "L11") | residual,
    # the other way round: int8 activations only where a block alone did no harm
    "int8_only_L01_L04": everything - in_blocks("L01", "L02", "L03", "L04", "embed", "connector"),
    "int8_only_L00_L04": everything - in_blocks("L00", "L01", "L02", "L03", "L04", "embed", "connector"),
    "int8_only_L00_L09_minus_L05_L08": everything - in_blocks("L00", "L01", "L02", "L03", "L04", "L09", "embed", "connector"),
}
res.setdefault("emb", {})
t0 = time.time()
for name, f in mixes.items():
    if name not in res["emb"]:
        res["emb"][name] = {"float_tensors": len(f), "of": len(pairs), "err": emb_err(variant(everything - f))}
        save()
        print(name, res["emb"][name], f"{time.time() - t0:.0f}s", flush=True)

# action check on the best few that still keep most activations int8
ranked = sorted((n for n in mixes if n != "all_int8"), key=lambda n: res["emb"][n]["err"])
pick = args.mixes.split(",") if args.mixes else ranked[: args.keep]
res.setdefault("picked", [])
res["picked"] += [n for n in pick if n not in res["picked"]]
paths = {}
for n in pick:
    p = os.path.join(args.out, f"vision_mix_{n}.onnx")
    onnx.save(variant(everything - mixes[n]), p)
    paths[n] = p
sessions = {n: sess(p) for n, p in paths.items()}
del qdq

speed = {}
for n, s in sessions.items():
    s.run(None, {"pixels": test[0]})
    t = time.perf_counter()
    for _ in range(5):
        s.run(None, {"pixels": test[0]})
    speed[n] = {"ms_per_image_cpu": round((time.perf_counter() - t) / 5 * 1000, 1),
                "MB": round(os.path.getsize(paths[n]) / 1e6)}
res.setdefault("speed", {}).update(speed)
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
for i, (obs, truth, fid) in enumerate(samples("ep*.parquet", args.every)):
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=torch.Generator().manual_seed(i)).to(dev)
    r = predict(obs, noise)
    row = {"frame": fid, "err_torch": float(np.abs(r - truth).mean())}
    for n, s in sessions.items():
        a = predict(obs, noise, s)
        row[f"err_{n}"] = float(np.abs(a - truth).mean())
        row[f"vs_torch_{n}"] = float(np.abs(a - r).mean())
    rows.append(row)
    print(json.dumps(row), flush=True)
res.setdefault("actions", {}).update({c: round(float(np.mean([r[c] for r in rows])), 3) for c in rows[0] if c != "frame"})
res.setdefault("action_rows", {})[",".join(pick)] = rows
save()
print("ACTIONS", json.dumps(res["actions"]), flush=True)
