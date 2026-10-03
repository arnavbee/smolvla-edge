# smolvla-edge

Trying to get SmolVLA (`lerobot/smolvla_base`, 450M) running in real time on cheap robot boards that aren't NVIDIA. First target is the Rockchip RK3588.

## Numbers so far

M1 Mac, 8 GB, MPS. One chunk = 50 actions, 10 denoise steps, images padded to 512x512.

```
python bench.py --device mps --dtype float16 --cams 3
```

| setup | chunk | vision | prefix | denoise step |
|---|---|---|---|---|
| fp32, 3 cams | 1700 ms | 1149 ms | 136 ms | 41 ms |
| fp16, 3 cams | 1570 ms | 1020 ms | 141 ms | 40 ms |
| bf16, 3 cams | 2064 ms | 1358 ms | 186 ms | 52 ms |
| fp16, 2 cams | 1196 ms | 695 ms | 109 ms | 39 ms |
| fp16, 1 cam | 798 ms | 355 ms | 76 ms | 36 ms |
| CPU fp32, 3 cams | 2217 ms | 1384 ms | 192 ms | 63 ms |

## Notes

- Vision is most of the time, about 350 ms per camera, and it scales linearly because the cameras go through the encoder one at a time. The action expert is cheap.
- fp16 only buys ~8% on the M1. bf16 is slower.
- Casting the policy to fp16/bf16 crashes on MPS. `denoise_step` casts to float32 right before `action_out_proj` (`modeling_smolvla.py:808`) and the Euler loop keeps `x_t` in float32, so the half-precision linears get float32 input. `bench.py` keeps the four action-side linears in fp32 to get around it.
- At 30 Hz a 50-action chunk lasts ~1.67 s, so 3 cams on the M1 barely keeps up. An RK3588 won't without changes.

## Batching cameras and image size

`fast.py` has `batch_vision()`, which sends all cameras through the vision encoder in one pass. Output matches the stock model (max diff ~1e-6 in fp32) but it's barely faster: 1555 ms vs 1570 ms for 3 cams. At 512 px each image is already 1024 patches, so the GPU is full with one image and batching doesn't help.

Image size is what moves it. Vision encoder alone, one image, fp16:

| input | time | tokens to the VLM |
|---|---|---|
| 512 px | 354 ms | 64 |
| 384 px | 157 ms | 36 |
| 256 px | 53 ms | 16 |

Full chunk, 3 cams, fp16:

```
python bench.py --device mps --dtype float16 --res 256
```

| input | chunk | vision | prefix |
|---|---|---|---|
| 512 px | 1570 ms | 1020 ms | 141 ms |
| 384 px | 970 ms | 502 ms | 94 ms |
| 256 px | 632 ms | 210 ms | 62 ms |

### Does it still pick the right moves?

No. `res_check.py` runs a fine-tuned SmolVLA ([edge-inference/smolvla-so101-pick-orange](https://huggingface.co/edge-inference/smolvla-so101-pick-orange)) on recorded frames from its dataset (LightwheelAI/leisaac-pick-orange, 3 episodes, 31 frames), same noise at every size, and compares the 50-step chunk to the recorded actions. Mean abs error, joint units:

| input | vs recorded | vs 512 output |
|---|---|---|
| 512 px | 1.2 | 0 |
| 384 px | 10.4 | 10.3 |
| 256 px | 11.7 | 11.5 |

For scale: rerunning 512 with different noise moves it by 1.0, and the arm's typical motion over 50 steps is 17.4. So at 384 or 256 the error is most of the movement. Naive downscaling is out.

Caveats: sim data, and these episodes were in the training set.

## int8 vision tower, still 512 px

`vision_int8.py` exports the vision tower + connector to ONNX, quantizes it three ways with onnxruntime (calibration on two episodes not in the test set), swaps each back in for `embed_image` and reruns the same 31-frame action check. Timing is onnxruntime on the M1 CPU, so it doesn't compare with the MPS numbers above.

| vision tower | vs recorded | vs torch output | ms / image (CPU) | size |
|---|---|---|---|---|
| torch fp32 | 1.16 | 0 | | |
| ONNX fp32 | 1.16 | 0.05 | 2785 | 393 MB |
| int8 dynamic (weights only) | 1.32 | 0.82 | 1384 | 102 MB |
| int8 static, everything (QDQ) | 12.36 | 12.24 | 1421 | 100 MB |
| int8 static, MatMul/Conv only | 2.63 | 2.43 | 2534 | 356 MB |

- Weight-only int8 is close to free: 2x faster on CPU, 4x smaller, error within the noise band (~1.0).
- Full static int8 breaks it the same way downscaling did. Quantizing activations is the problem.
- Keeping softmax, layernorm, GELU and the adds in float gets most of it back (2.6), but onnxruntime then runs it slower than fp32 because of all the QDQ pairs.
- The RK3588 NPU wants full int8, so the next job is finding which activations can't take int8 (per-layer sensitivity) and keeping only those in fp16.

## Todo

- ~~encode all cameras in one batched pass~~ done, no real gain on M1 (may still matter on the NPU)
- ~~check how much 384 / 256 px changes the actions~~ done, breaks the policy
- fine-tune at 256 px (free Colab/Kaggle GPU) and see how much comes back
- ~~int8 vision tower at 512~~ weight-only works, full static int8 breaks it
- per-layer sensitivity: which activations can't go int8
- reuse vision work across overlapping chunks
- export the vision tower to ONNX, then RKNN for the RK3588 NPU
