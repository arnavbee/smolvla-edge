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

## Which activations can't take int8?

The vision tower has 492 activation Q/DQ pairs in the static model. Dropping a pair leaves that tensor in float while the weights stay int8, so you can test any mix without recalibrating. `layer_sens.py` sweeps them, scored first by how far the tower output moves from fp32 on held-out images (relative error), then by the same 31-frame action check as above.

One block at a time, only that block's activations int8 (weight-only on its own is 0.29):

| block | L00 | L01-L04 | L05-L08 | L09 | L10 | L11 |
|---|---|---|---|---|---|---|
| tower error | 0.46 | 0.30-0.33 | 0.55-0.65 | 0.50 | 1.63 | 0.79 |

Layer 10 on its own does as much damage as quantizing everything (1.62). But taking it out isn't enough, because most of the other blocks also hurt a bit and it adds up.

Action check, mean abs error vs recorded (torch fp32 is 1.16, weight-only int8 is 1.32):

| activations kept in float | float / 492 | vs recorded | vs torch |
|---|---|---|---|
| none (full static) | 0 | 12.36 | 12.24 |
| top 8 most sensitive tensor roles | 85 | 3.75 | - |
| layers 10, 11 | 80 | 20.44 | 20.58 |
| layers 10, 11 + residual adds | 100 | 4.98 | 4.73 |
| layers 5-11 | 280 | 3.95 | 3.74 |
| everything except layers 1-4 | 321 | 2.80 | 2.46 |

`layer_mix.py` has the block mixes. Results in `layer_sens.json` and `layer_mix.json`.

- Nothing gets close to weight-only. Even with only layers 1-4 in int8 the error doubles.
- Floating layers 10 and 11 made it worse (20.4). The tower error went down (0.77) but the actions got worse, so tower error is a bad stand-in for the action check. Always run the actions.
- So it's not a few bad layers. Plain min/max int8 on activations is too coarse almost everywhere in this tower.

CPU speed, one 512 px image, onnxruntime, M1 idle (the table further up was timed with other jobs running, so it's slower):

| vision tower | ms / image | size |
|---|---|---|
| ONNX fp32 | 1328 | 393 MB |
| int8 weights only | 666 | 102 MB |
| int8 static, everything | 653 | 100 MB |
| mixed, layers 1-4 int8 | 853 | 100 MB |

Full int8 isn't faster than weight-only on CPU anyway. It only matters for the NPU, which wants int8 activations.

Next things to try for the NPU: 16-bit activations (RKNN has 16-bit modes, need to check which run on the RK3588), better calibration (percentile or entropy instead of min/max, needs a machine with more than 8 GB), or SmoothQuant to move the outliers into the weights.

## Todo

- ~~encode all cameras in one batched pass~~ done, no real gain on M1 (may still matter on the NPU)
- ~~check how much 384 / 256 px changes the actions~~ done, breaks the policy
- fine-tune at 256 px (free Colab/Kaggle GPU) and see how much comes back
- ~~int8 vision tower at 512~~ weight-only works, full static int8 breaks it
- ~~per-layer sensitivity: which activations can't go int8~~ done, no small set; it's spread across the tower
- w8a16 or better calibration for the activations
- reuse vision work across overlapping chunks
- export the vision tower to ONNX, then RKNN for the RK3588 NPU
