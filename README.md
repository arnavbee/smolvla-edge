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

## Todo

- encode all cameras in one batched pass
- fewer vision tokens, then check what it costs on a LeRobot dataset
- reuse vision work across overlapping chunks
- export the vision tower to ONNX, then RKNN for the RK3588 NPU
