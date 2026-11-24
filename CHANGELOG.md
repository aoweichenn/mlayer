# Changelog

## 0.1.0 — 2026-11-25

* 首批 layer：`AbsVal`、`Clip`、`ReLU`、`Sigmoid`、`Swish`。
* 第二批 layer：`ELU`、`HardSigmoid`、`TanH`、`Mish`、`Softplus`。
* 第三批 layer：`Flatten`、`BNLL`、`Exp`、`Log`、`HardSwish`、`SELU`。
* `Pooling`：通过 `CopyMakeBorder` 函数指针注入 padding，保持 mlayer 不依赖 ncnn。
* `InnerProduct`：通过 `Activation` / `QuantizeInt8` 函数指针注入 fused activation 与 int8 量化。
* ncnn 侧 base layer 切到 mlayer，arch 特化与注册不变；189/189 测试通过。
