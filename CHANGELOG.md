# Changelog

## 0.1.0 — 2026-11-25

* 首批 layer：`AbsVal`、`Clip`、`ReLU`、`Sigmoid`、`Swish`。
* 第二批 layer：`ELU`、`HardSigmoid`、`TanH`、`Mish`、`Softplus`。
* 第三批 layer：`Flatten`、`BNLL`、`Exp`、`Log`、`HardSwish`、`SELU`。
* `Pooling`：通过 `CopyMakeBorder` 函数指针注入 padding，保持 mlayer 不依赖 ncnn。
* `InnerProduct`：通过 `Activation` / `QuantizeInt8` 函数指针注入 fused activation 与 int8 量化。
* `BatchNorm`：推理态 BN，`load_model` 预计算 `a`/`b`。
* `Convolution`：通过 `CopyMakeBorder` / `Activation` / `QuantizeInt8` 注入 padding、fused activation 与 int8 量化；1D 输入兼容路径留在 ncnn 侧。
* ncnn 侧 base layer 切到 mlayer，arch 特化与注册不变；189/189 测试通过。
