# Changelog

## 0.1.0 — 2026-11-25

* 首批 layer：`AbsVal`、`Clip`、`ReLU`、`Sigmoid`、`Swish`。
* 第二批 layer：`ELU`、`HardSigmoid`、`TanH`、`Mish`、`Softplus`。
* ncnn 侧 base layer 切到 mlayer，arch 特化与注册不变；189/189 测试通过。
