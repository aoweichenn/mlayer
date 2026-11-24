# Changelog

## 0.1.0 — 2026-11-25

* 首批 layer：`AbsVal`、`Clip`、`ReLU`、`Sigmoid`、`Swish`。
* 第二批 layer：`ELU`、`HardSigmoid`、`TanH`、`Mish`、`Softplus`。
* 第三批 layer：`Flatten`、`BNLL`、`Exp`、`Log`、`HardSwish`、`SELU`。
* `Pooling`：通过 `CopyMakeBorder` 函数指针注入 padding，保持 mlayer 不依赖 ncnn。
* `InnerProduct`：通过 `Activation` / `QuantizeInt8` 函数指针注入 fused activation 与 int8 量化。
* `BatchNorm`：推理态 BN，`load_model` 预计算 `a`/`b`。
* `Convolution`：通过 `CopyMakeBorder` / `Activation` / `QuantizeInt8` 注入 padding、fused activation 与 int8 量化；1D 输入兼容路径留在 ncnn 侧。
* `ConvolutionDepthWise`：depthwise 与 group 两条路径，int8 group scale 展开，padding/flatten 注入。
* `Deconvolution`：转置卷积、`copy_cut_border` 输出裁剪、dynamic weight 转置，全部策略注入。
* `border.hpp` / `quantize.hpp`：共享 `kBorderConstant` 与 `float_to_int8`，避免多 header 重复定义。
* `Convolution1D` / `Convolution3D`：1D/3D 卷积，padding、activation、dynamic weight flatten 注入。
* `PReLU` / `Normalize`：无 arch 依赖的 in-place 归一化层。
* `Threshold` / `Power` / `Erf` / `GELU`：无权重 in-place 激活层。
* `Bias` / `Scale` / `Dropout`：带权重或参数的逐元素层。
* `ShuffleChannel` / `PixelShuffle` / `Reorg`：通道/空间重排层。
* ncnn 侧 base layer 切到 mlayer，arch 特化与注册不变；189/189 测试通过。
