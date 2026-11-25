# mlayer

`mcnn` 的具体 layer/kernel 实现模块。每个 layer 以模板形式提供，
`LayerT` / `MatT` / `OptionT`（以及需要时的 `ParamDictT`）由消费方（ncnn）注入，
模块本身不依赖 ncnn。

当前已抽取：

| Layer | 文件 | 说明 |
| --- | --- | --- |
| `AbsVal` | `include/mlayer/absval.hpp` | 逐元素绝对值，in-place |
| `Clip` | `include/mlayer/clip.hpp` | min/max 截断，in-place |
| `ReLU` | `include/mlayer/relu.hpp` | slope ReLU，in-place |
| `Sigmoid` | `include/mlayer/sigmoid.hpp` | in-place sigmoid |
| `Swish` | `include/mlayer/swish.hpp` | in-place swish |
| `ELU` | `include/mlayer/elu.hpp` | alpha ELU |
| `HardSigmoid` | `include/mlayer/hardsigmoid.hpp` | alpha/beta hard sigmoid |
| `TanH` | `include/mlayer/tanh.hpp` | in-place tanh |
| `Mish` | `include/mlayer/mish.hpp` | in-place mish |
| `Softplus` | `include/mlayer/softplus.hpp` | in-place softplus |
| `Flatten` | `include/mlayer/flatten.hpp` | flatten to 1D |
| `BNLL` | `include/mlayer/bnll.hpp` | in-place bnll |
| `Exp` | `include/mlayer/exp.hpp` | exp / pow with scale+shift |
| `Log` | `include/mlayer/log.hpp` | log / log_base with scale+shift |
| `HardSwish` | `include/mlayer/hardswish.hpp` | alpha/beta hard swish |
| `SELU` | `include/mlayer/selu.hpp` | alpha/lambda SELU |
| `Pooling` | `include/mlayer/pooling.hpp` | max/avg/global/adaptive pooling；padding 由消费方注入 |
| `InnerProduct` | `include/mlayer/innerproduct.hpp` | 全连接；activation/int8 quantize 由消费方注入 |
| `BatchNorm` | `include/mlayer/batchnorm.hpp` | 推理态 batch norm（预计算 a/b） |
| `Convolution` | `include/mlayer/convolution.hpp` | 2D 卷积；padding/activation/int8 量化由消费方注入 |
| `ConvolutionDepthWise` | `include/mlayer/convolutiondepthwise.hpp` | depthwise/group 卷积；padding/flatten/activation/int8 由消费方注入 |
| `Deconvolution` | `include/mlayer/deconvolution.hpp` | 转置卷积；输出裁剪/flatten/activation 由消费方注入 |
| `Convolution1D` | `include/mlayer/convolution1d.hpp` | 1D 卷积；padding/activation/flatten 由消费方注入 |
| `Convolution3D` | `include/mlayer/convolution3d.hpp` | 3D 卷积；3D padding/activation 由消费方注入 |
| `PReLU` | `include/mlayer/prelu.hpp` | 1D/2D/3D/4D PReLU |
| `Normalize` | `include/mlayer/normalize.hpp` | L2 normalize，支持 spatial/channel 组合与 eps mode |
| `Threshold` | `include/mlayer/threshold.hpp` | 阈值二值化 |
| `Power` | `include/mlayer/power.hpp` | pow(shift + x * scale, power) |
| `Erf` | `include/mlayer/erf.hpp` | in-place erf |
| `GELU` | `include/mlayer/gelu.hpp` | exact / fast GELU |
| `Bias` | `include/mlayer/bias.hpp` | per-channel bias |
| `Scale` | `include/mlayer/scale.hpp` | 静态/动态 scale + bias |
| `Dropout` | `include/mlayer/dropout.hpp` | 推理态缩放 |
| `ShuffleChannel` | `include/mlayer/shufflechannel.hpp` | channel shuffle / reverse |
| `PixelShuffle` | `include/mlayer/pixelshuffle.hpp` | upscale pixel shuffle，两种 mode |
| `Reorg` | `include/mlayer/reorg.hpp` | reorg，两种 mode |
| `LayerNorm` | `include/mlayer/layernorm.hpp` | 1D–4D layer norm，affine 可选 |
| `RMSNorm` | `include/mlayer/rmsnorm.hpp` | 1D–4D RMS norm，affine 可选 |
| `GroupNorm` | `include/mlayer/groupnorm.hpp` | group norm，affine 可选 |
| `Softmax` | `include/mlayer/softmax.hpp` | 1D–4D softmax，正/负 axis |
| `YoloV5Focus` | `include/mlayer/yolov5focus.hpp` | 可选 examples 自定义层（非 ncnn core） |
| `Slice` | `include/mlayer/slice.hpp` | 1D–4D 切片，slices/indices 两种模式 |
| `BinaryOp` | `include/mlayer/binaryop.hpp` | 逐元素二元运算，含标量与广播 |
| `Permute` | `include/mlayer/permute.hpp` | 1D–4D 维度重排 |
| `Reshape` | `include/mlayer/reshape.hpp` | 数值参数与简单 shape 表达式 |
| `MatMul` | `include/mlayer/matmul.hpp` | 1D–4D 矩阵乘，transB |
| `Interp` | `include/mlayer/interp.hpp` | nearest / bilinear 上采样 |
| `Embed` | `include/mlayer/embed.hpp` | token embedding 查表 |
| `Gemm` | `include/mlayer/gemm.hpp` | fp32 与 int8 block 量化 Gemm |
| `RotaryEmbed` | `include/mlayer/rotaryembed.hpp` | RoPE，interleaved/半宽 cache |
| `SDPA` | `include/mlayer/sdpa.hpp` | scaled dot-product attention + KV cache |

ncnn 侧对应 `src/layer/<name>.h` 使用
`using <Name> = mlayer::<Name><Layer, Mat, Option[, ParamDict]>;`，
arch 特化继续继承该别名。
