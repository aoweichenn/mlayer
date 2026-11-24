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

ncnn 侧对应 `src/layer/<name>.h` 使用
`using <Name> = mlayer::<Name><Layer, Mat, Option[, ParamDict]>;`，
arch 特化继续继承该别名。
