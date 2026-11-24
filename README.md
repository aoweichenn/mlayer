# mlayer

`mcnn` 的具体 layer/kernel 实现模块。每个 layer 以模板形式提供，
`LayerT` / `MatT` / `OptionT` 由消费方（ncnn）注入，避免模块反向依赖 ncnn。

当前已抽取：

| Layer | 文件 | 说明 |
| --- | --- | --- |
| `AbsVal` | `include/mlayer/absval.hpp` | 逐元素绝对值，in-place |

ncnn 侧 `src/layer/absval.h` 使用
`using AbsVal = mlayer::AbsVal<Layer, Mat, Option>;`，x86 特化继续继承它。
