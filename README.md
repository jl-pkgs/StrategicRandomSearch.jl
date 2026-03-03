# `StrategicRandomSearch.jl`

The Strategic Random Search (SRS) —— A New Global Optimization Algorithm in Julia

[![CI](https://github.com/jl-pkgs/StrategicRandomSearch.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/jl-pkgs/StrategicRandomSearch.jl/actions/workflows/CI.yml)
[![Codecov](https://codecov.io/gh/jl-pkgs/StrategicRandomSearch.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jl-pkgs/StrategicRandomSearch.jl)

## Usage

```julia
using StrategicRandomSearch

function functn1(x::Vector{Float64})
  # This is the Goldstein - Price Function
  # Bound X1 = [-2, 2], X2 = [-2, 2]
  # Global Optimum:3.0, (0.0, -1.0)
  x1 = x[1]
  x2 = x[2]
  u1 = (x1 + x2 + 1.0)^2
  u2 = 19 .- 14 .* x1 + 3 .* x1^2 - 14 .* x2 + 6 .* x1 * x2 + 3 .* x2^2
  u3 = (2 .* x1 - 3 .* x2)^2
  u4 = 18 .- 32 .* x1 + 12 .* x1^2 + 48 .* x2 - 36 .* x1 * x2 + 27 .* x2^2
  u5 = u1 * u2
  u6 = u3 * u4
  (1 .+ u5) * (30 .+ u6)
end

lower = Float64.([-2, -2])
upper = Float64.([2, 2])
x0 = Float64.([1, 1])
@time r = SRS(functn1, lower, upper; seed=1)
```

```julia
----------------------------------- 
feval     : 3.0000000000016254
x         : [-5.2170832587254385e-8, -1.0000000634657706]
Iterations: 19
f(x) calls: 1047
----------------------------------- 
  0.006796 seconds (11.26 k allocations: 611.835 KiB)
OptimOutput
  num_iter: Int64 19
  num_call: Int64 1047
  x: Array{Float64}((2,)) [-5.2170832587254385e-8, -1.0000000634657706]
  feval: Float64 3.0000000000016254
  x_iters: Array{Float64}((2, 1000)) [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0]
  feval_iters: Array{Float64}((19,)) [3.0000000009185137, 3.0000000009185137, 3.0000000009185137, 3.0000000009185137, 3.000000000229428, 3.0000000000328972, 3.0000000000032916, 3.0000000000032916, 3.0000000000021387, 3.000000000001862, 3.000000000001675, 3.000000000001675, 3.000000000001629, 3.0000000000016254, 3.0000000000016254, 3.0000000000016254, 3.0000000000016254, 3.0000000000016254, 3.0000000000016254]
  x_calls: Array{Any}((19,))
  feval_calls: Array{Float64}((19,)) [3.0000000009185137, 3.0000000009185137, 3.0000000009185137, 3.0000000009185137, 3.000000000229428, 3.0000000000328972, 3.0000000000032916, 3.0000000000032916, 3.0000000000021387, 3.000000000001862, 3.000000000001675, 3.000000000001675, 3.000000000001629, 3.0000000000016254, 3.0000000000016254, 3.0000000000016254, 3.0000000000016254, 3.0000000000016254, 3.0000000000016254]
```

## Parameters

> 和 Python 版本相比，参数有调整：保留 `update_eps`，并新增 `f_atol_inner` 作为进入精细搜索的阈值。
- `update_eps = true`: 开启逐步精细化搜索，耗时更长，适合高精度优化
- `update_eps = false`: 降低精细化搜索强度，耗时更短，适合参数率定场景
- `f_atol_inner`: 用于换算内部精度阶数，`n_eps = ceil(Int, -log10(f_atol_inner))`
<!-- params参数为目标函数的其他参数 -->


**Table 1.** SRS algorithm parameters

| Name             | Type    | Default       | Description |
| ---------------- | ------- | ------------- | ----------- |
| `maxn`           | Int     | `1000`        | 目标函数最大调用次数。 |
| `seed`           | Int     | `0`           | 随机种子，用于结果可复现。 |
| `verbose`        | Bool    | `true`        | 是否打印迭代日志。 |
| `p`              | Int     | `3`           | 外层搜索保留的精英点数量。 |
| `po`             | Int     | `guess_po(p)` | 内层搜索使用的精英子集数量。 |
| `delta`          | Float64 | `0.01`        | 边界收缩系数，用于局部收缩搜索区间。 |
| `f_atol`         | Float64 | `1e-5`        | 全局收敛阈值（当前版本预留，暂未实际用于终止判断）。 |
| `f_atol_inner`   | Float64 | `1e-4`        | 进入精细搜索的阈值；内部按 `n_eps = ceil(Int, -log10(f_atol_inner))` 转换精度阶数。 |
| `update_eps`     | Bool    | `true`        | 是否在精细搜索阶段动态提高内部精度阶数（`n_eps`）。 |
| `λ_short`        | Float64 | `0.02`        | 短程洗牌强度（精度提高后使用）。 |
| `λ_long`         | Float64 | `0.2`         | 长程洗牌强度（默认全局搜索阶段使用）。 |
| `init_min`       | Int     | `3`           | 启动内层搜索前，外层最少迭代轮数。 |
| `loop_min_inner` | Int     | `2`           | 进入内层搜索后，触发条件检查的最小间隔轮数。 |

## References

Please cite the following article:  

1. Haoshan Wei, Yongqiang Zhang, Changming Liu, Qi Huang, Pengxin Jia, Zhenwu
   Xu, Yuhan Guo, The strategic random search (SRS) – A new global optimizer for
   calibrating hydrological models, ***Environmental Modelling & Software***, 2024,
   https://doi.org/10.1016/j.envsoft.2023.105914.
