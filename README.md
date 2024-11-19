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
feval     : 3.0000000001097433
x         : [2.92590422779071e-7, -0.9999994691368533]
Iterations: 22
f(x) calls: 1002
----------------------------------- 
  0.038590 seconds (10.67 k allocations: 624.922 KiB)
OptimOutput
  num_iter: Int64 22
  num_call: Int64 1002
  x: Array{Float64}((2,)) [2.92590422779071e-7, -0.9999994691368533]
  feval: Float64 3.0000000001097433
  BY: Array{Float64}((22,)) [37.96459778926425, 37.96459778926425, 37.96459778926425, 37.96459778926425, 26.841025563057322, 26.841025563057322, 21.605702566807324, 10.148481159630524, 5.8236442027222175, 3.0498664685729526  …  3.0000590042771944, 3.000032913496003, 3.000032913496003, 3.000001379924515, 3.0000004974780015, 3.000000289269808, 3.000000289269808, 3.000000022318343, 3.0000000001097433, 3.0000000001097433]
  EachPar: Array{Float64}((22, 2)) [-0.616285424931557 -0.4649638918252177; -0.616285424931557 -0.4649638918252177; … ; 2.92590422779071e-7 -0.9999994691368533; 2.92590422779071e-7 -0.9999994691368533]
  x_iters: Array{Float64}((27, 2)) [1.1122372527217e-311 1.112237252943e-311; 1.1122372531644e-311 1.1122372533857e-311; … ; 1.1122372637887e-311 1.11223726401e-311; 1.1122372642314e-311 1.112237264453e-311]
  feval_iters: Array{Float64}((972,)) [37.96459778926425, 37.96459778926425, 37.96459778926425, 37.96459778926425, 37.96459778926425, 37.96459778926425, 37.96459778926425, 37.96459778926425, 37.96459778926425, 37.96459778926425  …  3.0000000001097433, 3.0000000001097433, 3.0000000001097433, 3.0000000001097433, 3.0000000001097433, 3.0000000001097433, 3.0000000001097433, 3.0000000001097433, 3.0000000001097433, 3.0000000001097433]
```

## Parameters

>和Python比有更新，加入update_eps参数，
- `update_eps = true`: 增加精细化搜索，耗时长，适合寻找精度要求高的测试函数最优解
- `update_eps = false`: 减小精细化搜索，耗时短，适合寻水文模型参数率定寻找最优解
<!-- params参数为目标函数的其他参数 -->


**Table 1.** SRS algorithm parameters

| Name             | Type  | Default | Description                                                                                                                                              |
| ---------------- | ----- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `p`              | int   | 3       | p is the key parameter, and the value is generally 3-20, which needs to be given according to the specific situation                                     |
| `deps`           | float | 12      | (0, inf), key parameter for adjusting the precision, the larger the value, the higher the precision and the longer the time                              |
| `delta`          | float | 0.01    | (0, 0.5), key parameter for adjusting the precision, the larger the value, the higher the precision and the longer the time                              |
| `Vectorization`  | bool  | false   | Whether the objective function satisfies the vectorization condition                                                                                     |
| `num`            | int   | 1000    | if Vectorization=True: num=1000 else: num=10000 (defult).                                                                                                |
|                  |       | 10000   | The key parameter, representing the maximum number of times the target function is called. When testing, the accuracy can be improved by increasing num. |
| `MAX`            | bool  | true    | Whether to find the maximum value of the objective function.                                                                                             |
| `OptimalValue`   | float | None    | The optimal value of the objective function.                                                                                                             |
| `ObjectiveLimit` | float | None    | When the optimal value is known, the algorithm terminates                                                                                                |
|                  |       |         | within `ObjectiveLimit` of the optimal value.                                                                                                            |
| `eps`            | Int   | 4       | (0, +inf), it is not critical, and adjustment is not recommended.                                                                                        |
| `update_eps`     | bool  | true    | Whether or not to update eps to do refined search parameters. Generally, it can be “false” for model parameter calibration.                              |
| `ShortLambda`    | float | 0.02    | (0, 0.1), not critical, and adjustment is not recommended.                                                                                               |
| `LongLambda`     | float | 0.2     | (0.1, 1), not critical, and adjustment is not recommended.                                                                                               |
| `InitialLt`      | int   | 3       | (0, 10), not critical, and adjustment is not recommended.                                                                                                |
| `Lt`             | int   | 2       | (0, 10), not critical, and adjustment is not recommended.                                                                                                |

## References

Please cite the following article:  

1. Haoshan Wei, Yongqiang Zhang, Changming Liu, Qi Huang, Pengxin Jia, Zhenwu
   Xu, Yuhan Guo, The strategic random search (SRS) – A new global optimizer for
   calibrating hydrological models, ***Environmental Modelling & Software***, 2024,
   https://doi.org/10.1016/j.envsoft.2023.105914.
