# Strategic Random Search in Julia

The Strategic Random Search (SRS) —— A New Global Optimization Algorithm  


>和Python比有更新，加入update_eps参数，
- `update_eps = true`: 增加精细化搜索，耗时长，适合寻找精度要求高的测试函数最优解
- `update_eps = false`: 减小精细化搜索，耗时短，适合寻水文模型参数率定寻找最优解
<!-- params参数为目标函数的其他参数 -->


**Table 1.** SRS algorithm parameters

| Name             | Type  | Default                                             | Description                                                                                                                                              |
| ---------------- | ----- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `p`              | int   | 3                                                   | p is the key parameter, and the value is generally 3-20, which needs to be given according to the specific situation                                     |
| `sp`             | int   | sp = p(p<=5)<br />sp = 5(5<p<12)<br />sp = 12(p=12) | [3, p]                                                                                                                                                   |
| `deps`           | float | 12                                                  | (0, inf), key parameter for adjusting the precision, the larger the value, the higher the precision and the longer the time                              |
| `delta`          | float | 0.01                                                | (0, 0.5), key parameter for adjusting the precision, the larger the value, the higher the precision and the longer the time                              |
| `Vectorization`  | bool  | false                                               | Whether the objective function satisfies the vectorization condition                                                                                     |
| `num`            | int   | 1000                                                | if Vectorization=True: num=1000 else: num=10000 (defult).                                                                                                |
|                  |       | 10000                                               | The key parameter, representing the maximum number of times the target function is called. When testing, the accuracy can be improved by increasing num. |
| `MAX`            | bool  | true                                                | Whether to find the maximum value of the objective function.                                                                                             |
| `OptimalValue`   | float | None                                                | The optimal value of the objective function.                                                                                                             |
| `ObjectiveLimit` | float | None                                                | When the optimal value is known, the algorithm terminates                                                                                                |
|                  |       |                                                     | within `ObjectiveLimit` of the optimal value.                                                                                                            |
| `eps`            | Int   | 4                                                   | (0, +inf), it is not critical, and adjustment is not recommended.                                                                                        |
| `update_eps`     | bool  | true                                                | Whether or not to update eps to do refined search parameters. Generally, it can be “false” for model parameter calibration.                              |
| `ShortLambda`    | float | 0.02                                                | (0, 0.1), not critical, and adjustment is not recommended.                                                                                               |
| `LongLambda`     | float | 0.2                                                 | (0.1, 1), not critical, and adjustment is not recommended.                                                                                               |
| `InitialLt`      | int   | 3                                                   | (0, 10), not critical, and adjustment is not recommended.                                                                                                |
| `Lt`             | int   | 2                                                   | (0, 10), not critical, and adjustment is not recommended.                                                                                                |
| `params`         | Tuple | ()                                                  | parameters to `ObjectiveFunction`                                                                                                                        |

## References

Please cite the following article:  

1. Haoshan Wei, Yongqiang Zhang, Changming Liu, Qi Huang, Pengxin Jia, Zhenwu
   Xu, Yuhan Guo, The strategic random search (SRS) – A new global optimizer for
   calibrating hydrological models, ***Environmental Modelling & Software***, 2024,
   https://doi.org/10.1016/j.envsoft.2023.105914.
