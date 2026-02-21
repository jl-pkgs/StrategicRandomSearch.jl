guess_po(p::Int) = p < 5 ? p : (p < 12 ? 5 : 12)

"""
  SRS(f, lower, upper; maxn=1000, )

## Arguments
- `f`     : The objective function to be optimized
- `lower` : The lower bound of the parameter to be determined
- `upper` : The upper bound of the parameter to be determined
- `args`  : Additional arguments to be passed to `f`

## Keyword Arguments
- `maxn`  : The nanmaximum number of iterations
- `kw`    : Additional keyword arguments to be passed to `f`

- `p`: 参数翻多少倍
"""
function SRS(
  f::Function, lower::Vector{Float64}, upper::Vector{Float64}, args...; maxn::Int=1000,
  verbose=true, goal_multiplier=-1,
  p::Int=3,                 # Fig 2a, p optimal points (purple squares and yellow squares)
  po::Int=guess_po(p),      # Fig 2a, po optimal points (yellow squares), 精英中的精英
  deps::Int=12,        # x 空间收缩终止条件
  delta::Float64=0.01, # 区间收缩因子, `delta_Mbounds = Mbounds * delta`
  f_opt::Float64=NaN, f_atol::Float64=NaN, eps::Int=4, # 进入/维持某些全局跳出逻辑
  update_eps::Bool=true, λ_short::Float64=0.02, λ_long::Float64=0.2, InitialLt::Int=3, Lt::Int=2,
  seed::Int=0, kw...)

  Random.seed!(seed) # make the result reproducible

  fun(x) = f(x, args...; kw...)

  OLindex = !isnan(f_atol)
  p1 = po
  OV = f_opt

  # 初始化参数
  npar = length(lower)
  n_ensemble = 3 * npar + 3
  n_reps = n_ensemble * p # 总参数，每次循环，总参数个数

  m1 = Int(nanmax(floor(Int, n_ensemble * p / po) + 1, 9))
  # popsize = Int(m1 * sp * ones(Int, n, 1))
  psize = m1 * ones(Int, npar, 1)
  Mbounds = upper .- lower

  ## 静态变量
  M = upper .- lower
  BE = copy(upper)
  BD = copy(lower)
  Bb = repeat(BD, 1, n_ensemble)
  Be = repeat(BE, 1, n_ensemble)

  k = M ./ (psize .- 1)
  Index = 0
  MM = m1 * p

  num_call = 0
  num_iter = 0
  feS = 0

  # 初始化解空间
  x = (upper .+ lower) ./ 2 .+ (M .* (rand(npar, MM) .- 1) ./ 2)
  y = Vector{Float64}(undef, MM)

  num_call, feS = calculate_goal!(y, fun, x, num_call, feS)

  ## 这是怎么回事？
  yps, Xp, Xb = select_optimal(y, x; p)
  # BestValueFE = yps[1]
  # EachParFE = Xp[:, 1]

  EachPar = zeros(Float64, npar, maxn)
  BestValue = zeros(Float64, maxn, p) # 前n个作为候选
  neps = eps
  sss = 0

  BY = Float64[]
  _feval_iters = Float64[]
  _x_iters = []

  X1 = zeros(Float64, npar, n_reps)

  # 主循环
  while true
    lambda, lt_val = (eps > neps + 2) ? (λ_short, copy(Lt)) : (λ_long, copy(Lt))
    sss == 0 && (lt_val = InitialLt)

    search_init_X!(X1, Xp, Xb, p, n_reps, lambda, Mbounds, Bb, Be)

    y = Vector{Float64}(undef, n_reps)
    num_call, feS = calculate_goal!(y, fun, X1, num_call, feS)
    update_best_theta!(yps, Xp, Xb, y, X1, p) # update yps, Xp, Xb

    num_iter += 1
    Index += 1

    BestValue[num_iter, :] .= yps # 这里近记录了一次表现最好的

    indexX = sortperm(yps)
    sort!(yps)

    append!(BY, nanminimum(yps)) # BY 记录的是 最佳
    EachPar[:, num_iter] = Xp[:, indexX[1]] # 做优的一个

    populate_best_value_fe!(_feval_iters, BestValue, num_iter, n_reps)
    populate_each_par_fe!(_x_iters, EachPar, num_iter, n_reps)

    if verbose
      feval = nanminimum(BestValue[num_iter, :])
      @printf("[iter = %3d, num_call = %4d] out: goal = %f\n", num_iter, num_call, feval)
    end

    if OLindex && !isnan(OV)
      if abs(OV * goal_multiplier - nanminimum(BestValue[num_iter, :])) < abs(f_atol)
        break
      end
    end

    if Index > lt_val
      if abs(nanminimum(log10.(Mbounds ./ M))) > deps || num_call > maxn
        break
      end
      ineed = abs(nanminimum(BY[num_iter-lt_val+1]) - nanminimum(BY[num_iter]))
      if abs(log10(nanmax(ineed, 10.0^(-eps - 1)))) ≥ eps
        sss = 1
        bb = nanminimum(Xp', dims=1)
        be = nanmaximum(Xp', dims=1)
        lower .= max.(min.(lower, bb[:] .- k[:]), BD) # lower and upper are updated
        upper .= min.(max.(upper, be[:] .+ k[:]), BE)
        Mbounds .= upper .- lower
        k .= Mbounds ./ (psize .- 1)
        Xp1 = Xp[:, indexX[1:po]]
        # println(Mbounds)

        BestX = copy(Xp1)
        x = zeros(npar, m1 * po)

        BestY = zeros(Float64, npar + 1, p1)
        BestY[1, :] .= yps[1:p1]

        BX = EachPar[:, num_iter]

        num_call, feS, Index1 = perform_inner_search!(x, Xp1, BestX, BestY, BX, lower, upper, k, psize, p1, m1,
          fun, num_call, feS)

        num_iter += 1
        BestValue[num_iter, 1:p1] .= nanminimum(BestY[Index1-npar:Index1, :], dims=1)'

        append!(BY, nanminimum(BestValue[num_iter, 1:p1]))
        EachPar[:, num_iter] = BX

        n_reps1 = m1 * p1 * npar
        populate_best_value_fe!(_feval_iters, BestValue, num_iter, [n_reps, n_reps1])
        populate_each_par_fe!(_x_iters, EachPar, num_iter, n_reps)

        if verbose
          feval = nanminimum(BestValue[num_iter, 1:p1])
          @printf("[iter = %3d, num_call = %4d]  in: goal = %f\n", num_iter, num_call, feval)
        end

        Xp[:, 1:p1] .= BestX
        bb = nanminimum(Xp', dims=1)
        be = nanmaximum(Xp', dims=1)

        delta_Mbounds = Mbounds * delta
        lower .= max.(lower, bb[:] .- delta_Mbounds)
        upper .= min.(upper, be[:] .+ delta_Mbounds)
        k .= (upper .- lower) ./ (psize .- 1)

        perform_secondary_search!(x, Xp, Xb, lower, upper, m1, p1)

        # 生成随机数 N
        N = (BE .- BD) .* rand(npar, m1 * p1) .+ BD
        x[x.<BD] .= N[x.<BD]
        x[x.>BE] .= N[x.>BE]

        # 计算 y
        MM = m1 * p1
        y = Vector{Float64}(undef, MM)
        num_call, feS = calculate_goal!(y, fun, x, num_call, feS)

        # 更新 yps 和 x
        yps[1:p1] .= nanminimum(BestY, dims=1)'
        y = vcat(y, yps)
        x = hcat(x, Xp)

        # 对 yps 排序并更新
        yps, indexY = sort(y), sortperm(y)
        yps = yps[1:p]
        xneed = abs(yps[1] - BY[num_iter])

        # 率定水文模型不开这个部分（因为水文模型要求精度不高，打开会使前面的等距搜索太慢了）
        # 检查是否需要更新 eps
        if abs(log10(nanmax(xneed, 10.0^(-eps - 1)))) ≥ eps && update_eps
          eps += 1
        end

        Xp = x[:, indexY[1:p]]
        Xb = x[:, indexY[end]]
        heihei1 = falses(npar)
        heihei2 = falses(npar)
        nx1 = copy(BE)
        nx2 = copy(BD)

        for i = 1:p
          nx = Xp[:, i]
          heihei1 .= heihei1 .| (nx .>= BE)
          heihei2 .= heihei2 .| (nx .<= BD)
          nx1[heihei1] .= min.(nx[heihei1], nx1[heihei1])
          nx2[heihei2] .= max.(nx[heihei2], nx2[heihei2])
        end

        upper[heihei1] .= min.(nx1[heihei1] .+ k[heihei1], BE[heihei1])
        lower[heihei2] .= max.(nx2[heihei2] .- k[heihei2], BD[heihei2])

        n_reps2 = m1 * p1
        populate_best_value_fe!(_feval_iters, BestValue, num_iter, n_reps2)
        populate_each_par_fe!(_x_iters, EachPar, num_iter, n_reps2)
      end
    end
  end
  return OptimOutput(_feval_iters, _x_iters, BY, EachPar, num_call, num_iter; verbose)
end
