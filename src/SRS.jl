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
    f::Function, lower::Vector{Float64}, upper::Vector{Float64}, args...;
    maxn::Int=1000,
    verbose=true, goal_multiplier=-1,
    p::Int=3,                 # Fig 2a, p optimal points (purple squares and yellow squares)
    po::Int=guess_po(p),      # Fig 2a, po optimal points (yellow squares), 精英中的精英
    # deps::Int=12,           # x 空间收缩终止条件
    delta::Float64=0.01,      # 区间收缩因子, `delta_Mbounds = Mbounds * delta`
    f_atol::Float64=NaN,
    eps::Int=4,               # 进入/维持某些全局跳出逻辑
    update_eps::Bool=true,
    λ_short::Float64=0.02, λ_long::Float64=0.2,
    init_loop_min::Int=3, loop_min::Int=2,
    seed::Int=0, kw...)

    Random.seed!(seed) # make the result reproducible
    fun(x) = f(x, args...; kw...)

    f_opt::Float64 = NaN
    p1 = po

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
    ub = copy(upper)
    lb = copy(lower)
    LB = repeat(lb, 1, n_ensemble)
    UB = repeat(ub, 1, n_ensemble)

    k = M ./ (psize .- 1)
    MM = m1 * p

    num_call = 0
    num_iter = 0
    feS = 0

    # 初始化解空间
    x = (upper .+ lower) ./ 2 .+ (M .* (rand(npar, MM) .- 1) ./ 2)
    y = Vector{Float64}(undef, MM)

    num_call, feS = calculate_goal!(y, fun, x, num_call, feS)

    y_best, X_best, X_worst = select_optimal(y, x; p)


    best_x_iters = zeros(Float64, npar, maxn)
    BestValue = zeros(Float64, maxn, p) # 前n个作为候选
    neps = eps

    best_feval_iters = Float64[]
    _feval_iters = Float64[]
    _x_iters = []
    inner_search_started = false

    X1 = zeros(Float64, npar, n_reps)

    FT = Float64
    feval = 0.0
    fevals = FT[] # 每次loop的最优值
    loop = 0

    # 主循环
    while true
        λ = (eps > neps + 2) ? λ_short : λ_long
        current_loop_min = inner_search_started ? loop_min : init_loop_min
        search_init_X!(X1, X_best, X_worst, p, n_reps, λ, Mbounds, LB, UB)

        y = Vector{Float64}(undef, n_reps)
        num_call, feS = calculate_goal!(y, fun, X1, num_call, feS)
        update_best_theta!(y_best, X_best, X_worst, y, X1, p) # update yps, Xp, Xb

        num_iter += 1
        loop += 1

        BestValue[num_iter, :] .= y_best # 这里近记录了一次表现最好的

        indexX = sortperm(y_best)
        sort!(y_best)

        append!(best_feval_iters, nanminimum(y_best)) # BY 记录的是 最佳
        best_x_iters[:, num_iter] = X_best[:, indexX[1]] # 做优的一个

        populate_best_value_fe!(_feval_iters, BestValue, num_iter, n_reps)
        populate_each_par_fe!(_x_iters, best_x_iters, num_iter, n_reps)

        feval = nanminimum(BestValue[num_iter, :])
        verbose && @printf("[iter = %3d, num_call = %4d] out: goal = %f\n", num_iter, num_call, feval)

        if loop > current_loop_min
            f_opt = fevals[loop-current_loop_min] # 不能是邻近的
            # isapprox(f_opt, feval; atol=f_atol) && break # _f_best收敛

            # abs(nanminimum(log10.(Mbounds ./ M))) > deps # M收缩过多
            (num_call > maxn) && break

            ineed = abs(nanminimum(best_feval_iters[num_iter-current_loop_min+1]) - nanminimum(best_feval_iters[num_iter]))
            
            if abs(log10(nanmax(ineed, 10.0^(-eps - 1)))) ≥ eps
                inner_search_started = true

                _lower = nanminimum(X_best', dims=1)
                _upper = nanmaximum(X_best', dims=1)
                lower .= max.(min.(lower, _lower[:] .- k[:]), lb) # lower and upper are updated
                upper .= min.(max.(upper, _upper[:] .+ k[:]), ub)
                Mbounds .= upper .- lower
                k .= Mbounds ./ (psize .- 1)
                Xp1 = X_best[:, indexX[1:po]]
                # println(Mbounds)

                BestX = copy(Xp1)
                x = zeros(npar, m1 * po)

                BestY = zeros(Float64, npar + 1, p1)
                BestY[1, :] .= y_best[1:p1]

                BX = best_x_iters[:, num_iter]

                num_call, feS, Index1 = perform_inner_search!(x, Xp1, BestX, BestY, BX, lower, upper, k, psize,
                    p1, m1, fun, num_call, feS)

                num_iter += 1
                BestValue[num_iter, 1:p1] .= nanminimum(BestY[Index1-npar:Index1, :], dims=1)'

                append!(best_feval_iters, nanminimum(BestValue[num_iter, 1:p1]))
                best_x_iters[:, num_iter] = BX

                n_reps1 = m1 * p1 * npar
                populate_best_value_fe!(_feval_iters, BestValue, num_iter, [n_reps, n_reps1])
                populate_each_par_fe!(_x_iters, best_x_iters, num_iter, n_reps)

                if verbose
                    feval = nanminimum(BestValue[num_iter, 1:p1])
                    @printf("[iter = %3d, num_call = %4d]  in: goal = %f\n", num_iter, num_call, feval)
                end

                X_best[:, 1:p1] .= BestX
                _lower = nanminimum(X_best', dims=1)
                _upper = nanmaximum(X_best', dims=1)

                delta_Mbounds = Mbounds * delta
                lower .= max.(lower, _lower[:] .- delta_Mbounds)
                upper .= min.(upper, _upper[:] .+ delta_Mbounds)
                k .= (upper .- lower) ./ (psize .- 1)

                perform_secondary_search!(x, X_best, X_worst, lower, upper, m1, p1)

                # 生成随机数 N
                N = (ub .- lb) .* rand(npar, m1 * p1) .+ lb
                x[x.<lb] .= N[x.<lb]
                x[x.>ub] .= N[x.>ub]

                # 计算 y
                MM = m1 * p1
                y = Vector{Float64}(undef, MM)
                num_call, feS = calculate_goal!(y, fun, x, num_call, feS)

                # 更新 yps 和 x
                y_best[1:p1] .= nanminimum(BestY, dims=1)'
                y = vcat(y, y_best)
                x = hcat(x, X_best)

                # 对 yps 排序并更新
                y_best, indexY = sort(y), sortperm(y)
                y_best = y_best[1:p]
                xneed = abs(y_best[1] - best_feval_iters[num_iter])

                # 率定水文模型不开这个部分（因为水文模型要求精度不高，打开会使前面的等距搜索太慢了）
                # 检查是否需要更新 eps
                if abs(log10(nanmax(xneed, 10.0^(-eps - 1)))) ≥ eps && update_eps
                    eps += 1
                end

                X_best = x[:, indexY[1:p]]
                X_worst = x[:, indexY[end]]
                hit_upper_bound = falses(npar)
                hit_lower_bound = falses(npar)
                _ub = copy(ub)
                _lb = copy(lb)

                for i = 1:p
                    nx = X_best[:, i]
                    hit_upper_bound .= hit_upper_bound .| (nx .>= ub)
                    hit_lower_bound .= hit_lower_bound .| (nx .<= lb)
                    _ub[hit_upper_bound] .= min.(nx[hit_upper_bound], _ub[hit_upper_bound])
                    _lb[hit_lower_bound] .= max.(nx[hit_lower_bound], _lb[hit_lower_bound])
                end

                upper[hit_upper_bound] .= min.(_ub[hit_upper_bound] .+ k[hit_upper_bound], ub[hit_upper_bound])
                lower[hit_lower_bound] .= max.(_lb[hit_lower_bound] .- k[hit_lower_bound], lb[hit_lower_bound])

                n_reps2 = m1 * p1
                populate_best_value_fe!(_feval_iters, BestValue, num_iter, n_reps2)
                populate_each_par_fe!(_x_iters, best_x_iters, num_iter, n_reps2)
            end
        end
        push!(fevals, feval) # fevals[loop] = feval
    end

    return OptimOutput(_feval_iters, _x_iters, best_feval_iters, best_x_iters, num_call, num_iter; verbose)
end
