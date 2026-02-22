guess_po(p::Int) = p < 5 ? p : (p < 12 ? 5 : 12)


function SRS(
    f::Function, lower::Vector{Float64}, upper::Vector{Float64}, args...;
    maxn::Int=1000, seed::Int=0, verbose=true,
    p::Int=3,                 # Fig 2a, p optimal points (purple squares and yellow squares)
    po::Int=guess_po(p),      # Fig 2a, po optimal points (yellow squares), 精英中的精英
    # deps::Int=12,           # x 空间收缩终止条件
    delta::Float64=0.01,      # 区间收缩因子, `delta_Mbounds = Mbounds * delta`
    f_atol::Float64=1e-5,
    eps::Int=4,               # 进入/维持某些全局跳出逻辑
    update_eps::Bool=true,
    λ_short::Float64=0.02, λ_long::Float64=0.2,
    init_loop_min::Int=3, loop_min::Int=2,
    kw...)

    Random.seed!(seed) # make the result reproducible
    fun(x) = f(x, args...; kw...)

    f_opt::Float64 = NaN
    p1 = po

    # 初始化参数
    n_param = length(lower)
    n_ensemble = 3 * n_param + 3
    n_pop_out = n_ensemble * p       # 总参数，每次循环，总参数个数

    search_size = Int(nanmax(floor(Int, n_ensemble * p / po) + 1, 9))
    search_param_sizes = search_size * ones(Int, n_param, 1) # n_param x 1, 每个参数维度的采样点数
    # popsize = Int(search_size * sp * ones(Int, n, 1))
    Mbounds = upper .- lower

    ## 静态变量
    M = upper .- lower
    ub = copy(upper)
    lb = copy(lower)
    LB = repeat(lb, 1, n_ensemble)
    UB = repeat(ub, 1, n_ensemble)

    search_steps = M ./ (search_param_sizes .- 1) # n_param x 1, 与 search_param_sizes 同维度
    MM = search_size * p

    num_call = 0
    num_iter = 0

    # 初始化解空间
    x = (upper .+ lower) ./ 2 .+ (M .* (rand(n_param, MM) .- 1) ./ 2)
    y = Vector{Float64}(undef, MM)

    num_call = calculate_goal!(y, fun, x, num_call)
    y_opt, X_opt, X_worst = select_optimal(y, x; p) # Optimal: 精英点

    best_x_iters = zeros(Float64, n_param, maxn)
    best_fvals_p = zeros(Float64, maxn, p) # 前n个作为候选

    neps = eps
    X1 = zeros(Float64, n_param, n_pop_out)

    best_feval_iters = Float64[]
    _x_iters = []
    _feval_iters = Float64[]

    _fevals_loops = Float64[] # 每次loop的最优值
    inner_search_started = false
    loop = 0

    # 主循环
    while true
        λ = (eps > neps + 2) ? λ_short : λ_long
        current_loop_min = inner_search_started ? loop_min : init_loop_min
        search_init_X!(X1, X_opt, X_worst, p, n_pop_out, λ, Mbounds, LB, UB)

        y = Vector{Float64}(undef, n_pop_out)
        num_call = calculate_goal!(y, fun, X1, num_call)
        update_best_theta!(y_opt, X_opt, X_worst, y, X1, p) # update yps, Xp, Xb

        num_iter += 1
        loop += 1

        best_fvals_p[num_iter, :] .= y_opt # 这里近记录了一次表现最好的

        indexX = sortperm(y_opt)
        sort!(y_opt)

        append!(best_feval_iters, nanminimum(y_opt)) # BY 记录的是 最佳
        best_x_iters[:, num_iter] = X_opt[:, indexX[1]] # 做优的一个

        push_best_history!(_feval_iters, _x_iters, best_fvals_p, best_x_iters, num_iter)

        _feval = nanminimum(best_fvals_p[num_iter, :])
        verbose && @printf("[iter = %3d, num_call = %4d] out: goal = %f\n", num_iter, num_call, _feval)

        if loop > current_loop_min
            f_opt = _fevals_loops[loop-current_loop_min] # 不能是邻近的
            # isapprox(f_opt, feval; atol=f_atol) && break # _f_best收敛

            # abs(nanminimum(log10.(Mbounds ./ M))) > deps # M收缩过多
            (num_call > maxn) && break

            ineed = abs(nanminimum(best_feval_iters[num_iter-current_loop_min+1]) - nanminimum(best_feval_iters[num_iter]))

            if abs(log10(nanmax(ineed, 10.0^(-eps - 1)))) ≥ eps
                inner_search_started = true

                update_bounds_and_steps!(X_opt, lower, upper, Mbounds, lb, ub,
                    search_steps, search_param_sizes; mode=:expand)
                Xp1 = X_opt[:, indexX[1:po]]

                BestX = copy(Xp1)
                x = zeros(n_param, search_size * po)

                BestY = zeros(Float64, n_param + 1, p1)
                BestY[1, :] .= y_opt[1:p1]

                BX = best_x_iters[:, num_iter]

                num_call, Index1 = perform_inner_search!(x, Xp1, BestX, BestY, BX, lower, upper, search_steps, search_param_sizes,
                    p1, search_size, fun, num_call)

                num_iter += 1
                best_fvals_p[num_iter, 1:p1] .= nanminimum(BestY[Index1-n_param:Index1, :], dims=1)'

                append!(best_feval_iters, nanminimum(best_fvals_p[num_iter, 1:p1]))
                best_x_iters[:, num_iter] = BX
                push_best_history!(_feval_iters, _x_iters, best_fvals_p, best_x_iters, num_iter)

                _feval = nanminimum(best_fvals_p[num_iter, 1:p1])
                verbose && @printf("[iter = %3d, num_call = %4d]  in: goal = %f\n", num_iter, num_call, _feval)

                X_opt[:, 1:p1] .= BestX # 
                update_bounds_and_steps!(X_opt, lower, upper, Mbounds, lb, ub,
                    search_steps, search_param_sizes; mode=:shrink, delta=delta, update_Mbounds=false)
                perform_secondary_search!(x, X_opt, X_worst, lower, upper, search_size, p1)

                # 生成随机数 N
                N = (ub .- lb) .* rand(n_param, search_size * p1) .+ lb
                x[x.<lb] .= N[x.<lb]
                x[x.>ub] .= N[x.>ub]

                # 计算 y
                MM = search_size * p1
                y = Vector{Float64}(undef, MM)
                num_call = calculate_goal!(y, fun, x, num_call)

                # 更新 yps 和 x
                y_opt[1:p1] .= nanminimum(BestY, dims=1)'
                y = vcat(y, y_opt)
                x = hcat(x, X_opt)

                # 对 yps 排序并更新
                y_opt, indexY = sort(y), sortperm(y)
                y_opt = y_opt[1:p]
                xneed = abs(y_opt[1] - best_feval_iters[num_iter])

                # 率定水文模型不开这个部分（因为水文模型要求精度不高，打开会使前面的等距搜索太慢了）
                # 检查是否需要更新 eps
                if abs(log10(nanmax(xneed, 10.0^(-eps - 1)))) ≥ eps && update_eps
                    eps += 1
                end

                X_opt = x[:, indexY[1:p]]
                X_worst = x[:, indexY[end]]
                hit_upper_bound = falses(n_param)
                hit_lower_bound = falses(n_param)
                _ub = copy(ub)
                _lb = copy(lb)

                for i = 1:p
                    nx = X_opt[:, i]
                    hit_upper_bound .= hit_upper_bound .| (nx .>= ub)
                    hit_lower_bound .= hit_lower_bound .| (nx .<= lb)
                    _ub[hit_upper_bound] .= min.(nx[hit_upper_bound], _ub[hit_upper_bound])
                    _lb[hit_lower_bound] .= max.(nx[hit_lower_bound], _lb[hit_lower_bound])
                end

                upper[hit_upper_bound] .= min.(_ub[hit_upper_bound] .+ search_steps[hit_upper_bound], ub[hit_upper_bound])
                lower[hit_lower_bound] .= max.(_lb[hit_lower_bound] .- search_steps[hit_lower_bound], lb[hit_lower_bound])

                push_best_history!(_feval_iters, _x_iters, best_fvals_p, best_x_iters, num_iter)
            end
        end
        push!(_fevals_loops, _feval) # _fevals_loop[loop] = feval
    end

    return OptimOutput(_feval_iters, _x_iters, best_feval_iters, best_x_iters, num_call, num_iter; verbose)
end
