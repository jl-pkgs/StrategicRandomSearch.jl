guess_po(p::Int) = p < 5 ? p : (p < 12 ? 5 : 12)

"""
- `_cand`: candidate
"""
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
    fn(x) = f(x, args...; kw...)

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

    # 初始化解空间
    n_cand = search_size * p
    X_cand = (upper .+ lower) ./ 2 .+ (M .* (rand(n_param, n_cand) .- 1) ./ 2)
    y_cand = Vector{Float64}(undef, n_cand)
    num_call = calculate_goal!(y_cand, fn, X_cand) # update y_cand
    y_opt, X_opt, X_worst = select_optimal(y_cand, X_cand; p) # Optimal: 精英点

    neps = eps

    fevals_loops = Float64[] # 每次loop的最优值
    feval_iters = Float64[]
    feval_calls = Float64[]
    x_calls = []

    fevals_iters_p = zeros(Float64, maxn, p)  # 每次，保存了前p个精英
    x_iters = zeros(Float64, n_param, maxn)   # 每次，只保存了一个最优

    _yp = zeros(Float64, n_param + 1, p1)     # 

    # 外部搜索一次
    X_cand_out = zeros(Float64, n_param, n_pop_out)
    y_cand_out = Vector{Float64}(undef, n_pop_out)

    inner_search_started = false
    loop = 0
    num_iter = 0

    ineed = NaN
    f_opt = NaN

    # 主循环
    while num_call < maxn
        loop += 1

        λ = (eps > neps + 2) ? λ_short : λ_long
        current_loop_min = inner_search_started ? loop_min : init_loop_min
        search_init_X!(X_cand_out, X_opt, X_worst, p, n_pop_out, λ, Mbounds, LB, UB)

        num_call = calculate_goal!(y_cand_out, fn, X_cand_out, num_call)
        update_optimal!(y_opt, X_opt, X_worst, y_cand_out, X_cand_out, p) # update yps, Xp, Xb
        num_iter += 1

        fevals_iters_p[num_iter, :] .= y_opt # 这里近记录了一次表现最好的
        append!(feval_iters, nanminimum(y_opt)) # BY 记录的是 最佳

        i_opt = sortperm(y_opt)
        y_opt = @view y_opt[i_opt]
        x_iters[:, num_iter] = X_opt[:, i_opt[1]] # 做优的一个

        push_best_history!(feval_calls, x_calls, fevals_iters_p, x_iters, num_iter)

        feval = nanminimum(fevals_iters_p[num_iter, :])
        verbose && @printf("[iter = %3d, num_call = %4d] out: goal = %f\n", num_iter, num_call, feval)

        (num_call > maxn) && break

        if loop > current_loop_min
            f_opt = fevals_loops[loop-current_loop_min] # 不能是邻近的
            # isapprox(f_opt, feval; atol=f_atol) && break # _f_best收敛
            ineed = abs(nanminimum(feval_iters[num_iter-current_loop_min+1]) - nanminimum(feval_iters[num_iter]))
        end

        if (loop > current_loop_min) && abs(log10(nanmax(ineed, 10.0^(-eps - 1)))) ≥ eps
            # abs(nanminimum(log10.(Mbounds ./ M))) > deps # M收缩过多
            inner_search_started = true

            update_bounds_and_steps!(X_opt, lower, upper, Mbounds, lb, ub,
                search_steps, search_param_sizes; mode=:expand)
            Xp1 = X_opt[:, i_opt[1:po]]

            BestX = copy(Xp1)
            X_cand = zeros(n_param, search_size * po)

            _X = x_iters[:, num_iter]
            _yp .= 0.0
            _yp[1, :] .= y_opt[1:p1]

            num_call, Index1 = perform_inner_search!(X_cand, Xp1, BestX, _yp, _X, lower, upper,
                search_steps, search_param_sizes, search_size, p1, fn, num_call)
            num_iter += 1

            fevals_iters_p[num_iter, 1:p1] .= nanminimum(_yp[Index1-n_param:Index1, :], dims=1)'

            append!(feval_iters, nanminimum(fevals_iters_p[num_iter, 1:p1]))
            x_iters[:, num_iter] = _X
            push_best_history!(feval_calls, x_calls, fevals_iters_p, x_iters, num_iter)

            feval = nanminimum(fevals_iters_p[num_iter, 1:p1])
            verbose && @printf("[iter = %3d, num_call = %4d]  in: goal = %f\n", num_iter, num_call, feval)

            X_opt[:, 1:p1] .= BestX # 
            update_bounds_and_steps!(X_opt, lower, upper, Mbounds, lb, ub,
                search_steps, search_param_sizes; mode=:shrink, delta=delta, update_Mbounds=false)
            perform_secondary_search!(X_cand, X_opt, X_worst, lower, upper, search_size, p1)

            # 生成随机数 N
            N = (ub .- lb) .* rand(n_param, search_size * p1) .+ lb
            X_cand[X_cand.<lb] .= N[X_cand.<lb]
            X_cand[X_cand.>ub] .= N[X_cand.>ub]

            n_cand = search_size * p1
            y_cand = Vector{Float64}(undef, n_cand)
            num_call = calculate_goal!(y_cand, fn, X_cand, num_call)

            # 更新 yps 和 x
            y_opt[1:p1] .= nanminimum(_yp, dims=1)'
            y_cand = vcat(y_cand, y_opt)
            X_cand = hcat(X_cand, X_opt)

            y_opt, X_opt, X_worst = select_optimal(y_cand, X_cand; p) # second update opt

            adjust_bounds_for_hits!(X_opt, lower, upper, lb, ub; search_steps, p)
            # push_best_history!(feval_calls, x_calls, fevals_iters_p, x_iters, num_iter)

            # 率定水文模型不开这个部分（因为水文模型要求精度不高，打开会使前面的等距搜索太慢了）
            # 检查是否需要更新 eps
            xneed = abs(y_opt[1] - feval_iters[num_iter])
            if (abs(log10(nanmax(xneed, 10.0^(-eps - 1)))) ≥ eps && update_eps)
                eps += 1
            end
        end
        push!(fevals_loops, feval) # _fevals_loop[loop] = feval
    end

    return OptimOutput(feval_calls, x_calls, feval_iters, x_iters, num_call, num_iter; verbose)
end
