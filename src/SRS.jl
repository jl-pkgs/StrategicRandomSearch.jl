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
    f_atol::Float64=1e-5,       # 终止条件：全局最优值收敛
    f_atol_inner::Float64=1e-4, # 低于该阈值则进入精细搜索
    update_eps::Bool=true,
    λ_short::Float64=0.02, λ_long::Float64=0.2, # 洗牌强度 决定收敛速度
    init_loop_min::Int=3, loop_min::Int=2,
    kw...)

    Random.seed!(seed) # make the result reproducible
    fn(x) = f(x, args...; kw...)

    p1 = po

    # 初始化参数
    n_param = length(lower)
    n_ensemble = 3(n_param + 1)      # 内部已写死，这里无意义

    search_size = Int(nanmax(floor(Int, n_ensemble * p / po) + 1, 9))
    search_param_sizes = search_size * ones(Int, n_param, 1) # n_param x 1, 每个参数维度的采样点数
    # popsize = Int(search_size * sp * ones(Int, n, 1))
    Mbounds = upper .- lower

    ## 静态变量
    M = upper .- lower
    ub = copy(upper)
    lb = copy(lower)

    search_steps = M ./ (search_param_sizes .- 1) # n_param x 1, 与 search_param_sizes 同维度

    # 初始化解空间
    n_cand = search_size * p
    X_cand = (upper .+ lower) ./ 2 .+ (M .* (rand(n_param, n_cand) .- 1) ./ 2)
    y_cand = Vector{Float64}(undef, n_cand)
    num_call = calculate_goal!(y_cand, fn, X_cand) # update y_cand
    y_opt, X_opt, X_worst = select_optimal(y_cand, X_cand; p) # Optimal: 精英点

    n_eps = ceil(Int, -log10(f_atol_inner)) # f_atol: 1e-5 -> 5
    n_eps_back = n_eps

    _fevals_loops = Float64[] # 每次loop的最优值
    feval_iters = Float64[]
    feval_calls = Float64[]
    x_calls = []

    x_iters = zeros(Float64, n_param, maxn)   # 每次，只保存了一个最优
    _yp = zeros(Float64, n_param + 1, p1)     # 

    # 外部搜索一次
    n_out = 3 * (n_param + 1) * p
    X_cand_out = zeros(Float64, n_param, n_out)
    y_cand_out = Vector{Float64}(undef, n_out)

    inner_search_started = false
    loop = 0
    num_iter = 0

    ineed = NaN
    f_opt = NaN

    # 主循环
    while num_call < maxn
        loop += 1
        current_loop_min = inner_search_started ? loop_min : init_loop_min

        λ = (n_eps > n_eps_back + 2) ? λ_short : λ_long
        shuffle_cand!(X_cand_out, X_opt, X_worst, Mbounds, lb, ub; p, λ) # 洗牌

        num_call = calculate_goal!(y_cand_out, fn, X_cand_out, num_call)
        num_iter += 1

        update_optimal!(y_opt, X_opt, X_worst, y_cand_out, X_cand_out, p) # update yps, Xp, Xb

        i_opt = sortperm(y_opt)
        y_opt = y_opt[i_opt]
        # X_opt = X_opt[:, i_opt] # 不能排序，因为后续要根据位置更新边界

        _x_iter = X_opt[:, 1]      # [n_param], 当前迭代的最优解
        # x_iters[:, num_iter] = _x_iter

        _feval = nanminimum(y_opt)
        verbose && @printf("[iter = %3d, num_call = %4d] out: goal = %f\n", num_iter, num_call, _feval)

        push_best_history!(feval_calls, x_calls, feval_iters, _feval, _x_iter)

        (num_call > maxn) && break

        if loop > current_loop_min
            f_opt = _fevals_loops[loop-current_loop_min] # 不能是邻近的
            # isapprox(f_opt, feval; atol=f_atol) && break # _f_best收敛
            f_prev = nanminimum(feval_iters[num_iter-current_loop_min+1])
            f_curr = nanminimum(feval_iters[num_iter])
            ineed = abs(f_prev - f_curr)
            ## 同时增加一个结束判断
        end

        eps = 10.0^(-n_eps)
        eps_floor = eps / 10.0

        if (loop > current_loop_min) && nanmax(ineed, eps_floor) <= eps
            # abs(nanminimum(log10.(Mbounds ./ M))) > deps # M收缩过多
            inner_search_started = true

            update_bounds_and_steps!(X_opt, lower, upper, Mbounds, lb, ub,
                search_steps, search_param_sizes; mode=:expand)

            _X = X_opt[:, i_opt[1:po]]  # [n_param, po], nobug: 从全局搜索继承
            _yp .= 0.0                  # [n_param + 1, p1]
            _yp[1, :] .= y_opt[1:p1]

            X_cand .= 0.0               # [n_param, search_size * p1]
            # update: x_iter
            num_call = perform_inner_search!(X_cand, _X, _yp, _x_iter, lower, upper,
                search_steps, search_param_sizes, search_size, p1, fn, num_call)
            num_iter += 1

            feval_p = nanminimum(_yp, dims=1)[:] # [p1]
            # x_iters[:, num_iter] = _x_iter

            _feval = nanminimum(feval_p)
            verbose && @printf("[iter = %3d, num_call = %4d]  in: goal = %f\n", num_iter, num_call, _feval)

            push_best_history!(feval_calls, x_calls, feval_iters, _feval, _x_iter)

            X_opt[:, 1:p1] .= _X # nobug: 选出精英，放到固定位置
            update_bounds_and_steps!(X_opt, lower, upper, Mbounds, lb, ub,
                search_steps, search_param_sizes; mode=:shrink, delta=delta, update_Mbounds=false)
            perform_secondary_search!(X_cand, X_opt, X_worst, lower, upper, search_size, p1)

            # 生成随机数 N
            N = (ub .- lb) .* rand(n_param, search_size * p1) .+ lb
            X_cand[X_cand.<lb] .= N[X_cand.<lb]
            X_cand[X_cand.>ub] .= N[X_cand.>ub]

            n_cand = search_size * p1
            num_call = calculate_goal!(y_cand, fn, X_cand, num_call)

            # 更新 yps 和 x
            y_opt[1:p1] .= nanminimum(_yp, dims=1)'
            _y_cand = vcat(y_cand, y_opt)
            _X_cand = hcat(X_cand, X_opt)
            y_opt, X_opt, X_worst = select_optimal(_y_cand, _X_cand; p) # second update opt

            adjust_bounds_for_hits!(X_opt, lower, upper, lb, ub; search_steps, p)

            # 率定水文模型不开这个部分（因为水文模型要求精度不高，打开会使前面的等距搜索太慢了）
            # 检查是否需要更新 eps
            xneed = abs(y_opt[1] - feval_iters[num_iter])
            (nanmax(xneed, eps_floor) <= eps && update_eps) && (n_eps += 1)
        end
        push!(_fevals_loops, _feval) # _fevals_loop[loop] = feval
    end
    return OptimOutput(feval_calls, x_calls, feval_iters, x_iters, num_call, num_iter; verbose)
end
