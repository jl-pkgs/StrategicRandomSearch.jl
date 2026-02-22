# y是修改地址，无需返回
function calculate_goal!(Y::Vector{Float64}, f::Function, X::Matrix{Float64},
    num_call::Int=0)

    N = size(X, 2)
    @inbounds @threads for i in 1:N
        Y[i] = f(X[:, i]) # TODO: 多线程存在data race
    end
    num_call += N
    return num_call
end


"""
保存每次迭代的最优历史（简要维度）。

- `fevals_calls`    : [ncalls, 1   ]，每次call最优f值
- `x_calls`         : [ncalls, npar]，每次call最优x

- `fevals_p`        : [p, 1] 或 [p1, 1]，当前迭代精英f值

- `feval_iters`     : [niter, 1   ]，每次iter最优f值
- `x_iter`          : [npar, 1]，当前迭代最优x
"""
function push_best_history!(
    fevals_calls::Vector{Float64}, x_calls::Vector,
    feval_iters::Vector{Float64},
    val::Float64, x_iter::AbstractVector{Float64})

    push!(feval_iters, val)
    push!(fevals_calls, val)
    push!(x_calls, copy(x_iter))
end


# 执行外层搜索迭代
# X_cand_out: [n_param, 3(n_param + 1) * p]，每次迭代生成的候选解
function shuffle_cand!(X_cand_out, X_opt, X_worst, Mbounds, lb, ub; p::Int, λ::Float64)
    n_ensemble = 3 * (length(lb) + 1) # 内部已写死
    LB = repeat(lb, 1, n_ensemble) # [n_param, n_ensemble]
    UB = repeat(ub, 1, n_ensemble)

    n_param = length(Mbounds)
    lam_Mbounds = λ * Mbounds # 洗牌强度
    lam_Mbounds_IN = λ * Mbounds .* Diagonal(ones(n_param))

    for i in 1:p
        r1 = 2 .* rand(Bool, n_param, n_param) .- 1
        XPi = repeat(X_opt[:, i], 1, n_param)

        xx1 = XPi .+ lam_Mbounds_IN    # n_param
        xx2 = XPi .- lam_Mbounds_IN    # n_param 
        xx3 = XPi .- lam_Mbounds .* r1 # n_param

        xb1 = (X_opt[:, i] .+ X_worst) ./ 2 # 1
        xb2 = 2 * X_worst .- X_opt[:, i]    # 1
        xb3 = 2 * X_opt[:, i] .- X_worst    # 1

        xx = hcat(xx1, xx2, xb1, xb2, xb3, xx3) # [n_param, 3npar + 3]
        xx .= clamp.(xx, LB, UB)
        X_cand_out[:, i:p:end] .= xx
    end
end


# UPDATE: lower, upper, search_steps
function update_bounds_and_steps!(
    X_opt::AbstractMatrix{T}, lower::AbstractVector{T}, upper::AbstractVector{T}, Mbounds::AbstractVector{T},
    lb::AbstractVector{T}, ub::AbstractVector{T},
    search_steps::AbstractVecOrMat{T}, search_param_sizes::AbstractVecOrMat{<:Integer};
    mode::Symbol,
    delta::T=zero(T), update_Mbounds::Bool=true) where {T<:AbstractFloat}

    opt_lower = nanminimum(X_opt, dims=2)[:]
    opt_upper = nanmaximum(X_opt, dims=2)[:]

    # 约定维度:
    # - search_param_sizes: n_param x 1（或等价的一维向量）
    # - search_steps: 与 search_param_sizes 同维度
    step_vec = search_steps[:]
    if mode === :expand
        lower .= max.(min.(lower, opt_lower .- step_vec), lb)
        upper .= min.(max.(upper, opt_upper .+ step_vec), ub)
    elseif mode === :shrink
        delta_Mbounds = (Mbounds*delta)[:]
        lower .= max.(lower, opt_lower .- delta_Mbounds)
        upper .= min.(upper, opt_upper .+ delta_Mbounds)
    else
        throw(ArgumentError("mode must be :expand or :shrink"))
    end

    update_Mbounds && (Mbounds .= upper .- lower)
    search_steps .= (upper .- lower) ./ (search_param_sizes .- 1)
    return nothing
end


function adjust_bounds_for_hits!(
    X_opt::AbstractMatrix{T}, lower::AbstractVector{T}, upper::AbstractVector{T},
    lb::AbstractVector{T}, ub::AbstractVector{T};
    search_steps::AbstractVecOrMat{T}, p::Int) where {T<:AbstractFloat}

    n_param = length(lower)
    hit_upper_bound = falses(n_param)
    hit_lower_bound = falses(n_param)
    ub_hit = copy(ub)
    lb_hit = copy(lb)
    step_vec = search_steps[:]

    for i = 1:p
        x = X_opt[:, i]
        hit_upper_bound .= hit_upper_bound .| (x .>= ub)
        hit_lower_bound .= hit_lower_bound .| (x .<= lb)
        ub_hit[hit_upper_bound] .= min.(x[hit_upper_bound], ub_hit[hit_upper_bound])
        lb_hit[hit_lower_bound] .= max.(x[hit_lower_bound], lb_hit[hit_lower_bound])
    end
    upper[hit_upper_bound] .= min.(ub_hit[hit_upper_bound] .+ step_vec[hit_upper_bound], ub[hit_upper_bound])
    lower[hit_lower_bound] .= max.(lb_hit[hit_lower_bound] .- step_vec[hit_lower_bound], lb[hit_lower_bound])
    return nothing
end


function select_optimal(y, x; p::Int)
    inds = sortperm(y)
    y_opt = y[inds[1:p]]
    X_opt = x[:, inds[1:p]]
    X_worst = x[:, inds[end]]
    y_opt, X_opt, X_worst
end


# 更新最优解
function update_optimal!(y_opt, X_opt, X_worst, y_cand, x_cand, p::Int)
    for i in 1:p
        yp = vcat(y_cand[i:p:end], y_opt[i])
        xp = hcat(x_cand[:, i:p:end], X_opt[:, i])

        iopt = argmin(yp)
        y_opt[i] = yp[iopt] # update
        X_opt[:, i] .= xp[:, iopt] # update
    end
    X_worst .= x_cand[:, argmax(y_cand)]
end

# 执行内层精细搜索
function perform_inner_search!(X_cand, BestX, BestY, BX, lower, upper,
    search_steps, search_param_sizes, search_size, p1, fn, num_call)

    n_param = length(lower)

    max_param_grid_size = nanmaximum(search_param_sizes)
    for i in 1:p1
        X_cand[:, (i-1)*max_param_grid_size+1:i*max_param_grid_size] .= repeat(BestX[:, i], 1, max_param_grid_size)
    end

    Pi = zeros(Int, n_param, p1)
    LL = zeros(Float64, n_param, max_param_grid_size)

    for i = 1:p1
        Pi[:, i] .= randperm(n_param)
    end

    for i in 1:n_param
        LL[i, :] .= lower[i] .+ search_steps[i] .* (0:max_param_grid_size-1)
    end

    n_cand = search_size * p1
    y = Vector{Float64}(undef, n_cand)

    for i = 1:n_param
        for j = 1:p1
            # 更新 x 矩阵的部分
            _i = Pi[i, j]
            xneed = LL[_i, 1:max_param_grid_size-1] + (search_steps[_i]*rand(1, search_param_sizes[_i] - 1))[:]
            X_cand[_i, (j-1)*search_param_sizes[_i]+2:j*search_param_sizes[_i]] .= xneed
            X_cand[_i, (j-1)*search_param_sizes[_i]+1] = X_cand[_i, 1] + search_steps[_i] * (2 * rand() - 1)

            # 处理边界情况
            if X_cand[_i, (j-1)*search_param_sizes[_i]+1] < lower[_i]
                X_cand[_i, (j-1)*search_param_sizes[_i]+1] =
                    lower[_i] + search_steps[_i] * rand()
            elseif X_cand[_i, (j-1)*search_param_sizes[_i]+1] > upper[_i]
                X_cand[_i, (j-1)*search_param_sizes[_i]+1] =
                    upper[_i] - search_steps[_i] * rand()
            end
        end

        num_call = calculate_goal!(y, fn, X_cand, num_call)

        for j = 1:p1
            _i = Pi[i, j]
            _min, index = findmin(y[(j-1)*search_param_sizes[_i]+1:j*search_param_sizes[_i]])
            BestY[i+1, j] = _min

            X_cand[_i, (j-1)*search_param_sizes[_i]+1:j*search_param_sizes[_i]] .=
                X_cand[_i, (j-1)*search_param_sizes[_i]+index] * ones(Float64, max_param_grid_size)

            if _min == nanminimum(BestY[1:i+1, j])
                BestX[:, j] .= X_cand[:, (j-1)*search_param_sizes[_i]+index]
            end
            if _min == nanminimum([nanminimum(BestY[1:i, :]), nanminimum(BestY[i+1, 1:j])])
                BX .= X_cand[:, (j-1)*search_param_sizes[_i]+index]
            end
        end
    end
    return num_call
end


# 执行二次搜索
function perform_secondary_search!(X_cand, X_best, X_worst, lower, upper, search_block_size::Int, p1::Int)
    n_param = length(lower)
    pp = search_block_size

    for j = 1:p1
        ra1 = 1:p1         # 创建 0 到 p1-1 的数组
        ra1 = collect(ra1) # 将 ra1 转换为可变的数组类型 (Vector)
        shuffle!(ra1)      # 打乱 ra1 数组

        j1 = mod(j, p1 + 1)
        j2 = nanmax(1, mod(j + 1, p1 + 1))
        ra = j1, j2  # 计算 ra 数组

        xx = min.(X_best[:, j] .- lower, upper .- X_best[:, j]) ./ 4
        xxx = randn(n_param, pp - 9) .* xx

        # 更新 x 数组的某些列
        X_cand[:, (j-1)*pp+1:j*pp-9] .= repeat(X_best[:, j], 1, pp - 9) .+ xxx

        X_cand[:, j*pp-8] .= X_best[:, ra1[3]] .- X_best[:, ra1[1]] .+ X_best[:, ra1[2]]
        X_cand[:, j*pp-7] .= (2 * X_best[:, ra1[1]] .- X_best[:, ra1[3]] .- X_best[:, ra1[2]]) / 2

        X_cand[:, j*pp-6] .= X_worst .- (X_best[:, ra[2]] .+ X_best[:, ra[1]]) / 2
        X_cand[:, j*pp-5] .= X_best[:, ra[2]] .- X_worst .+ X_best[:, ra[1]]
        X_cand[:, j*pp-4] .= X_cand[:, j*pp-5] .- (X_best[:, ra[2]] .- 2 * X_worst .+ X_best[:, ra[1]]) / 2
        X_cand[:, j*pp-3] .= X_worst .+ (X_best[:, ra[2]] .- 2 * X_worst .+ X_best[:, ra[1]]) / 4

        X_cand[:, j*pp-2] .= (X_best[:, j] .+ X_worst) / 2
        X_cand[:, j*pp-1] .= 2 * X_worst .- X_best[:, j]
        X_cand[:, j*pp-0] .= 2 * X_best[:, j] .- X_worst
    end
end
