# y是修改地址，无需返回
function calculate_goal!(y::Vector{Float64}, f::Function, x::Matrix{Float64},
    num_call::Int, feS::Int)

    N = size(x, 2)
    @inbounds @threads for i in 1:N
        y[i] = f(x[:, i])
    end
    num_call += N
    feS += N
    return num_call, feS
end


# 保存迭代过程中的最优值 BstValueFE
function populate_best_value_fe!(BestValueFE::Vector{Float64}, BestValue::Matrix{Float64}, num_iter::Int, n_reps::Int)
    fs = nanminimum(@view BestValue[num_iter, :])
    append!(BestValueFE, fs)
end

function populate_best_value_fe!(BestValueFE::Vector{Float64}, BestValue::Matrix{Float64}, num_iter::Int, n_reps::Vector{Int})
    # fs = nanminimum(view(BestValue, num_iter, :))
    fs = nanminimum(@view BestValue[num_iter, :])
    # values_to_add = fill(fs, n_reps[2])
    append!(BestValueFE, fs)
end

# 保存迭代过程中的最优参数 EachParFE
function populate_each_par_fe!(EachParFE::Vector, EachPar::Matrix{Float64}, num_iter::Int, n_reps::Int)
    xs = EachPar[:, num_iter]
    append!(EachParFE, [xs])
end


# 执行外层搜索迭代
function search_init_X!(X, X_best, X_worst, p, n_reps, lambda, Mbounds, LB, UB)
    npar = length(Mbounds)
    lam_Mbounds = lambda * Mbounds
    lam_Mbounds_IN = lambda * Mbounds .* Diagonal(ones(npar))

    for i in 1:p
        r1 = 2 .* rand(Bool, npar, npar) .- 1
        XPi = repeat(X_best[:, i], 1, npar)

        xx1 = XPi .+ lam_Mbounds_IN    # npar
        xx2 = XPi .- lam_Mbounds_IN    # npar 
        xx3 = XPi .- lam_Mbounds .* r1 # npar

        xb1 = (X_best[:, i] .+ X_worst) ./ 2 # 1
        xb2 = 2 * X_worst .- X_best[:, i]    # 1
        xb3 = 2 * X_best[:, i] .- X_worst    # 1

        xx = hcat(xx1, xx2, xb1, xb2, xb3, xx3) # [3npar + 3, ]
        xx .= clamp.(xx, LB, UB)
        X[:, i:p:n_reps] .= xx
    end
end


# yps, X_best, X_worst = select_theta(y, x; p)
function select_optimal(y, x; p::Int)
    inds = sortperm(y)
    y_optimal = y[inds[1:p]]
    X_optimal = x[:, inds[1:p]]
    X_worst = x[:, inds[end]]
    y_optimal, X_optimal, X_worst
end


# 更新最优解
function update_best_theta!(yps, X_best, X_worst, y, x, p::Int)
    for i in 1:p
        yp = copy(y[i:p:end])
        yp = vcat(yp, yps[i])
        indexY = argmin(yp)
        yps[i] = copy(yp[indexY]) # update

        xp = hcat(x[:, i:p:end], X_best[:, i])
        X_best[:, i] .= xp[:, indexY] # update
    end

    indexYb = argmax(y)
    X_worst .= x[:, indexYb]
end

# function select_optimal(yps, X_best)
#   i = sortperm(yps)[1]
#   yps[i], X_best[:, i]
# end


# 执行内层精细搜索
# not used
# num_call, feS, Index1 = perform_inner_search!(x, Xp1, BestX, BestY, BX, lower, upper, k, psize, p1, m1, fun)
function perform_inner_search!(x, Xp1, BestX, BestY, BX, lower, upper, k, psize, p1, m1, fun, num_call, feS)
    npar = length(lower)

    maxpsize = nanmaximum(psize)
    for i in 1:p1
        x[:, (i-1)*maxpsize+1:i*maxpsize] .= repeat(Xp1[:, i], 1, maxpsize)
    end

    Pi = zeros(Int, npar, p1)
    LL = zeros(Float64, npar, maxpsize)

    for i = 1:p1
        Pi[:, i] .= randperm(npar)
    end

    for i in 1:npar
        LL[i, :] .= lower[i] .+ k[i] .* (0:maxpsize-1)
    end

    Index1 = 1

    MM = m1 * p1
    y = Vector{Float64}(undef, MM)

    for i = 1:npar
        for j = 1:p1
            # 更新 x 矩阵的部分
            _i = Pi[i, j]
            xneed = LL[_i, 1:maxpsize-1] + (k[_i]*rand(1, psize[_i] - 1))[:]
            x[_i, (j-1)*psize[_i]+2:j*psize[_i]] .= xneed
            x[_i, (j-1)*psize[_i]+1] = x[_i, 1] + k[_i] * (2 * rand() - 1)

            # 处理边界情况
            if x[_i, (j-1)*psize[_i]+1] < lower[_i]
                x[_i, (j-1)*psize[_i]+1] =
                    lower[_i] + k[_i] * rand()
            elseif x[_i, (j-1)*psize[_i]+1] > upper[_i]
                x[_i, (j-1)*psize[_i]+1] =
                    upper[_i] - k[_i] * rand()
            end
        end

        num_call, feS = calculate_goal!(y, fun, x, num_call, feS)
        Index1 += 1

        for j = 1:p1
            _i = Pi[i, j]
            nash, index = findmin(y[(j-1)*psize[_i]+1:j*psize[_i]])
            BestY[Index1, j] = nash

            x[_i, (j-1)*psize[_i]+1:j*psize[_i]] .=
                x[_i, (j-1)*psize[_i]+index] * ones(Float64, maxpsize)

            if nash == nanminimum(BestY[1:Index1, j])
                BestX[:, j] .= x[:, (j-1)*psize[_i]+index]
            end
            if nash == nanminimum([nanminimum(BestY[1:Index1-1, :]), nanminimum(BestY[Index1, 1:j])])
                BX .= x[:, (j-1)*psize[_i]+index]
            end
        end
    end
    return num_call, feS, Index1
end


# 执行二次搜索
function perform_secondary_search!(x, X_best, X_worst, lower, upper, m1::Int, p1::Int)
    npar = length(lower)
    pp = m1

    for j = 1:p1
        ra1 = 1:p1         # 创建 0 到 p1-1 的数组
        ra1 = collect(ra1) # 将 ra1 转换为可变的数组类型 (Vector)
        shuffle!(ra1)      # 打乱 ra1 数组

        j1 = mod(j, p1 + 1)
        j2 = nanmax(1, mod(j + 1, p1 + 1))
        ra = j1, j2  # 计算 ra 数组

        xx = min.(X_best[:, j] .- lower, upper .- X_best[:, j]) ./ 4
        xxx = randn(npar, pp - 9) .* xx

        # 更新 x 数组的某些列
        x[:, (j-1)*pp+1:j*pp-9] .= repeat(X_best[:, j], 1, pp - 9) .+ xxx

        x[:, j*pp-8] .= X_best[:, ra1[3]] .- X_best[:, ra1[1]] .+ X_best[:, ra1[2]]
        x[:, j*pp-7] .= (2 * X_best[:, ra1[1]] .- X_best[:, ra1[3]] .- X_best[:, ra1[2]]) / 2

        x[:, j*pp-6] .= X_worst .- (X_best[:, ra[2]] .+ X_best[:, ra[1]]) / 2
        x[:, j*pp-5] .= X_best[:, ra[2]] .- X_worst .+ X_best[:, ra[1]]
        x[:, j*pp-4] .= x[:, j*pp-5] .- (X_best[:, ra[2]] .- 2 * X_worst .+ X_best[:, ra[1]]) / 2
        x[:, j*pp-3] .= X_worst .+ (X_best[:, ra[2]] .- 2 * X_worst .+ X_best[:, ra[1]]) / 4

        x[:, j*pp-2] .= (X_best[:, j] .+ X_worst) / 2
        x[:, j*pp-1] .= 2 * X_worst .- X_best[:, j]
        x[:, j*pp-0] .= 2 * X_best[:, j] .- X_worst
    end
end
