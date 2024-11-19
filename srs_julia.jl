module srs_julia
    # 和Python比有更新，加入update_eps参数，
    # update_eps = true: 增加精细化搜索，耗时更长，适合寻找精度要求高的测试函数最优解
    # update_eps = false:减小精细化搜索，耗时短，适合寻水文模型参数率定寻找最优解
    # params参数为目标函数的其他参数

    using Random
    using LinearAlgebra
    using Printf
    using BenchmarkTools

    export SRS

    # 定义ResultType结构体
    mutable struct ResultType
        OptimalValueFE::Vector{Float64}
        EachParFE::Matrix{Float64}
        Generation::Int
        FunctionEvaluations::Int
        FunctionEvaluationsScalar::Union{Int, Nothing}
        Besttargetfunvalue::Float64
        AE::Float64
        BestX::Vector{Float64}
        Time::Float64
    end

    # 最优化结果保存
    function optimize_result(A::Float64, BestValueFE::Vector{Float64}, EachParFE::Matrix{Float64}, 
        BY::Vector{Float64}, EachPar::Matrix{Float64}, fe::Int, s::Int, Vectorization::Bool, 
        feS::Int, t1::Float64, OV::Float64, DispValue::Bool)
        # 使用 `view` 以避免拷贝
        # s -= 1
        optimal_value_fe = A * vec(BestValueFE)
        each_par_fe = EachParFE
        best_target_fun_value = A * BY[s]
        best_x = EachPar[:, s]
        
        # 计算 AE
        ae = isnothing(OV) ? best_target_fun_value : abs(best_target_fun_value - OV)
        
        # 计算耗时
        total_time = time() - t1

        if DispValue
            @printf("目标函数最优值: \t%.4e\n", best_target_fun_value)
            println("最优参数: \t", string(best_x))
            println("迭代次数: \t", s)
            if Vectorization
                println("目标函数调用次数: \t", fe)
                println("标量目标函数调用次数: \t", feS)
            else
                println("目标函数调用次数: \t", fe)
            end
            println("耗时: \t", total_time)
        end

        # 返回结果结构体
        return ResultType(
            optimal_value_fe,
            each_par_fe,
            s,
            fe,
            Vectorization ? feS : nothing,
            best_target_fun_value,
            ae,
            best_x,
            total_time
        )
    end

    # 保存迭代过程中的最优值 BstValueFE
    function populate_best_value_fe!(BestValueFE::Vector{Float64}, BestValue::Matrix{Float64}, s::Int, n_reps::Int, vectorized::Bool)
        best_value = minimum(view(BestValue, s, :))
        values_to_add = vectorized ? [best_value] : fill(best_value, n_reps)
        append!(BestValueFE, values_to_add)
    end
    function populate_best_value_fe!(BestValueFE::Vector{Float64}, BestValue::Matrix{Float64}, s::Int, n_reps::Vector{Int}, vectorized::Bool)
        best_value = minimum(view(BestValue, s, :))
        values_to_add = vectorized ? fill(best_value, n_reps[1]) : fill(best_value, n_reps[2])
        append!(BestValueFE, values_to_add)
    end

    # 保存迭代过程中的最优参数 EachParFE
    function populate_each_par_fe!(EachParFE::Matrix{Float64}, EachPar::Matrix{Float64}, s::Int, n_reps::Int, vectorized::Bool)
        each_par_slice = EachPar[:, s]
        values_to_add = vectorized ? each_par_slice : repeat(each_par_slice, 1, n_reps)
        EachParFE = hcat(EachParFE, values_to_add)
    end

    # 定义计算函数，分别针对矢量化和非矢量化场景
    function compute_y_vectorized!(y::Vector{Float64}, A::Float64, ObjectiveFunction::Function, x::Matrix{Float64}, fe::Int, feS::Int, mm::Int, params::Tuple)
        # 矢量化实现，直接操作矩阵乘法
        y .= A * ObjectiveFunction(x', params...)
        fe += 1
        feS += mm
        return y, fe, feS
    end

    function compute_y_non_vectorized!(y::Vector{Float64}, A::Float64, ObjectiveFunction::Function, x::Matrix{Float64}, fe::Int, feS::Int, mm::Int, params)
        # 非矢量化实现，逐列计算
        @inbounds for iii in 1:mm
            y[iii] = A * ObjectiveFunction(x[:, iii], params...)
            fe += 1
        end
        return y, fe, feS
    end

    # 主函数
    function SRS(ObjectiveFunction::Function, n::Int, boundsbegin::Vector{Float64}, boundsend::Vector{Float64}; p::Int=3, sp::Union{Nothing, Int}=nothing, deps::Int=12, 
        delta::Float64=0.01, Vectorization::Bool=false, num::Int=0, MAX::Bool=true, OptimalValue::Union{Nothing, Float64}=nothing, DispValue::Bool=true, DispProcess::Bool=false, 
        ObjectiveLimit::Union{Nothing, Float64}=nothing, eps::Int=4, update_eps::Bool=true, ShortLambda::Float64=0.02, LongLambda::Float64=0.2, InitialLt::Int=3, Lt::Int=2, params::Tuple=())
        #=
        n:  The dimension of the objective function.
        boundsbegin: The lower bound of the parameter to be determined.
        boundsend: The upper bound of the parameter to be determined
        *other parameters:
        | name          | type    | defult      | describe
        | p             | int     | 3           | p is the key parameter, and the value is generally 3-20,
        |               |         |             | which needs to be given according to the specific situation
        | sp            | int     | sp=p(p<=5)  | Its range is [3, p]
        |               |         | sp=5(5<p<12)|
        |               |         | sp=12(p<=12)|
        | deps          | float   | 12          | Its range is (0, +infty),
        |               |         |             | It is a key parameter for adjusting the precision,
        |               |         |             | The larger the value, the higher the precision and the longer the time
        | delta         | float   | 0.01        | Its range is (0, 0.5),
        |               |         |             | It is a key parameter for adjusting the precision,
        |               |         |             | the larger the value, the higher the precision and the longer the time
        | Vectorization | bool    | false       | Whether the objective function satisfies the vectorization condition
        | num           | int     | 1000        | if Vectorization=True: num=1000 else: num=10000 (defult).
        |               |         | 10000       | The key parameter, representing the maximum number of
        |               |         |             | times the target function is called. When testing, the accuracy
        |               |         |             | can be improved by increasing num.
        | MAX           | bool    | true        | Whether to find the maximum value of the objective function.
        | OptimalValue  | float   | None        | The optimal value of the objective function.
        | ObjectiveLimit| float   | None        | When the optimal value is known, the algorithm terminates
        |               |         |             | within ObjectiveLimit of the optimal value.
        | eps           | Int     | 4           | Its range is (0, +infty),
        |               |         |             | it is not critical, and adjustment is not recommended.
        | update_eps    | bool    | true        | Whether or not to update eps to do refined search parameters.
        |               |         |             | Generally, it can be “false” for model parameter calibration.
        | ShortLambda   | float   | 0.02        | Its range is (0, 0.1),
        |               |         |             | it is not critical, and adjustment is not recommended.
        | LongLambda    | float   | 0.2         | Its range is (0.1, 1),
        |               |         |             | it is not critical, and adjustment is not recommended.
        | InitialLt     | int     | 3           | Its range is (0, 10),
        |               |         |             | it is not critical, and adjustment is not recommended.
        | Lt            | int     | 2           | Its range is (0, 10),
        |               |         |             | it is not critical, and adjustment is not recommended.
        | params        | Tuple   | ()          | ObjectiveFunction‘s parameters
        =#
        t1 = time()
        compute_y = Vectorization ? compute_y_vectorized! : compute_y_non_vectorized!
        if !isa(ObjectiveFunction, Function)
            throw(ArgumentError("ObjectiveFunction must be a function"))
        end
        if isnothing(sp)
            sp = p < 5 ? p : (p < 12 ? 5 : 12)
        end
        if isnothing(ObjectiveLimit)
            OLindex = false
        else
            OLindex = true
        end
        p1 = sp
        OV = OptimalValue
        
        # 根据是否最小化或最大化调整A
        A = MAX == true ? -1.0 : 1.0

        # 根据是否向量化调整num
        if num == 0
            num = Vectorization ? 1000 : 10000
        end
        
        # 初始化参数
        n1 = 3 * n + 3
        m1 = Int(max(floor(Int, n1 * p / sp) + 1, 9))
        # popsize = Int(m1 * sp * ones(Int, n, 1))
        psize = m1 * ones(Int, n, 1)
        Mbounds = boundsend .- boundsbegin
        M = boundsend .- boundsbegin
        BE = copy(boundsend)
        BD = copy(boundsbegin)
        fe = 0
        feS = 0
        k = (boundsend .- boundsbegin) ./ (psize .- 1)
        s = 0
        Index = 0
        MM = m1 * p
        
        # 初始化解空间
        x = (boundsend .+ boundsbegin) ./ 2 .+ ((boundsend .- boundsbegin) .* (rand(n, MM) .- 1) ./ 2)
        y = Vector{Float64}(undef, MM)

        y, fe, feS = compute_y(y, A, ObjectiveFunction, x, fe, feS, MM, params)
        
        BestValueFE = minimum(y)
        EachParFE = x[:, argmin(y)]

        yps, indexY = sort(y), sortperm(y)
        yps = yps[1:p]
        Xp = x[:, indexY[1:p]]
        Xb = x[:, indexY[end]]
        
        EachPar = zeros(Float64, n, num)
        BestValue = zeros(Float64, num, p)
        neps = eps
        sss = 0
        n_reps = n1 * p
        BY = Float64[]
        BestValueFE = Float64[]
        EachParFE = Matrix{Float64}(undef, size(EachPar, 1), n_reps)
        
        # 主循环
        while true
            lambda, lt_val = (eps > neps + 2) ? (ShortLambda, copy(Lt)) : (LongLambda, copy(Lt))
            if sss == 0
                lt_val = InitialLt
            end
            
            x = zeros(Float64, n, n_reps)
            Bb =  repeat(BD, 1, n1)
            Be =  repeat(BE, 1, n1)

            lam_Mbounds = lambda * Mbounds
            lam_Mbounds_IN = lam_Mbounds .* Diagonal(ones(n))
            # 在搜索过程中更新解
            for i in 1:p
                r1 = 2 .* rand(Bool, n, n) .- 1
                XPi = repeat(Xp[:, i], 1, n)
                xx1 = XPi .+ lam_Mbounds_IN
                xx2 = XPi .- lam_Mbounds_IN
                xx3 = XPi .- lam_Mbounds .* r1
                xb1 = (Xp[:, i] .+ Xb) ./ 2
                xb2 = 2 * Xb .- Xp[:, i]
                xb3 = 2 * Xp[:, i] .- Xb
                xx = hcat(xx1, xx2, xb1, xb2, xb3, xx3)
                
                xx .= clamp.(xx, Bb, Be)
                x[:, i:p:n1*p] .= xx
            end
            
            MM = n1 * p
            y = Vector{Float64}(undef, MM)
            y, fe, feS = compute_y(y, A, ObjectiveFunction, x, fe, feS, MM, params)

            for i in 1:p
                yp = copy(y[i:p:end])
                yp = vcat(yp, yps[i])
                indexY = argmin(yp)
                # println(yp[indexY])
                yps[i] = copy(yp[indexY])
                xp = hcat(x[:, i:p:end], Xp[:, i])
                Xp[:, i] .= xp[:, indexY]
            end
            
            indexYb = argmax(y)
            Xb .= x[:, indexYb]
            s += 1
            
            Index += 1
            BestValue[s, :] .= yps
            append!(BY, minimum(yps))
            indexX = sortperm(yps)
            sort!(yps)
            EachPar[:, s] = Xp[:, indexX[1]]

            populate_best_value_fe!(BestValueFE, BestValue, s, n_reps, Vectorization)
            populate_each_par_fe!(EachParFE, EachPar, s, n_reps, Vectorization)
            
            # s += 1

            if DispProcess
                println(A * minimum(BestValue[s, :]), "\tout")
            end
            
            # println(OLindex && !isnothing(OV))
            if OLindex && !isnothing(OV)
                # println(minimum(BestValue[s, :]), "abcde")
                if abs(OV*A - minimum(BestValue[s, :])) < abs(ObjectiveLimit)
                    break
                end
            end
        
            if Index > lt_val
                if abs(minimum(log10.(Mbounds ./ M))) > deps || fe > num
                    break
                end
                ineed = abs(minimum(BY[s - lt_val+1]) - minimum(BY[s]))
                if abs(log10(max(ineed, 10.0 ^ (-eps - 1)))) ≥ eps
                    sss = 1
                    bb = minimum(Xp', dims=1)
                    be = maximum(Xp', dims=1)
                    boundsbegin .= max.(min.(boundsbegin, bb[:] .- k[:]), BD)
                    boundsend .= min.(max.(boundsend, be[:] .+ k[:]), BE)
                    k .= (boundsend .- boundsbegin) ./ (psize .- 1)
                    Mbounds .= boundsend .- boundsbegin
                    x = zeros(n, m1 * sp)
                    Xp1 = Xp[:, indexX[1:sp]]
                    # println(Mbounds)

                    BestX = copy(Xp1)
                    maxpsize = maximum(psize)
                    for i in 1:p1
                        x[:, (i-1)*maxpsize+1:i*maxpsize] .= repeat(Xp1[:, i] , 1, maxpsize)
                    end
                    
                    Pi = zeros(Int, n, p1)
                    for i = 1:p1
                        Pi[:, i] .= randperm(n)
                    end

                    LL = zeros(Float64, n, maxpsize)
                    for i in 1:n  # 遍历 n 行
                        LL[i, :] .= boundsbegin[i] .+ k[i] .* (0:maxpsize-1)  # 填充每行
                    end
                    
                    Index1 = 1
                    BestY = zeros(Float64, n+1, p1)
                    BestY[1, :] .= yps[1:p1]
                    BX = EachPar[:, s]
                    MM = m1 * p1
                    y = Vector{Float64}(undef, MM)
                    for i = 1:n
                        for j = 1:p1
                            # 更新 x 矩阵的部分
                            xneed = LL[Pi[i, j], 1:maxpsize-1] + (k[Pi[i, j]] * rand(1, psize[Pi[i, j]] - 1))[:]
                            x[Pi[i, j], (j-1) * psize[Pi[i, j]] + 2 : j * psize[Pi[i, j]]] .= xneed
                            x[Pi[i, j], (j-1) * psize[Pi[i, j]] + 1] = x[Pi[i, j], 1] + k[Pi[i, j]] * (2*rand()-1)
                            
                            # 处理边界情况
                            if x[Pi[i, j], (j-1) * psize[Pi[i, j]] + 1] < boundsbegin[Pi[i, j]]
                                x[Pi[i, j], (j-1) * psize[Pi[i, j]] + 1] = 
                                    boundsbegin[Pi[i, j]] + k[Pi[i, j]] * rand()
                            elseif x[Pi[i, j], (j-1) * psize[Pi[i, j]] + 1] > boundsend[Pi[i, j]]
                                x[Pi[i, j], (j-1) * psize[Pi[i, j]] + 1] = 
                                    boundsend[Pi[i, j]] - k[Pi[i, j]] * rand()
                            end
                        end
                        
                        
                        y, fe, feS = compute_y(y, A, ObjectiveFunction, x, fe, feS, MM, params)
                        Index1 += 1
                    
                        for j = 1:p1
                            nash, index = findmin(y[(j-1) * psize[Pi[i, j]] + 1 : j * psize[Pi[i, j]]])
                            BestY[Index1, j] = nash
                            
                            x[Pi[i, j], (j-1) * psize[Pi[i, j]] + 1 : j * psize[Pi[i, j]]] .= 
                                x[Pi[i, j], (j-1) * psize[Pi[i, j]] + index] * ones(Float64, maxpsize)
                    
                            if nash == minimum(BestY[1:Index1, j])
                                BestX[:, j] .= x[:, (j-1) * psize[Pi[i, j]] + index]
                            end
                            if nash == minimum([minimum(BestY[1:Index1-1, :]), minimum(BestY[Index1, 1:j])])
                                BX .= x[:, (j-1) * psize[Pi[i, j]] + index]
                            end
                        end
                    end
                    
                    s += 1
                    BestValue[s, 1:p1] .= minimum(BestY[Index1-n:Index1, :], dims=1)'
                    append!(BY, minimum(BestValue[s, 1:p1]))
                    EachPar[:, s] = BX

                    n_reps1 = m1 * p1 * n
                    populate_best_value_fe!(BestValueFE, BestValue, s, [n_reps, n_reps1], Vectorization)
                    populate_each_par_fe!(EachParFE, EachPar, s, n_reps, Vectorization)
                    
                    if DispProcess
                        println(A*minimum(BestValue[s, 1:p1]), "\tin")
                    end
                    
                    Xp[:, 1:p1] .= BestX
                    bb = minimum(Xp', dims=1)
                    be = maximum(Xp', dims=1)
                    # println(bb[:])
                    delta_Mbounds = Mbounds * delta
                    boundsbegin .= max.(boundsbegin, bb[:] .- delta_Mbounds)
                    boundsend .= min.(boundsend, be[:] .+ delta_Mbounds)
                    k .= (boundsend .- boundsbegin) ./ (psize .- 1)

                    pp = m1
                    x = zeros(n, m1 * p1)
                    
                    for j = 1:p1
                        ra1 = 1:p1  # 创建 0 到 p1-1 的数组
                        ra1 = collect(ra1) # 将 ra1 转换为可变的数组类型 (Vector)
                        shuffle!(ra1)  # 打乱 ra1 数组
                        ra1 = Int64.(ra1)  # 将 ra1 转换为 Int64 类型

                        ra = Int64.([mod(j, p1+1), mod(j+1, p1+1)])  # 计算 ra 数组
                        if ra[2] == 0
                            ra .= Int64.([j, 1])
                        end
  
                        xx = min.(Xp[:, j] .- boundsbegin, boundsend .- Xp[:, j]) ./ 4 
                        xxx = randn(n, pp - 9) .* xx 

                        # 更新 x 数组的某些列
                        x[:, (j-1)*pp+1:j*pp-9] .= repeat(Xp[:, j], 1, pp - 9) .+ xxx

                        x[:, j*pp - 8] .= Xp[:, ra1[3]] .- Xp[:, ra1[1]] .+ Xp[:, ra1[2]]
                        x[:, j*pp - 7] .= (2 * Xp[:, ra1[1]] .- Xp[:, ra1[3]] .- Xp[:, ra1[2]]) / 2

                        x[:, j*pp - 6] .= Xb .- (Xp[:, ra[2]] .+ Xp[:, ra[1]]) / 2
                        x[:, j*pp - 5] .= Xp[:, ra[2]] .- Xb .+ Xp[:, ra[1]]
                        x[:, j*pp - 4] .= x[:, j*pp - 5] .- (Xp[:, ra[2]] .- 2 * Xb .+ Xp[:, ra[1]]) / 2
                        x[:, j*pp - 3] .= Xb .+ (Xp[:, ra[2]] .- 2 * Xb .+ Xp[:, ra[1]]) / 4

                        x[:, j*pp - 2] .= (Xp[:, j] .+ Xb) / 2
                        x[:, j*pp - 1] .= 2 * Xb .- Xp[:, j]
                        x[:, j*pp - 0] .= 2 * Xp[:, j] .- Xb
                    end
                    
                    # 生成随机数 N
                    N = (BE .- BD) .* rand(n, m1 * p1) .+ BD
                    x[x .< BD] .= N[x .< BD]
                    x[x .> BE] .= N[x .> BE]
                    
                    # 计算 y
                    MM = m1 * p1
                    y = Vector{Float64}(undef, MM)
                    y, fe, feS = compute_y(y, A, ObjectiveFunction, x, fe, feS, MM, params)
                    
                    # 更新 yps 和 x
                    # println("ymin: ", minimum(y))
                    yps[1:p1] .= minimum(BestY, dims=1)'
                    y = vcat(y, yps)
                    # println(yps)
                    x = hcat(x, Xp)
                    
                    # 对 yps 排序并更新
                    yps, indexY = sort(y), sortperm(y)
                    yps = yps[1:p]
                    xneed = abs(yps[1] - BY[s])
                    # println("yps: ", yps[1])
                    
                    # 率定水文模型不开这个部分（因为水文模型要求精度不高，打开会使前面的等距搜索太慢了）
                    # 检查是否需要更新 eps
                    if abs(log10(max(xneed, 10.0^(-eps - 1)))) ≥ eps && update_eps
                        eps += 1
                    end
                    
                    Xp = x[:, indexY[1:p]]
                    Xb = x[:, indexY[end]]
                    heihei1 = falses(n)
                    heihei2 = falses(n)
                    nx1 = copy(BE)
                    nx2 = copy(BD)

                    for i = 1:p
                        nx = Xp[:, i]
                        heihei1 .= heihei1 .| (nx .>= BE)
                        heihei2 .= heihei2 .| (nx .<= BD)
                        nx1[heihei1] .= min.(nx[heihei1], nx1[heihei1])
                        nx2[heihei2] .= max.(nx[heihei2], nx2[heihei2])
                    end

                    boundsend[heihei1] .= min.(nx1[heihei1] .+ k[heihei1], BE[heihei1])
                    boundsbegin[heihei2] .= max.(nx2[heihei2] .- k[heihei2], BD[heihei2])
                    
                    n_reps2 = m1*p1
                    populate_best_value_fe!(BestValueFE, BestValue, s, n_reps2, Vectorization)
                    populate_each_par_fe!(EachParFE, EachPar, s, n_reps2, Vectorization)
                    # s += 1
                end
            end
        end
        
        # 返回结果
        Result = optimize_result(A, BestValueFE, EachParFE, 
        BY, EachPar, fe, s, Vectorization, feS, t1, OV, DispValue)
        return Result
    end

    # 例子
    # 定义目标函数
    function Zakharov(x::AbstractMatrix{Float64})
        n_rows, n_cols = size(x)
        y1 = zeros(Float64, n_rows)  # 初始化为零向量
        y2 = zeros(Float64, n_rows)  # 初始化为零向量

        @inbounds for i in 1:n_cols
            xi = @view x[:, i]  # 使用视图来引用 x 的第 i 列
            xi_shifted = xi .- 0.5  # 减去 0.5 的操作
            y1 .+= xi_shifted .^ 2  # 累加平方项
            y2 .+= xi_shifted .* (i + 1)  # 累加加权项
        end

        y = y1 .+ (0.5 .* y2) .^ 2 .+ (0.5 .* y2) .^ 4 .+ 1
        return y
    end
    function Zakharov(x::AbstractVector{Float64})
        x .-= 0.5
        n = length(x)

        # 计算 y1 和 y2
        y1 = sum(x .^ 2)  # 向量化计算 x[i]^2
        y2 = 0.0
        for i in 1:n
            y2 += (i + 1) * x[i]  # 直接计算 y2
        end

        # 计算目标函数值
        y2_half = 0.5 * y2
        y = y1 + y2_half^2 + y2_half^4 + 1
        return y
    end

    # # 例子
    # n = 100
    # boundsbegin = -5*ones(n)
    # boundsend = 10*ones(n)
    # result = SRS(Zakharov, n, boundsbegin, boundsend, p=3, sp=3, delta=0.01, deps=12, DispProcess=false, num=20000,
    #             Vectorization=true, DispValue=true, MAX=false, OptimalValue=1.0, ObjectiveLimit=1e-20)
end
