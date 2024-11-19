module StrategicRandomSearch

using Random
using LinearAlgebra
using Printf
using Parameters


export SRS

include("OptimOutput.jl")

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
function compute_y_vectorized!(y::Vector{Float64}, f::Function, x::Matrix{Float64},
  fe::Int, feS::Int, N::Int; goal_multiplier::Int=1)
  # 矢量化实现，直接操作矩阵乘法
  y .= f(x') * goal_multiplier
  fe += 1
  feS += N
  return y, fe, feS
end

function compute_y_non_vectorized!(y::Vector{Float64}, f::Function, x::Matrix{Float64},
  fe::Int, feS::Int, N::Int; goal_multiplier::Int=1)
  @inbounds for i in 1:N
    y[i] = f(x[:, i]) * goal_multiplier
    fe += 1
  end
  feS += N
  return y, fe, feS
end

"""
## Arguments
- `f`     : The objective function to be optimized
- `lower` : The lower bound of the parameter to be determined
- `upper` : The upper bound of the parameter to be determined
- `args`  : Additional arguments to be passed to `f`

## Keyword Arguments
- `maxn`  : The maximum number of iterations
- `kw`    : Additional keyword arguments to be passed to `f`
"""
function SRS(
  f::Function, lower::Vector{Float64}, upper::Vector{Float64}, args...;
  
  maxn::Int=1000,
  p::Int=3,
  sp::Union{Nothing,Int}=nothing, 
  deps::Int=12,
  delta::Float64=0.01,
  Vectorization::Bool=false,
  MAX::Bool=false,
  OptimalValue::Float64=NaN,
  ObjectiveLimit::Float64=NaN, 
  eps::Int=4,
  DispProcess::Bool=false, verbose=true,
  update_eps::Bool=true, ShortLambda::Float64=0.02, LongLambda::Float64=0.2,
  InitialLt::Int=3, Lt::Int=2, 
  seed::Int= 0, 
  kw...)

  Random.seed!(seed) # make the result reproducible

  fun(x) = f(x, args...; kw...)
  n = length(lower)

  compute_y = Vectorization ? compute_y_vectorized! : compute_y_non_vectorized!

  if isnothing(sp)
    sp = p < 5 ? p : (p < 12 ? 5 : 12)
  end

  OLindex = !isnan(ObjectiveLimit)
  p1 = sp
  OV = OptimalValue

  # 根据是否最小化或最大化调整A
  goal_multiplier::Int = MAX == true ? -1 : 1
  
  # 初始化参数
  n1 = 3 * n + 3
  m1 = Int(max(floor(Int, n1 * p / sp) + 1, 9))
  # popsize = Int(m1 * sp * ones(Int, n, 1))
  psize = m1 * ones(Int, n, 1)
  Mbounds = upper .- lower
  M = upper .- lower
  BE = copy(upper)
  BD = copy(lower)
  fe = 0
  feS = 0
  k = (upper .- lower) ./ (psize .- 1)
  s = 0
  Index = 0
  MM = m1 * p

  # 初始化解空间
  x = (upper .+ lower) ./ 2 .+ ((upper .- lower) .* (rand(n, MM) .- 1) ./ 2)
  y = Vector{Float64}(undef, MM)

  y, fe, feS = compute_y(y, fun, x, fe, feS, MM; goal_multiplier)

  BestValueFE = minimum(y)
  EachParFE = x[:, argmin(y)]

  yps, indexY = sort(y), sortperm(y)
  yps = yps[1:p]
  Xp = x[:, indexY[1:p]]
  Xb = x[:, indexY[end]]

  EachPar = zeros(Float64, n, maxn)
  BestValue = zeros(Float64, maxn, p)
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
    Bb = repeat(BD, 1, n1)
    Be = repeat(BE, 1, n1)

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
    y, fe, feS = compute_y(y, fun, x, fe, feS, MM; goal_multiplier)

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
      println(goal_multiplier * minimum(BestValue[s, :]), "\tout")
    end

    # println(OLindex && !isnothing(OV))
    if OLindex && !isnan(OV)
      # println(minimum(BestValue[s, :]), "abcde")
      if abs(OV * goal_multiplier - minimum(BestValue[s, :])) < abs(ObjectiveLimit)
        break
      end
    end

    if Index > lt_val
      if abs(minimum(log10.(Mbounds ./ M))) > deps || fe > maxn
        break
      end
      ineed = abs(minimum(BY[s-lt_val+1]) - minimum(BY[s]))
      if abs(log10(max(ineed, 10.0^(-eps - 1)))) ≥ eps
        sss = 1
        bb = minimum(Xp', dims=1)
        be = maximum(Xp', dims=1)
        lower .= max.(min.(lower, bb[:] .- k[:]), BD)
        upper .= min.(max.(upper, be[:] .+ k[:]), BE)
        k .= (upper .- lower) ./ (psize .- 1)
        Mbounds .= upper .- lower
        x = zeros(n, m1 * sp)
        Xp1 = Xp[:, indexX[1:sp]]
        # println(Mbounds)

        BestX = copy(Xp1)
        maxpsize = maximum(psize)
        for i in 1:p1
          x[:, (i-1)*maxpsize+1:i*maxpsize] .= repeat(Xp1[:, i], 1, maxpsize)
        end

        Pi = zeros(Int, n, p1)
        for i = 1:p1
          Pi[:, i] .= randperm(n)
        end

        LL = zeros(Float64, n, maxpsize)
        for i in 1:n  # 遍历 n 行
          LL[i, :] .= lower[i] .+ k[i] .* (0:maxpsize-1)  # 填充每行
        end

        Index1 = 1
        BestY = zeros(Float64, n + 1, p1)
        BestY[1, :] .= yps[1:p1]
        BX = EachPar[:, s]
        MM = m1 * p1
        y = Vector{Float64}(undef, MM)

        for i = 1:n
          for j = 1:p1
            # 更新 x 矩阵的部分
            xneed = LL[Pi[i, j], 1:maxpsize-1] + (k[Pi[i, j]]*rand(1, psize[Pi[i, j]] - 1))[:]
            x[Pi[i, j], (j-1)*psize[Pi[i, j]]+2:j*psize[Pi[i, j]]] .= xneed
            x[Pi[i, j], (j-1)*psize[Pi[i, j]]+1] = x[Pi[i, j], 1] + k[Pi[i, j]] * (2 * rand() - 1)

            # 处理边界情况
            if x[Pi[i, j], (j-1)*psize[Pi[i, j]]+1] < lower[Pi[i, j]]
              x[Pi[i, j], (j-1)*psize[Pi[i, j]]+1] =
                lower[Pi[i, j]] + k[Pi[i, j]] * rand()
            elseif x[Pi[i, j], (j-1)*psize[Pi[i, j]]+1] > upper[Pi[i, j]]
              x[Pi[i, j], (j-1)*psize[Pi[i, j]]+1] =
                upper[Pi[i, j]] - k[Pi[i, j]] * rand()
            end
          end


          y, fe, feS = compute_y(y, fun, x, fe, feS, MM; goal_multiplier)
          Index1 += 1

          for j = 1:p1
            nash, index = findmin(y[(j-1)*psize[Pi[i, j]]+1:j*psize[Pi[i, j]]])
            BestY[Index1, j] = nash

            x[Pi[i, j], (j-1)*psize[Pi[i, j]]+1:j*psize[Pi[i, j]]] .=
              x[Pi[i, j], (j-1)*psize[Pi[i, j]]+index] * ones(Float64, maxpsize)

            if nash == minimum(BestY[1:Index1, j])
              BestX[:, j] .= x[:, (j-1)*psize[Pi[i, j]]+index]
            end
            if nash == minimum([minimum(BestY[1:Index1-1, :]), minimum(BestY[Index1, 1:j])])
              BX .= x[:, (j-1)*psize[Pi[i, j]]+index]
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
          println(goal_multiplier * minimum(BestValue[s, 1:p1]), "\tin")
        end

        Xp[:, 1:p1] .= BestX
        bb = minimum(Xp', dims=1)
        be = maximum(Xp', dims=1)
        # println(bb[:])
        delta_Mbounds = Mbounds * delta
        lower .= max.(lower, bb[:] .- delta_Mbounds)
        upper .= min.(upper, be[:] .+ delta_Mbounds)
        k .= (upper .- lower) ./ (psize .- 1)

        pp = m1
        x = zeros(n, m1 * p1)

        for j = 1:p1
          ra1 = 1:p1  # 创建 0 到 p1-1 的数组
          ra1 = collect(ra1) # 将 ra1 转换为可变的数组类型 (Vector)
          shuffle!(ra1)  # 打乱 ra1 数组
          ra1 = Int64.(ra1)  # 将 ra1 转换为 Int64 类型

          ra = Int64.([mod(j, p1 + 1), mod(j + 1, p1 + 1)])  # 计算 ra 数组
          if ra[2] == 0
            ra .= Int64.([j, 1])
          end

          xx = min.(Xp[:, j] .- lower, upper .- Xp[:, j]) ./ 4
          xxx = randn(n, pp - 9) .* xx

          # 更新 x 数组的某些列
          x[:, (j-1)*pp+1:j*pp-9] .= repeat(Xp[:, j], 1, pp - 9) .+ xxx

          x[:, j*pp-8] .= Xp[:, ra1[3]] .- Xp[:, ra1[1]] .+ Xp[:, ra1[2]]
          x[:, j*pp-7] .= (2 * Xp[:, ra1[1]] .- Xp[:, ra1[3]] .- Xp[:, ra1[2]]) / 2

          x[:, j*pp-6] .= Xb .- (Xp[:, ra[2]] .+ Xp[:, ra[1]]) / 2
          x[:, j*pp-5] .= Xp[:, ra[2]] .- Xb .+ Xp[:, ra[1]]
          x[:, j*pp-4] .= x[:, j*pp-5] .- (Xp[:, ra[2]] .- 2 * Xb .+ Xp[:, ra[1]]) / 2
          x[:, j*pp-3] .= Xb .+ (Xp[:, ra[2]] .- 2 * Xb .+ Xp[:, ra[1]]) / 4

          x[:, j*pp-2] .= (Xp[:, j] .+ Xb) / 2
          x[:, j*pp-1] .= 2 * Xb .- Xp[:, j]
          x[:, j*pp-0] .= 2 * Xp[:, j] .- Xb
        end

        # 生成随机数 N
        N = (BE .- BD) .* rand(n, m1 * p1) .+ BD
        x[x.<BD] .= N[x.<BD]
        x[x.>BE] .= N[x.>BE]

        # 计算 y
        MM = m1 * p1
        y = Vector{Float64}(undef, MM)
        y, fe, feS = compute_y(y, fun, x, fe, feS, MM; goal_multiplier)

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

        upper[heihei1] .= min.(nx1[heihei1] .+ k[heihei1], BE[heihei1])
        lower[heihei2] .= max.(nx2[heihei2] .- k[heihei2], BD[heihei2])

        n_reps2 = m1 * p1
        populate_best_value_fe!(BestValueFE, BestValue, s, n_reps2, Vectorization)
        populate_each_par_fe!(EachParFE, EachPar, s, n_reps2, Vectorization)
      end
    end
  end

  return OptimOutput(BestValueFE, EachParFE, BY, EachPar, fe, s; verbose)
end

end
