@with_kw mutable struct OptimOutput
    num_iter::Int = 0                   # 迭代次数
    num_call::Int = 0                   # 目标函数调用次数

    # 最优解和最优值
    x::Vector{Float64}
    feval::Float64 = NaN

    # 历史记录
    x_iters::Matrix{Float64}     # EachPar, 图1b, 精英中的精英, 红色点中的最优一个
    feval_iters::Vector{Float64} # BY

    x_calls::Vector
    feval_calls::Vector{Float64}
end

function print_item(key, value)
    name = @sprintf("%-10s: ", key)
    printstyled(name, bold=true, color=:blue)
    println(value)
end


function OptimOutput(
    feval_calls::Vector{Float64}, x_calls::Vector,
    feval_iters::Vector{Float64}, x_iters::Matrix{Float64},
    num_call::Int, num_iter::Int; verbose::Bool=true)

    feval = feval_iters[num_iter]
    x = x_iters[:, num_iter]

    if verbose
        printstyled("----------------------------------- \n", bold=true, color=:blue)
        print_item("feval", feval)
        print_item("x", x)
        print_item("Iterations", num_iter)
        print_item("f(x) calls", num_call)
        printstyled("----------------------------------- \n", bold=true, color=:blue)
    end

    return OptimOutput(; num_call, num_iter,
        x, feval,
        feval_iters, x_iters,
        x_calls, feval_calls)
end
