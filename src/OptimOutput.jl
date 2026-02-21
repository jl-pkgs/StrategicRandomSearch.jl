@with_kw mutable struct OptimOutput
    num_iter::Int = 0                   # 迭代次数
    num_call::Int = 0                   # 目标函数调用次数

    x::Vector{Float64}
    feval::Float64 = NaN

    x_iters::Matrix{Float64}     # EachPar, 图1b, 精英中的精英, 红色点中的最优一个
    feval_iters::Vector{Float64} # BY

    _x_iters::Vector
    _feval_iters::Vector{Float64}
end

function print_item(key, value)
    name = @sprintf("%-10s: ", key)
    printstyled(name, bold=true, color=:blue)
    println(value)
end


function OptimOutput(
    _feval_iters::Vector{Float64}, _x_iters::Vector,
    BY::Vector{Float64}, EachPar::Matrix{Float64},
    num_call::Int, num_iter::Int; verbose::Bool=true)

    # x_iters = collect(x_iters')
    feval = BY[num_iter]
    EachPar = EachPar[:, 1:num_iter]' |> collect # 每次只保存了最佳的
    x = EachPar[num_iter, :]

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
        feval_iters=BY, x_iters=EachPar,
        _feval_iters, _x_iters)
end
