# Zakharov: 最优值为1
function Zakharov(x::AbstractMatrix{Float64})
    n_dim, n_samples = size(x)
    y = zeros(Float64, n_samples)

    @inbounds for j in 1:n_samples
        y1 = 0.0
        y2 = 0.0
        for i in 1:n_dim
            xi = x[i, j] - 0.5
            y1 += xi * xi
            y2 += (i + 1) * xi
        end
        y2_half = 0.5 * y2
        y[j] = y1 + y2_half^2 + y2_half^4 + 1
    end
    return y
end

function Zakharov(x::AbstractVector{Float64})
    y1 = 0.0
    y2 = 0.0
    @inbounds for i in eachindex(x)
        xi = x[i] - 0.5
        y1 += xi * xi
        y2 += (i + 1) * xi
    end
    y2_half = 0.5 * y2
    return y1 + y2_half^2 + y2_half^4 + 1
end


@testset "Zakharov" begin
    n = 20
    lower = -5.0 * ones(n)
    upper = 10.0 * ones(n)
    @time result = SRS(Zakharov, lower, upper,
        p=10, delta=0.01, deps=12,
        maxn=Int(1e5), verbose=true)
end
