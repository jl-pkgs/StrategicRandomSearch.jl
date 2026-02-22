using StrategicRandomSearch, Test

# Zakharov: 最优值为1
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
    n = 10
    lower = -1.0 * ones(n)
    upper = 1.0 * ones(n)
    @time result = SRS(Zakharov, lower, upper,
        p=5, delta=0.3,
        maxn=Int(5e4), verbose=true)
end
