using Test
using StrategicRandomSearch


function functn1(x::Vector{Float64})
  # This is the Goldstein - Price Function
  # Bound X1 = [-2, 2], X2 = [-2, 2]
  # Global Optimum:3.0, (0.0, -1.0)
  x1 = x[1]
  x2 = x[2]
  u1 = (x1 + x2 + 1.0)^2
  u2 = 19 .- 14 .* x1 + 3 .* x1^2 - 14 .* x2 + 6 .* x1 * x2 + 3 .* x2^2
  u3 = (2 .* x1 - 3 .* x2)^2
  u4 = 18 .- 32 .* x1 + 12 .* x1^2 + 48 .* x2 - 36 .* x1 * x2 + 27 .* x2^2
  u5 = u1 * u2
  u6 = u3 * u4
  (1 .+ u5) * (30 .+ u6)
end


@testset "functn1: Goldstein - Price Function" begin
  lower = Float64.([-2, -2])
  upper = Float64.([2, 2])
  x0 = Float64.([1, 1])
  @time r = SRS(functn1, lower, upper; seed=1)
  @test isapprox(r.x, [0.0, -1.0], atol=1e-6)
  @test isapprox(r.feval, 3, atol=1e-6)
end
