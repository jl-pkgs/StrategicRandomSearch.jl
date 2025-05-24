using Test
using StrategicRandomSearch

include("test-01.jl")

@testset "SRS" begin
  include("funcs.jl")

  n = 100
  lower = -5.0 * ones(n)
  upper = 10.0 * ones(n)

  @test_nowarn result = SRS(Zakharov, lower, upper,
    n_candidate=3, sp=3, delta=0.01, deps=12,
    maxn=1000, verbose=true)
  
  @time result = SRS(Zakharov, lower, upper,
    n_candidate=3, sp=3, delta=0.01, deps=12,
    maxn=1000, verbose=true)
end
