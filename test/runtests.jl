using Test
using StrategicRandomSearch

include("test-01.jl")

@testset "SRS" begin
  include("funcs.jl")

  n = 100
  lower = -5.0 * ones(n)
  upper = 10.0 * ones(n)

  @test_nowarn result = SRS(Zakharov, lower, upper,
    p=3, sp=3, delta=0.01, deps=12, 
    maxn=1000, Vectorization=true, DispProcess=false, verbose=false)
  
  @time result = SRS(Zakharov, lower, upper,
    p=3, sp=3, delta=0.01, deps=12, 
    maxn=1000, Vectorization=false, DispProcess=false, verbose=false)
end
