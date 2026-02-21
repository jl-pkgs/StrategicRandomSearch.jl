module StrategicRandomSearch

using Random
using LinearAlgebra
using Printf
using Parameters
using Base.Threads

import NaNStatistics: nanminimum, nanmaximum, nanmax, nanmin

include("OptimOutput.jl")
include("SRS.jl")
include("tools.jl")


export SRS
export OptimOutput

end
