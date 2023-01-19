module RegressionTools

using GLM
using DataFrames
using CSV
using Statistics
using StatsBase
using LinearAlgebra
using RCall
using Clustering
using Combinatorics
using UnPack


export make_raw_design_matrix,
       process_design_matrix,
       make_design_matrix,
       merge_columns,
       find_colinearities,
       lasso,
       elastic_net

include("design_matrix.jl")
include("data_checking.jl")
include("penalized_regression.jl")
include("post_regression.jl")

end # module RegressionTools
