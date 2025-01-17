module MyModule

using Flux
using JSON
using Metatheory
# using .Metatheory.EGraphs: SaturationReport, eqsat_search!, eqsat_apply!, extract!
using Statistics
using Mill
using CSV
using DataStructures
using DataFrames
using Base.Threads
using LRUCache
using SimpleChains
using SparseArrays
using LinearAlgebra
using ChainRulesCore
using Metatheory.TermInterface

# include("deduplication/lazyvcatmatrix.jl")
include("deduplication/dedu_matrix.jl")
include("deduplication/deduplication.jl")

# using Metatheory: @theory, @rule
# using InternedExpr
include("onlynode.jl")
include("onlynodes_rules.jl")
include("scoping.jl")
# include("scoping2.jl")
include("my_theory.jl")
export theory
include("small_data_loader.jl")
export load_data, preprosses_data_to_expressions
# Initialize model and example tree
all_symbols = [:+, :-, :/, :*, :<=, :>=, :min, :max, :<, :>, :select, :&&, :||, :(==), :!, :rem, :%, :(!=)]
symbols_to_index = Dict(i=>ind for (ind, i) in enumerate(all_symbols))

number_of_variables = 13
var_one_enc = Flux.onehotbatch(collect(1:number_of_variables), 1:number_of_variables)
variable_names = Dict(Symbol("v$(i-1)")=>j for (i,j) in enumerate(1:number_of_variables))

var_symbols = [Symbol("v$(i-1)") for (i,j) in enumerate(1:number_of_variables)]
new_all_symbols = vcat(all_symbols, var_symbols, :Number)
sym_enc = Dict(i=>ind for (ind,i) in enumerate(new_all_symbols))

mutable struct SearchTreePipelineConfig{TD, H, BTF}
    training_data::TD
    heuristic::H
    build_tree_function::BTF
    max_steps::Int
    max_depth::Int
    theory::Vector
    inference_cache::LRU
    size_cache::LRU
    matching_expr_cache::LRU
    hash_expr_cache::LRU
    heuristic_cache::LRU
    exp_args_cache::LRU
    alpha::Float32
end


function SearchTreePipelineConfig(training_data, heuristic, build_tree_function::Function, max_steps=1000, max_depth=50, theory=theory, alpha=0.5f0)
    matching_expr_cache = LRU(maxsize=100_000)
    inference_cache = LRU(maxsize=1_000_000)
    size_cache = LRU(maxsize=100_000)
    hash_expr_cache = LRU(maxsize=100_000)
    exp_args_cache = LRU(maxsize=100_000)
    return SearchTreePipelineConfig(
        training_data, heuristic, build_tree_function, max_steps, max_depth, theory, 
        inference_cache, size_cache, matching_expr_cache, hash_expr_cache, inference_cache, exp_args_cache, alpha
    )
end

include("tree_simplifier.jl")
include("interned_pipeline.jl")

include("tree_embedding.jl")
include("new_ex2mill.jl")
include("policy_tree_simplifier.jl")
# include("generate_random_expressions.jl")
export ExprModelSimpleChains
export ex2mill, heuristic_loss, ExprModel, all_symbols, symbols_to_index, TrainingSample, train_heuristic!, build_tree, execute, policy_loss_func, test_policy, tree_sample_to_policy_sample, PolicyTrainingSample, variable_names, new_all_symbols, sym_enc, cache_hits, cache_misses 
include("tests.jl")
include("utils.jl")
export test_training_samples, test_heuristic

end
