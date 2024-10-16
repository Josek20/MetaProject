module MyModule

using Flux
using JSON
using Metatheory
# using .Metatheory.EGraphs: SaturationReport, eqsat_search!, eqsat_apply!, extract!
using Statistics
using Mill
using CSV
# using Metatheory.Rewriters
using DataStructures
using DataFrames
using Base.Threads
using LRUCache
using SimpleChains

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
# new_all_symbols = append(all_symbols, keys(variable_names))
# include("egraph_processor.jl")
# import .get_number_of_enods
# include("graph_neural_network_module.jl")
include("tree_simplifier.jl")
include("tree_embedding.jl")
include("new_ex2mill.jl")
include("policy_tree_simplifier.jl")
# include("generate_random_expressions.jl")
export ex2mill, heuristic_loss, ExprModel, all_symbols, symbols_to_index, TrainingSample, train_heuristic!, build_tree, execute, policy_loss_func, test_policy, tree_sample_to_policy_sample, PolicyTrainingSample, variable_names, new_all_symbols, sym_enc, cache_hits, cache_misses 
include("tests.jl")
export test_training_samples, test_heuristic 

end
