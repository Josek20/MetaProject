module MyModule

using Flux
using JSON
using Metatheory
using .Metatheory.EGraphs: SaturationReport, eqsat_search!, eqsat_apply!, extract!
using GraphNeuralNetworks: GNNGraph, GCNConv
using Statistics
using Mill
using BSON
using JLD2
using CSV
# using Metatheory.Rewriters
using DataStructures
using DataFrames

include("small_data_loader.jl")
export load_data, preprosses_data_to_expressions
# include("egraph_processor.jl")
# import .get_number_of_enods
# include("graph_neural_network_module.jl")
include("tree_embedding.jl")
include("tree_simplifier.jl")
include("policy_tree_simplifier.jl")
export ex2mill, heuristic_loss, ExprModel, all_symbols, symbols_to_index, TrainingSample, train_heuristic!, build_tree, execute, policy_loss_func, test_policy, tree_sample_to_policy_sample, PolicyTrainingSample

include("tests.jl")
export test_training_samples, test_heuristic 

end
