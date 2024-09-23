module MyModule

using Flux
using JSON
using Metatheory
using .Metatheory.EGraphs: SaturationReport, eqsat_search!, eqsat_apply!, extract!
using Statistics
using Mill
using CSV
# using Metatheory.Rewriters
using DataStructures
using DataFrames
using Base.Threads

include("small_data_loader.jl")
export load_data, preprosses_data_to_expressions
# include("egraph_processor.jl")
# import .get_number_of_enods
# include("graph_neural_network_module.jl")
include("tree_embedding.jl")
include("new_ex2mill.jl")
sym_enc = create_all_symbols_encoding(new_all_symbols)
include("tree_simplifier.jl")
include("policy_tree_simplifier.jl")
# include("generate_random_expressions.jl")
export ex2mill, heuristic_loss, ExprModel, all_symbols, symbols_to_index, TrainingSample, train_heuristic!, build_tree, execute, policy_loss_func, test_policy, tree_sample_to_policy_sample, PolicyTrainingSample, variable_names, fast_ex2mill
include("tests.jl")
export test_training_samples, test_heuristic 

end
