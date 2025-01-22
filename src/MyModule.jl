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
using MacroTools: isexpr, combinedef, namify, splitarg, splitdef

include("memoization/memoize.jl")
# include("deduplication/lazyvcatmatrix.jl")
include("deduplication/dedu_matrix.jl")
include("deduplication/deduplication.jl")

# using Metatheory: @theory, @rule
# using InternedExpr
include("onlynode.jl")
include("onlynodes_rules.jl")
include("scoping.jl")
# include("scoping2.jl")
get_value(x) = x
typeof_value(x)  = typeof(x)

function Metatheory.matcher(slot::Metatheory.PatVar)
    pred = slot.predicate
    if slot.predicate isa Type
    # pred = x -> typeof(x) <: slot.predicate
        pred = x -> MyModule.typeof_value(x) <: slot.predicate
    end
    function slot_matcher(next, data, bindings)
        !Metatheory.islist(data) && return
        val = Metatheory.get(bindings, slot.idx, nothing)
        if val !== nothing
            if isequal(val, Metatheory.car(data))
                return next(bindings, 1)
            end
        else
            if pred(Metatheory.car(data))
                next(Metatheory.assoc(bindings, slot.idx, Metatheory.car(data)), 1)
            end
        end
    end
end


const EMPTY_DICT = Base.ImmutableDict{Int,Any}()
function (r::Metatheory.DynamicRule)(term)
    # n == 1 means that exactly one term of the input (term,) was matched
    success(bindings, n) =
    if n == 1
        bvals = [bindings[i] for i in 1:length(r.patvars)]
        bvals = map(MyModule.get_value, bvals)
        v = r.rhs_fun(term, nothing, bvals...)
        if isnothing(v)
            return nothing
        end 
        v = term isa MyModule.NodeID ? MyModule.intern!(v) : v
        return(v)
    end
    
    try
        return r.matcher(success, (term,), EMPTY_DICT)
    catch err
        rethrow(err)
        throw(RuleRewriteError(r, term))
    end
end


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

# Todo: Should be deleted. Have splited to different files
# include("tree_embedding.jl")
include("expr_model.jl")
include("expr_model_loss.jl")
# Todo: Fix the simple chains model with updated api
include("simple_chains_model.jl")
export ExprModelSimpleChains
include("new_ex2mill.jl")
include("policy_tree_simplifier.jl")
# include("generate_random_expressions.jl")
export ex2mill, heuristic_loss, ExprModel, all_symbols, symbols_to_index, TrainingSample, train_heuristic!, build_tree, execute, policy_loss_func, test_policy, tree_sample_to_policy_sample, PolicyTrainingSample, variable_names, new_all_symbols, sym_enc, cache_hits, cache_misses 
include("tests.jl")
include("utils.jl")
export test_training_samples, test_heuristic

end
