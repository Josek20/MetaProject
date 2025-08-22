using CSV
using Graphs
using DataFrames
using Plots
using D3Trees
using Optimisers
using Base.Threads
using Random
using Statistics
using MyModule
using MyModule.Flux
using MyModule.Mill
using MyModule: all_expand, exp_size, Node, NodeID, intern!, expr, DeduplicatingNode, AbstractModel
using Serialization
using TimerOutputs
const TO = TimerOutput()
reset_timer!(TO)
Random.seed!(42)
include("rl_pipeline.jl")
include("my_env.jl")
include("sampler.jl")
mutable struct DoubleHeadedModel{MB, FH, SH} <: AbstractModel
    main_body::MB
    first_head::FH
    second_head::SH
end
# (m::DoubleHeadedModel)(x) = (only(m.first_head(x)), only(m.second_head(x)))
DoubleHeadedModel(m::ExprModel, b::Chain) = DoubleHeadedModel(m, m.heuristic, b)
function (m::DoubleHeadedModel)(x)
    inference_type = MyModule.get_inference_type(x)
    ds = MyModule.general_cached_inference(x, inference_type, m.main_body)
    return (only(m.first_head(ds)), only(m.second_head(ds)))
end

include("learner.jl")

Base.only(t::Tuple{Float32, Float32}) = t
# Base.only(t::Matrix{Float32}) = Tuple(vec(t))
Base.vec(t::Tuple{Float32, Float32}) = t
Base.isless(a::Tuple{Float32, Float32}, b::Tuple{Float32, Float32}) = a[1] < b[1] || (a[1] == b[1] && a[2] < b[2])
Base.round(t::Tuple{Float32, Float32}, m::RoundingMode{:Nearest}; digits::Int64) = t
function get_data()
    train_data_path = "../data/neural_rewrter/train.json"
    train_data = load_data(train_data_path)[1:1_000]
    train_data = filter(x->!occursin("select", x[1]), train_data)
    train_data = preprosses_data_to_expressions(train_data)
    sorted_data = sort(train_data, by=x->MyModule.exp_size(x))
    data = sorted_data
    return Vector{Expr}(data)
end

data = get_data()

hidden_size=64
input_size = 64

function ffnn(idim, hidden_size, layers)
    layers == 1 && return Dense(idim, hidden_size, Flux.gelu)
    layers == 2 && return Chain(Dense(idim, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu))
end

head_model = ProductModel(
    (;head = ffnn(length(new_all_symbols), hidden_size, 1),
      args = ffnn(hidden_size, hidden_size, 1),  
        ),
    ffnn(2*hidden_size, hidden_size, 1)
    )

args_model = ProductModel(
    (;args = ffnn(hidden_size, hidden_size, 1),  
      position = Dense(2,hidden_size),  
        ),
    ffnn(2*hidden_size, hidden_size, 1)
    )

# sampler = PlanningTreeSampler(max_steps=10, max_depth=100, epsilon=1.0, eps_decay=0.75, is_directed=false, n_best=10, batch=64)
sampler = TreeSampler(max_steps=1000, max_depth=100, epsilon=0.5, eps_decay=0.80, is_directed=false, n_best=-1, batch=128)
# sampler = TreeSampler2Values(max_steps=100, max_depth=100, epsilon=1.0, eps_decay=0.80, is_directed=false, n_best=-1, batch=128)

# sampler = RLSampler1(max_steps=50, epsilon=1.0, eps_decay=0.80)
# sampler = RLSampler(max_steps=50, epsilon=1.0, eps_decay=0.80)
# sampler = DQNSampler(max_steps=100, epsilon=1.0, eps_decay=0.80)

model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
    );
target_model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
    );
target_model = deepcopy(model)
# model = DoubleHeadedModel(model, Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1)))
learner = DummyLerner(Flux.mse, model, max_iter=10)
# learner = DummyLerner(Flux.crossentropy, model, max_iter=10)
# learner = DummyLerner(my_reinforce_loss, model, max_iter=1)
# learner = HeadLerner(Flux.mse, model, max_iter=50)
env = MyTreeEnv(data[300], model)

pipeline = SimpleRLPipeline(env, model, sampler, learner, target_model)
# pipeline = RLPipeline(env, model, sampler, learner)

function full_validation(data, pipeline)
    res = 0
    validation_sampler = RLSampler(max_steps=50, epsilon=0.0, eps_decay=0.80)
    for i in data
        env = MyTreeEnv(i, model)
        pipeline.env = env
        traj = sample_trajectory(validation_sampler, pipeline.env, pipeline.model)
        res += validation(traj)
    end
    return res / length(data)
end

function plot_stats(stats::NamedTuple; exp_name="")
    if hasproperty(stats, :loss_stats)
        aggregated_loss = hcat(map(x->x.loss_over_time, training_stats.loss_stats)...)
        # plot(aggregated_loss)
        mean_loss = mean(aggregated_loss, dims=2)
        std_loss = std(aggregated_loss, dims=2)
        plot()

        plot!(mean_loss[:], label="Mean Loss", lw=2, color=:red)
        plot!(mean_loss[:] .+ std_loss[:], ribbon=(std_loss[:]), fillalpha=0.2, linealpha=0, color=:red)
        plot!(mean_loss[:] .- std_loss[:], ribbon=(std_loss[:]), fillalpha=0.2, linealpha=0, color=:red)
        savefig("stats/loss_stats_$(exp_name).png")
    end
    if hasproperty(stats, :val_stats)
        aggregated_linear = hcat(map(x->map(y->first(y), x.val), training_stats.val_stats)...)
        aggregated_tree = hcat(map(x->map(y->last(y), x.val), training_stats.val_stats)...)
        # plot(aggregated_loss)
        mean_val = mean(aggregated_tree, dims=2)
        std_val = std(aggregated_tree, dims=2)
        plot()
        plot!(mean_val[:], label="Mean Validation Reduction", lw=2, color=:red)
        plot!(mean_val[:] .+ std_val[:], ribbon=(std_val[:]), fillalpha=0.2, linealpha=0,color=:red)
        plot!(mean_val[:] .- std_val[:], ribbon=(std_val[:]), fillalpha=0.2, linealpha=0,color=:red)
        savefig("stats/validation_stats_$(exp_name).png")
    end
end
function validate_train(pipeline, data, names)
    empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
    empty!(MyModule.memoize_cache(MyModule.exp_size))
    df = map(data) do ex
        pipeline.env.s_init = intern!(ex)
        reset!(pipeline.env)
        sampler = pipeline.sampler
        env = pipeline.env
        model = pipeline.model
        # Base.Filesystem.touch("$(dir_name)/$(ex)")
        soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(state(env), model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=0.0)
        (; s₀ = MyModule.exp_size(root.ex), sₙ = MyModule.exp_size(smallest_node.ex), se = smallest_node.ex, pr = [])
    end |> DataFrame
    CSV.write("stats/results_of_$(experiment_name)_$(names)_hidden$(hidden_size).csv", df)
end
function get_convergence_stats(data, pipeline; epsilon=0.1, exp_name="")
    empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
    empty!(MyModule.memoize_cache(MyModule.exp_size))
    convergence_stats = map(data) do d
        pipeline.env.s_init = intern!(d)
        reset!(pipeline.env)
        sampler = pipeline.sampler
        env = pipeline.env
        model = pipeline.model
        soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(state(env), model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=0.0)
        all_inner_nodes = filter(i->length(i.children) != 0, collect(values(soltree)))
        # check if converged
        converged = 0
        for i in all_inner_nodes
            all_children_nodes = map(x->soltree[x].ex, i.children)
            all_values = only.(model.(all_children_nodes))
            min_val = minimum(all_values)
            max_val = maximum(all_values)
            converged += max_val - min_val <= epsilon
        end
        (;ex=env.s_init, number_of_converged_parents=converged,total_nodes=length(all_inner_nodes))
    end |> DataFrame
    CSV.write("stats/convergence_stats_$(exp_name).csv", convergence_stats)
    return convergence_stats
end

# for ep in 1:2
#     for i in data[1:1]
#         @show i
#         env = MyTreeEnv(i, model)
#         pipeline.env = env
#         pipeline.sampler.epsilon = 1.0
#         train!(pipeline, episodes=10)
#     end
#     @show full_validation(data, pipeline)
# end
# tmp1, training_stats = train!(pipeline, data, episodes=1)
MyModule.reset_all_function_caches()
reset_timer!(MyModule.TO)
@show MyModule.cache_status()
@show @elapsed map(enumerate(data[1:10])) do (ind, d)
    pipeline.env.s_init = intern!(d)
    reset!(pipeline.env)
    # traj = sample_trajectory(pipeline.sampler, pipeline.env, pipeline.model)
    soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(state(pipeline.env), pipeline.model; max_expansions=pipeline.sampler.max_steps, max_depth=pipeline.sampler.max_depth, epsilon=pipeline.sampler.epsilon)
    # search_time = @elapsed soltree, smallest_node, root = MyModule.initialize_tree_search(MyModule.intern!(d), model, max_expansions=sampler.max_steps, max_depth=sampler.max_depth)
    open("stats/search_tree_stats.txt", "a") do io
        write(io, "time=$(search_time);number of nodes=$(length(soltree)).\n")
    end
    if mod(ind, 100) == 0
        empty!(MyModule.nc)
    end
end
@show MyModule.cache_status()
@show TO
@show MyModule.TO
# tmp1, training_stats = train1!(pipeline, data[1:200], episodes=1)
# @timeit TO "whole_time" tmp1, training_stats = train1!(pipeline, data, episodes=1)
# exp_name = "second_DG2"
# exp_name = "first_DAG1"
# serialize("models/trained_DQN_$(exp_name)_ep$(100)_hidden$(hidden_size).bin", model)
# "models/trained_DQN_first_DAG1_ep$(episode)_for_graph_stats.bin"
# plot_stats(training_stats, exp_name=exp_name)
# c = get_convergence_stats(data[300:302], pipeline, epsilon=0.01, exp_name=exp_name)

# tmp1 = train1!(pipeline, data[300:300]; episodes=100)
# train!(pipeline; episodes=1000)
# soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(state(env), pipeline.model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon)
# soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(state(env), pipeline.model; max_expansions=3, max_depth=100, epsilon=0.0)
# mcache = Dict()
# root_id = findfirst(x->x.depth == 0, soltree)
# target_from_cache2!(mcache, soltree[root_id], soltree, soltree[root_id])
# target_from_cache!(mcache, soltree[root_id], soltree)
# # @show length(mcache), length(soltree)
# @assert length(mcache) == length(soltree)
# # Test 
# traj = sample_trajectory(pipeline.sampler, pipeline.env, pipeline.model)
# target_rew = Dict(k=>i for (k,i) in zip(traj.next_states, traj.rewards))
# visualization(soltree1, pipeline.model, tmp1)