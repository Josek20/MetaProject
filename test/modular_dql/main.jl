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
# mutable struct DoubleHeadedModel{MB, FH, SH} <: AbstractModel
#     main_body::MB
#     first_head::FH
#     second_head::SH
# end
# (m::DoubleHeadedModel)(x) = (only(m.first_head(x)), only(m.second_head(x)))
# DoubleHeadedModel(m::ExprModel, b::Chain) = DoubleHeadedModel(m, m.heuristic, b)
# function (m::DoubleHeadedModel)(x)
#     inference_type = MyModule.get_inference_type(x)
#     ds = MyModule.general_cached_inference(x, inference_type, m.main_body)
#     return (only(m.first_head(ds)), only(m.second_head(ds)))
# end

include("learner.jl")

Base.only(t::Tuple{Float32, Float32}) = t
Base.only(t::Matrix{Float32}) = size(t)[1] > 1 ? Tuple(vec(t)) : first(t)
Base.vec(t::Tuple{Float32, Float32}) = t
Base.isless(a::Tuple{Float32, Float32}, b::Tuple{Float32, Float32}) = a[1] < b[1] || (a[1] == b[1] && a[2] < b[2])
Base.round(t::Tuple{Float32, Float32}, m::RoundingMode{:Nearest}; digits::Int64) = t
function get_data(;path="train.json")
    train_data_path = "./data/neural_rewrter/$(path)"
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
# sampler = TreeSampler(max_steps=100, max_depth=100, epsilon=0.5, eps_decay=0.80, is_directed=true, n_best=-1, batch=128, gamma=0.6)
# sampler = DGSampler(max_steps=100, max_depth=100, epsilon=0.5, eps_decay=0.80, is_directed=false, n_best=-1, batch=128, gamma=0.9)
# sampler = DAGSampler(max_steps=100, max_depth=100, epsilon=0.5, eps_decay=0.80, is_directed=true, n_best=-1, batch=128, gamma=0.6)
# sampler = LinearSampler(max_steps=1000, max_depth=100, epsilon=0.0, eps_decay=0.80, is_directed=true, n_best=-1, batch=128, gamma=1)

sampler = TreeSampler2Values(max_steps=100, max_depth=100, epsilon=1.0, eps_decay=0.80, is_directed=false, n_best=-1, batch=128)
# sampler = RLSampler1(max_steps=50, epsilon=1.0, eps_decay=0.80)
# sampler = RLSampler(max_steps=50, epsilon=1.0, eps_decay=0.80)

model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 2))
    );
target_model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 2))
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
function validate_trained_linear(pipeline, data, names)
    sampler = RLSampler(max_steps=100, epsilon=0.0, eps_decay=0.80)
    MyModule.reset_all_function_caches()
    df = map(data) do ex
        pipeline.env.s_init = intern!(ex)
        reset!(pipeline.env)
        env = pipeline.env
        soltree = Dict()
        traj = Trajectory()
        for t in 1:sampler.max_steps
            possible_actions = action_space(env)
            if rand() >= sampler.epsilon
                o = [vec(model(pa)) for pa in possible_actions]
                a = possible_actions[argmin(o)]
            else
                a = rand(possible_actions)
            end
            s = state(env)
            act!(env, a)
            ns = state(env)
            r = reward(env, a, s)
            is_done = isempty(action_space(env)) || is_terminal(env)
            push2traj!(traj, (s,a,r,ns,is_done))
            if is_done
                break
            end
        end
        smallest_node_id = argmin(MyModule.exp_size.(traj.next_states))
        (; s₀ = MyModule.exp_size(pipeline.env.s_init), sₙ = MyModule.exp_size(traj.next_states[smallest_node_id]), se = traj.next_states[smallest_node_id], pr = [])
    end |> DataFrame
    CSV.write("stats/linear_results_of_$(names)_hidden$(hidden_size).csv", df)
    # return df
end
function validate_train_tree(pipeline, data, names; path="stats/")
    MyModule.reset_all_function_caches()
    df = map(enumerate(data)) do (ind,ex)
        if mod(ind, 100) == 0
            empty!(MyModule.nc)
            MyModule.reset_all_function_caches()
        end
        pipeline.env.s_init = intern!(ex)
        reset!(pipeline.env)
        sampler = pipeline.sampler
        env = pipeline.env
        model = pipeline.model
        # Base.Filesystem.touch("$(dir_name)/$(ex)")
        
        t = @elapsed soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(state(env), model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=0.0)
        println("ind=$(ind), ex=$(ex), time=$(t)")
        (; s₀ = MyModule.exp_size(root.ex), sₙ = MyModule.exp_size(smallest_node.ex), se = expr(MyModule.nc,smallest_node.ex), pr = [])
    end |> DataFrame
    CSV.write(path * "results_of_$(names)_hidden$(hidden_size).csv", df)
end
function get_convergence_stats(data, pipeline, names; path="stats/")
    convergence_stats = map(data) do d
        pipeline.env.s_init = intern!(d)
        reset!(pipeline.env)
        sampler = pipeline.sampler
        env = pipeline.env
        model = pipeline.model
        soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(state(env), model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=0.0, gamma=sampler.gamma)
        all_inner_nodes = filter(i->length(i.children) != 0, collect(values(soltree)))
        # check if converged
        stds_values = map(all_inner_nodes) do i
            all_children_nodes = map(x->soltree[x].ex, i.children)
            all_values = [exp_size(root.ex) - exp_size(i.ex) + sampler.gamma * only(model(j)) for j in all_children_nodes]
            return length(all_values) > 1 ? std(all_values) : 0
        end
        # (;ex=env.s_init, number_of_converged_parents=converged,total_nodes=length(all_inner_nodes))
        # (;ex=env.s_init, mean_std=mean(stds_values),total_nodes=length(all_inner_nodes))
        filtered_stds = filter(!=(0),stds_values)
        (;ex=env.s_init, mean_std=mean(filtered_stds),filtered_total_nodes=length(filtered_stds), total_nodes=length(all_inner_nodes))
    end |> DataFrame
    CSV.write(path * "convergence_stats_$(names)_hidden$(hidden_size).csv", convergence_stats)
    return convergence_stats
end

# model_path = "models/dqn_first_Tree_gamma1/"
# for i in 1:2
#     if i == 1
#         wich_one = "trained_"
#         data = get_data(;path="train.json")
#     else
#         wich_one = "trained_test_"
#         data = get_data(;path="test.json")
#     end
#     for ep in 1:20
#         model = deserialize(model_path * "trained_parallel_DQN_second_DG_not_boosted_ep$(ep)_batch128_gamma1_for_graph_stats.bin")
#         pipeline.model = model
#         validate_train_tree(pipeline, data, wich_one * "DQN_second_DG_not_boosted_ep$(ep)_batch128_gamma1", path="stats/dqn_first_Tree_gamma1/")
#     end
# end

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
tmp1, training_stats = train!(pipeline, data[900:910], episodes=1)
# plot_res = map(1:19) do ep
#     pipeline.model = deserialize("models/trained_DQN_linear_ep$(ep)_for_graph_stats.bin")
#     res = map(data) do d
#         pipeline.env.s_init = intern!(d)
#         traj = sample_trajectory(pipeline.sampler, pipeline.env, pipeline.model)
#         exp_size(env.s_init) - minimum(exp_size.(traj.next_states))
#     end
#     mean(res)
# end
# 291
# validate_train_tree(pipeline, data, "trained_test_DQN_first_DG_not_boosted_ep$(18)_batch128_gamma1", path="stats/dqn_first_4th_20ep/")
#=
MyModule.reset_all_function_caches()
empty!(MyModule.nc)
reset_timer!(MyModule.TO)
@show MyModule.cache_status()
test1 = map(enumerate(data)) do (ind, d)
    pipeline.env.s_init = intern!(d)
    reset!(pipeline.env)
    # traj = sample_trajectory(pipeline.sampler, pipeline.env, pipeline.model)
    search_time = @elapsed soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(state(pipeline.env), pipeline.model; max_expansions=pipeline.sampler.max_steps, max_depth=pipeline.sampler.max_depth, epsilon=pipeline.sampler.epsilon)
    # search_time = @elapsed soltree, smallest_node, root = MyModule.initialize_tree_search(MyModule.intern!(d), model, max_expansions=sampler.max_steps, max_depth=sampler.max_depth)
    open("stats/search_tree_stats.txt", "a") do io
        write(io, "time=$(search_time);number of nodes=$(length(soltree)).\n")
    end
    if mod(ind, 100) == 0
        MyModule.reset_all_function_caches()
        empty!(MyModule.nc)
    end
end
@show MyModule.cache_status()
@show TO
@show MyModule.TO
=#
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
