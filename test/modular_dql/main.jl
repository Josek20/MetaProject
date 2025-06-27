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
include("rl_pipeline.jl")
include("my_env.jl")
include("sampler.jl")
include("learner.jl")

function get_data()
    train_data_path = "./data/neural_rewrter/train.json"
    train_data = load_data(train_data_path)[1:1_000]
    train_data = filter(x->!occursin("select", x[1]), train_data)
    train_data = preprosses_data_to_expressions(train_data)
    sorted_data = sort(train_data, by=x->MyModule.exp_size(x))
    data = sorted_data
    return data
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

sampler = TreeSampler(max_steps=50, max_depth=100, epsilon=1.0, eps_decay=0.50, is_directed=true, n_best=100, batch=64)
# sampler = RLSampler(max_steps=50, epsilon=1.0, eps_decay=0.80)
model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
    );
learner = DummyLerner(Flux.mse, model, max_iter=10)
env = MyTreeEnv(data[1], model)

pipeline = RLPipeline(env, model, sampler, learner)

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
train!(pipeline, episodes=10)


# Test 
# traj = sample_trajectory(pipeline.sampler, pipeline.env, pipeline.model)

# visualization(traj.soltree, pipeline.model, Dict())