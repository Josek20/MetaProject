using CSV
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
using MyModule: all_expand, exp_size, Node, NodeID, intern!, expr, DeduplicatingNode
using Serialization
include("my_test_env.jl")
include("tree_env_setup.jl")
include("clean_dql.jl")

hidden_size=64
# epsilone_decay = 0.9 / 2.5e2
epsilon = eps_start = 1.0
eps_end = 0.01
eps_decay = 0.95

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
online_q = Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
target_q = Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
target_q = deepcopy(online_q)
online_policy = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    online_q
    );

target_policy = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    target_q
    );


train(data[1:1], online_policy, target_policy; lr=0.0003, batch_size=30, max_steps=20, max_depth=100, epochs=100, update_iter=10, epsilon=0.0)