using MyModule
using POMDPs
using POMDPModelTools
using Random
using MCTS
using MCTS.POMDPTools

using MyModule.Mill
using MyModule.Flux
using MyModule: intern!, NodeID, exp_size, ExprModel


input_dim = 512
hidden_dim = 256
max_steps = 50
hidden_size = 64

function ffnn(idim, hidden_size, layers)
    layers == 1 && return Dense(idim, hidden_size, Flux.gelu)
    layers == 2 && return Flux.Chain(Dense(idim, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu))
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

value_model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Flux.Chain(Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, 1)),
    );


policy_model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Flux.Chain(Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, length(theory)), softmax)
    );


function training_data(n=typemax(Int))
    train_data_path = "../../data/neural_rewrter/test.json"
    train_data = load_data(train_data_path)[1:1000]
    train_data = filter(x->!occursin("select", x[1]), train_data)
    train_data = preprosses_data_to_expressions(train_data)
    train_data = sort(train_data, by=x->MyModule.exp_size(x))
    last(train_data, min(length(train_data), n))
end

trn_data = training_data(1000)

ex = first(trn_data)
ex = intern!(ex)


struct ExprEnv <: POMDPs.MDP{NodeID, NodeID}
	s₀::NodeID
end



function POMDPs.gen(m::ExprEnv, s::NodeID, a::NodeID, rng::AbstractRNG)
 	(;sp = a, r = reward(m, s, a))
end

function POMDPs.actions(m::ExprEnv, s::NodeID)
	all_actions = first(MyModule.all_expand(s, theory))
    tmp = filter(x->x!=s, all_actions)
    length(tmp) == 0 && return NodeID[s]
    return tmp
end


# function MCTS.POMDPTools.action_info(p::AbstractMCTSPlanner, s)
# 	@show typeof(s)
#     tree = MCTS.plan!(p, s)
#     best = best_sanode_Q(get_state_node(tree, s))
#     return action(best), (tree=tree, best_Q=q(best))
# end
function value_init_q(mdp::ExprEnv, s, a)
    only(value_model(a))
end

function rollout_estimate(mdp::ExprEnv, s, remaining_depth)
    policy_model(s)
end


function  POMDPs.reward(m::ExprEnv, s::NodeID, a::NodeID) 
    exp_size(s) - exp_size(a)
end

POMDPs.initialstate(e::ExprEnv) = Deterministic(e.s₀)
POMDPs.discount(e::ExprEnv) = 1
# POMDPs.terminated(e::ExprEnv) = 
# solver = MCTSSolver(n_iterations=20, depth=20, exploration_constant=5.0, init_Q=value_init_q)
max_expansions = 20
final_res = []
for (ind, ex) in enumerate(trn_data)
    println("$(ind): $(ex)")
    if rem(ind, 10) == 0
        empty!(MyModule.memoize_cache(MyModule.all_expand))
        empty!(MyModule.memoize_cache(exp_size))
        empty!(MyModule.nc)
    end
    
    ex = trn_data[100]
    solver = MCTSSolver(n_iterations=100,reuse_tree = true, depth=50, init_N = 0, exploration_constant=10.0)
    ex = intern!(ex)
    env = ExprEnv(ex)
    policy = solve(solver, env)
    expansion_path = NodeID[ex]
    tmp = ex
    t = @elapsed for i in 1:max_expansions
        a = action(policy, tmp)
        push!(expansion_path, a)
        tmp = a
    end
    exp_size.(expansion_path)

    @show t
    @show expansion_path[end]
    push!(final_res, exp_size(expansion_path[end]))
end

function visulization()
    using D3Trees
    ex = trn_data[100]
    solver = MCTSSolver(n_iterations=100,reuse_tree = true, depth=50, init_N = 3, exploration_constant=5.0, enable_tree_vis=true)
    ex = intern!(ex)
    env = ExprEnv(ex)
    planner = solve(solver, env)
    expansion_path = NodeID[ex]
    state = ex

    a = action(planner, state)
    a, info = action_info(planner, state)
    D3Tree(info[:tree], init_expand=2) # click on the node to expand it


    t = @elapsed for i in 1:max_expansions
        a = action(policy, tmp)
        push!(expansion_path, a)
        tmp = a
    end
    exp_size.(expansion_path)
end