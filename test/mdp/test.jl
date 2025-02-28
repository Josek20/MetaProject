using MyModule
using POMDPs
using POMDPModelTools
using Random
using MCTS
using MCTS.POMDPTools

using MyModule: intern!, NodeID, exp_size

function training_data(n=typemax(Int))
    train_data_path = "../../data/neural_rewrter/test.json"
    train_data = load_data(train_data_path)[1:100]
    train_data = filter(x->!occursin("select", x[1]), train_data)
    train_data = preprosses_data_to_expressions(train_data)
    train_data = sort(train_data, by=x->MyModule.exp_size(x))
    last(train_data, min(length(train_data), n))
end

trn_data = training_data(10)

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
    filter(x->x!=s, all_actions)
end


# function MCTS.POMDPTools.action_info(p::AbstractMCTSPlanner, s)
# 	@show typeof(s)
#     tree = MCTS.plan!(p, s)
#     best = best_sanode_Q(get_state_node(tree, s))
#     return action(best), (tree=tree, best_Q=q(best))
# end

POMDPs.reward(m::ExprEnv, s::NodeID, a::NodeID) = exp_size(s) - exp_size(a)

POMDPs.initialstate(e::ExprEnv) = Deterministic(e.s₀)
POMDPs.discount(e::ExprEnv) = 0.01
env = ExprEnv(ex)
solver = MCTSSolver(n_iterations=50, depth=20, exploration_constant=5.0)
planner = solve(solver, env)

a = action(planner, ex)
