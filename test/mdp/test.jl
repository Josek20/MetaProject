using MyModule
using POMDPs
using POMDPModelTools
using Random
using MCTS
using MCTS.POMDPTools
using D3Trees

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
    train_data_path = "data/neural_rewrter/train.json"
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
function my_estimate_value(mdp, s, remaining_depth)
    exp_size(mdp.s₀) - exp_size(s)
end

POMDPs.reward(m::ExprEnv, s::NodeID, a::NodeID) = exp_size(s) - exp_size(a)

POMDPs.initialstate(e::ExprEnv) = Deterministic(e.s₀)
POMDPs.discount(e::ExprEnv) = 0.8

max_expansions = 50
niter = 30
exp_depth = 10
expr_const = 10.0
@assert 1 == 0 
# function run_mcts(max_expansions, niter, exp_depth, expr_const, trn_data)
experiment_name = "mdp_test_max_exp$(max_expansions)_niter$(niter)_exp_depth$(exp_depth)_expr_const$(expr_const)"
final_res = []
for (ind, ex) in enumerate(trn_data)
    println("$(ind): $(ex)")
    if rem(ind, 10) == 0
        empty!(MyModule.memoize_cache(MyModule.all_expand))
        empty!(MyModule.memoize_cache(exp_size))
        empty!(MyModule.nc)
    end
    
    solver = MCTSSolver(n_iterations=niter, depth=exp_depth, exploration_constant=expr_const, reuse_tree=true, estimate_value=my_estimate_value, enable_tree_vis=true)
    ex = intern!(ex)
    env = ExprEnv(ex)
    planner = solve(solver, env)
    a = action(planner, ex)
    # @assert 1 == 0
    expansion_path = NodeID[ex]
    # a = action(planner, ex)
    tmp = ex
    t = @elapsed for i in 1:max_expansions
        a = action(planner, tmp)
        push!(expansion_path, a)
        tmp = a
    end
    @show t
    # @show expansion_path
    # @show exp_size.(expansion_path)
    push!(final_res, min(exp_size.(expansion_path)...))
end
tmp = sum([exp_size(intern!(trn_data[ind])) - i for (ind,i) in enumerate(final_res)]) / length(trn_data)
println(experiment_name)
@show tmp


function visulization()
    ex = trn_data[100]
    solver = MCTSSolver(n_iterations=50,reuse_tree = true, depth=50, exploration_constant=5.0, enable_tree_vis=true)
    ex = intern!(ex)
    env = ExprEnv(ex)
    planner = solve(solver, env)
    expansion_path = NodeID[ex]
    state = ex

    a = action(planner, state)
    a, info = action_info(planner, state)
    d3tr = D3Tree(info[:tree], init_expand=2) # click on the node to expand it


    t = @elapsed for i in 1:max_expansions
        a = action(planner, tmp)
        push!(expansion_path, a)
        tmp = a
    end
    exp_size.(expansion_path)
end
# end

# run_mcts(max_expansions, niter, exp_depth, expr_const)
# max_expansions, niter
# function my_premutation(objects, obj_index=1, all_premutations=[])
#     if length(objects) < obj_index
#         return
#     end

#     for i in objects[obj_index]
#         my_premutation(object, obj_index + 1, all_premutations)
#     end
# end
# run_mcts(10, 10, 10, 1.0, trn_data)
# run_mcts(30, 10, 100, 1.0)
# run_mcts(50, 10, 100, 1.0)
# run_mcts(100, 10, 100, 1.0)
# run_mcts(10, 30, 100, 1.0)
# run_mcts(30, 30, 100, 1.0)
# run_mcts(50, 30, 100, 1.0)
# run_mcts(100, 30, 100, 1.0)
# run_mcts(10, 50, 100, 1.0)
# run_mcts(30, 50, 100, 1.0)
# run_mcts(50, 50, 100, 1.0)
# run_mcts(100, 50, 100, 1.0)
# run_mcts(10, 100, 100, 1.0)
# run_mcts(30, 100, 100, 1.0)
# run_mcts(50, 100, 100, 1.0)
# run_mcts(100, 100, 100, 1.0)

tmp = """
estim_value_mdp_test_max_exp10_niter10_exp_depth10_expr_const1.0
tmp = 0.842f0
estim_value_mdp_test_max_exp10_niter10_exp_depth10_expr_const5.0
tmp = 0.853f0
estim_value_mdp_test_max_exp10_niter10_exp_depth10_expr_const10.0
tmp = 0.996f0
estim_value_mdp_test_max_exp10_niter10_exp_depth50_expr_const1.0
tmp = 1.07f0
estim_value_mdp_test_max_exp10_niter10_exp_depth50_expr_const5.0
tmp = 1.078f0
estim_value_mdp_test_max_exp10_niter10_exp_depth50_expr_const10.0
tmp = 0.869f0
estim_value_mdp_test_max_exp10_niter10_exp_depth100_expr_const1.0
tmp = 1.298f0
estim_value_mdp_test_max_exp10_niter10_exp_depth100_expr_const5.0
tmp = 0.85f0
estim_value_mdp_test_max_exp10_niter10_exp_depth100_expr_const10.0
tmp = 0.994f0
estim_value_mdp_test_max_exp10_niter30_exp_depth10_expr_const1.0
tmp = 2.902f0
estim_value_mdp_test_max_exp10_niter30_exp_depth10_expr_const5.0
tmp = 3.493f0
estim_value_mdp_test_max_exp10_niter30_exp_depth10_expr_const10.0
tmp = 3.412f0
estim_value_mdp_test_max_exp10_niter30_exp_depth50_expr_const1.0
tmp = 2.931f0
estim_value_mdp_test_max_exp10_niter30_exp_depth50_expr_const5.0
tmp = 4.065f0
estim_value_mdp_test_max_exp10_niter30_exp_depth50_expr_const10.0
tmp = 2.869f0
estim_value_mdp_test_max_exp10_niter30_exp_depth100_expr_const1.0
tmp = 2.949f0
estim_value_mdp_test_max_exp10_niter30_exp_depth100_expr_const5.0
tmp = 2.482f0
estim_value_mdp_test_max_exp10_niter30_exp_depth100_expr_const10.0
tmp = 2.227f0
estim_value_mdp_test_max_exp10_niter50_exp_depth10_expr_const1.0
tmp = 2.747f0
estim_value_mdp_test_max_exp10_niter50_exp_depth10_expr_const5.0
tmp = 4.271f0
estim_value_mdp_test_max_exp10_niter50_exp_depth10_expr_const10.0
tmp = 3.691f0
estim_value_mdp_test_max_exp10_niter50_exp_depth50_expr_const1.0
tmp = 2.901f0
estim_value_mdp_test_max_exp10_niter50_exp_depth50_expr_const5.0
tmp = 4.294f0
estim_value_mdp_test_max_exp10_niter50_exp_depth50_expr_const10.0
tmp = 3.238f0
estim_value_mdp_test_max_exp10_niter50_exp_depth100_expr_const1.0
tmp = 1.286f0
estim_value_mdp_test_max_exp10_niter50_exp_depth100_expr_const5.0
tmp = 4.371f0
estim_value_mdp_test_max_exp10_niter50_exp_depth100_expr_const10.0
tmp = 3.053f0
estim_value_mdp_test_max_exp30_niter10_exp_depth10_expr_const1.0
tmp = 1.649f0
estim_value_mdp_test_max_exp30_niter10_exp_depth10_expr_const5.0
tmp = 2.404f0
estim_value_mdp_test_max_exp30_niter10_exp_depth10_expr_const10.0
tmp = 2.483f0
estim_value_mdp_test_max_exp30_niter10_exp_depth50_expr_const1.0
tmp = 1.645f0
estim_value_mdp_test_max_exp30_niter10_exp_depth50_expr_const5.0
tmp = 3.113f0
estim_value_mdp_test_max_exp30_niter10_exp_depth50_expr_const10.0
tmp = 2.85f0
estim_value_mdp_test_max_exp30_niter10_exp_depth100_expr_const1.0
tmp = 1.753f0
estim_value_mdp_test_max_exp30_niter10_exp_depth100_expr_const5.0
tmp = 3.152f0
estim_value_mdp_test_max_exp30_niter10_exp_depth100_expr_const10.0
tmp = 2.548f0
estim_value_mdp_test_max_exp30_niter30_exp_depth10_expr_const1.0
tmp = 3.0f0
estim_value_mdp_test_max_exp30_niter30_exp_depth10_expr_const5.0
tmp = 4.816f0
estim_value_mdp_test_max_exp30_niter30_exp_depth10_expr_const10.0
tmp = 5.276f0
estim_value_mdp_test_max_exp30_niter30_exp_depth50_expr_const1.0
tmp = 3.138f0
estim_value_mdp_test_max_exp30_niter30_exp_depth50_expr_const5.0
tmp = 4.643f0
estim_value_mdp_test_max_exp30_niter30_exp_depth50_expr_const10.0
tmp = 4.793f0
estim_value_mdp_test_max_exp30_niter30_exp_depth100_expr_const1.0
tmp = 3.181f0
estim_value_mdp_test_max_exp30_niter30_exp_depth100_expr_const5.0
tmp = 4.157f0
estim_value_mdp_test_max_exp30_niter30_exp_depth100_expr_const10.0
tmp = 4.397f0
estim_value_mdp_test_max_exp30_niter50_exp_depth10_expr_const1.0
tmp = 3.011f0
estim_value_mdp_test_max_exp30_niter50_exp_depth10_expr_const5.0
tmp = 4.966f0
estim_value_mdp_test_max_exp30_niter50_exp_depth10_expr_const10.0
tmp = 5.614f0
estim_value_mdp_test_max_exp30_niter50_exp_depth50_expr_const1.0
tmp = 3.163f0
estim_value_mdp_test_max_exp30_niter50_exp_depth50_expr_const5.0
tmp = 4.448f0
estim_value_mdp_test_max_exp30_niter50_exp_depth50_expr_const10.0
tmp = 4.888f0
estim_value_mdp_test_max_exp30_niter50_exp_depth100_expr_const1.0
tmp = 1.356f0
estim_value_mdp_test_max_exp30_niter50_exp_depth100_expr_const5.0
tmp = 4.696f0
estim_value_mdp_test_max_exp30_niter50_exp_depth100_expr_const10.0
tmp = 4.662f0
estim_value_mdp_test_max_exp50_niter10_exp_depth10_expr_const1.0
tmp = 1.967f0
estim_value_mdp_test_max_exp50_niter10_exp_depth10_expr_const5.0
tmp = 3.901f0
estim_value_mdp_test_max_exp50_niter10_exp_depth10_expr_const10.0
tmp = 4.195f0
estim_value_mdp_test_max_exp50_niter10_exp_depth50_expr_const1.0
tmp = 1.657f0
estim_value_mdp_test_max_exp50_niter10_exp_depth50_expr_const5.0
tmp = 3.789f0
estim_value_mdp_test_max_exp50_niter10_exp_depth50_expr_const10.0
tmp = 4.19f0
estim_value_mdp_test_max_exp50_niter10_exp_depth100_expr_const1.0
tmp = 1.769f0
estim_value_mdp_test_max_exp50_niter10_exp_depth100_expr_const5.0
tmp = 4.016f0
estim_value_mdp_test_max_exp50_niter10_exp_depth100_expr_const10.0
tmp = 3.948f0
estim_value_mdp_test_max_exp50_niter30_exp_depth10_expr_const1.0
tmp = 3.192f0
estim_value_mdp_test_max_exp50_niter30_exp_depth10_expr_const5.0
tmp = 4.916f0
estim_value_mdp_test_max_exp50_niter30_exp_depth10_expr_const10.0
tmp = 5.865f0
estim_value_mdp_test_max_exp50_niter30_exp_depth50_expr_const1.0
tmp = 3.224f0
estim_value_mdp_test_max_exp50_niter30_exp_depth50_expr_const5.0
tmp = 4.713f0
estim_value_mdp_test_max_exp50_niter30_exp_depth50_expr_const10.0
tmp = 5.451f0
estim_value_mdp_test_max_exp50_niter30_exp_depth100_expr_const1.0
tmp = 3.187f0
estim_value_mdp_test_max_exp50_niter30_exp_depth100_expr_const5.0
tmp = 4.228f0
estim_value_mdp_test_max_exp50_niter30_exp_depth100_expr_const10.0
tmp = 4.719f0
estim_value_mdp_test_max_exp50_niter50_exp_depth10_expr_const1.0
tmp = 3.113f0
estim_value_mdp_test_max_exp50_niter50_exp_depth10_expr_const5.0
tmp = 5.425f0
estim_value_mdp_test_max_exp50_niter50_exp_depth10_expr_const10.0
tmp = 5.662f0
estim_value_mdp_test_max_exp50_niter50_exp_depth50_expr_const1.0
tmp = 3.167f0
estim_value_mdp_test_max_exp50_niter50_exp_depth50_expr_const5.0
tmp = 4.478f0
estim_value_mdp_test_max_exp50_niter50_exp_depth50_expr_const10.0
tmp = 5.472f0
estim_value_mdp_test_max_exp50_niter50_exp_depth100_expr_const1.0
tmp = 1.364f0
estim_value_mdp_test_max_exp50_niter50_exp_depth100_expr_const5.0
tmp = 4.813f0
estim_value_mdp_test_max_exp50_niter50_exp_depth100_expr_const10.0
tmp = 5.158f0
"""

tmp1 = split(tmp, "\n")
max_exps = []                                               
max_iters = []                                              
max_depth = []                                              
econsts = []                                                
fresults = []                                               
for i in 1:2:length(tmp1)-1                                 
    mexp, miter, mdepth, econst, _ = [m.match for m in eachmatch(r"\d+", tmp1[i])]
    res = Meta.parse(split(tmp1[i+1], "=")[2])
    push!(max_exps, mexp)
    push!(max_iters, miter)
    push!(max_depth, mdepth)
    push!(econsts, econst)
    push!(fresults, res)
end
df = DataFrame([max_exps,max_iters,max_depth,econsts,fresults], [:expansions, :iterations, :depth, :const, :res])
