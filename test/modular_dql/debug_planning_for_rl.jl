using Serialization
using Statistics
using MyModule
using MyModule.Flux
using MyModule.Mill
using MyModule: all_expand, exp_size, Node, NodeID, intern!, expr, DeduplicatingNode, AbstractModel
using TimerOutputs
using JLD2
const TO = TimerOutput()
reset_timer!(TO)
include("rl_pipeline.jl")
include("sampler.jl")
include("learner.jl")

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
model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
);

function get_data()
    train_data_path = "./data/neural_rewrter/train.json"
    train_data = load_data(train_data_path)[1:1_000]
    train_data = filter(x->!occursin("select", x[1]), train_data)
    train_data = preprosses_data_to_expressions(train_data)
    sorted_data = sort(train_data, by=x->MyModule.exp_size(x))
    data = sorted_data
    return Vector{Expr}(data)
end
function not_boosted(d, sampler, traj, mbatch=128)
    if sampler.batch < 1
    	returns = []
    	for i in 1:mbatch:length(traj.next_states)
            full_range = i + mbatch > length(traj.next_states) ? length(traj.next_states) : i + mbatch - 1
            tmp = traj.next_states[i: full_range]
            input_values = [expr(MyModule.nc, i) for i in tmp]
            input_values = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_values, sym_enc))
            tmp = traj.rewards[i: full_range]
                target = get_target(tmp, sampler)
                new_smallest_node_size = minimum(exp_size.(traj.next_states))
            res = (;ds=input_values,rew=target,goal_size=new_smallest_node_size, initial_expr=d, depth=traj.pointer)
            push!(returns, res)
        end
	    return returns
    else	
        input_values = get_input_values(traj)
        target = get_target(traj.rewards, sampler)
        new_smallest_node_size = minimum(exp_size.(traj.next_states))
        return (;ds=input_values,rew=target,goal_size=new_smallest_node_size, initial_expr=d, depth=traj.pointer)
    end
end
function compute_targets_from_existing_tree(soltree::Dict, target_model, sampler::TreeSampler)
    root_ind = findfirst(x->x.depth == 0, collect(values(soltree)))
    root = collect(values(soltree))[root_ind]
    smallest_node = MyModule.extract_smallest_node(soltree)
    
    mcache = Dict()
    @timeit TO "compute targets" target_from_cache_value_network_fixed!(mcache, root, soltree, target_model, gamma=sampler.gamma)
    if sampler.batch < 1
        return get_trajectory_from_mcache(root.ex, mcache, sampler)
    else
        @timeit TO "get trajectory back batched" return get_trajectory_from_mcache_planning(soltree, root, smallest_node, mcache, sampler)
    end
end
function compute_targets_from_existing_tree(soltree::Dict, sampler::TreeSampler)
    @timeit TO "get the root" begin 
        root_ind = findfirst(x->x.depth == 0, collect(values(soltree)))
        root = collect(values(soltree))[root_ind]
    end
    @timeit TO "extact smallest node" smallest_node = MyModule.extract_smallest_node(soltree)
    
    mcache = Dict()
    @timeit TO "compute targets" target_from_cache!(mcache, root, soltree)
    if sampler.batch < 1
        @timeit TO "get trajectory back full" return get_trajectory_from_mcache(root.ex, mcache, sampler)
    else
        @timeit TO "get trajectory back batched" return get_trajectory_from_mcache_planning(soltree, root, smallest_node, mcache, sampler)
    end
end


data = get_data()
sampler = TreeSampler(max_steps=1000, max_depth=100, epsilon=1.0, eps_decay=0.80, is_directed=false, n_best=-1, batch=128, gamma=0.99)
MyModule.reset_all_function_caches()
samples = map(enumerate(data)) do (ind, d)
    @show ind
    # empty!(MyModule.nc)
    # empty!(MyModule.memoize_cache(exp_size))
    @timeit TO "reading time" soltree = deserialize("data/planning_extracted/" * "solution_tree_from_planning_for_exp_$(ind).bin")
    # JLD2.@save "data/planning_extracted/" * "solution_tree_from_planning_for_exp_$(ind).jld2" soltree
    @timeit TO "redoing the nodes" begin
        for (k,v) in soltree
            soltree[k] = Node(intern!(v.ex), v.rule_index, v.children, v.parent, v.depth, k)
        end
    end
    # traj = compute_targets_from_existing_tree(soltree, sampler)
    traj = compute_targets_from_existing_tree(soltree, model, sampler)
    # @show reading_time, traj_time, MyModule.cache_status()
    not_boosted(d, sampler, traj)
    GC.gc()
end

@show TO