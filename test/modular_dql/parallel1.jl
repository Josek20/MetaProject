using DataFrames
using CSV
using Distributed
println("Number of workers: ", nworkers())
println("Threads per worker: ", Threads.nthreads())
@everywhere begin
    using Serialization
    using Statistics
    using MyModule
    using MyModule.StatsBase
    using MyModule.Flux
    using MyModule.Mill
    using MyModule: all_expand, exp_size, Node, NodeID, intern!, expr, DeduplicatingNode, AbstractModel
    using TimerOutputs
    const TO = TimerOutput()
    reset_timer!(TO)
    include("rl_pipeline.jl")
    include("my_env.jl")
    include("sampler1.jl")
    include("learner.jl")
    Base.only(t::Tuple{Float32, Float32}) = t
    Base.only(t::Matrix{Float32}) = size(t)[1] > 1 ? Tuple(vec(t)) : first(t)
    Base.vec(t::Tuple{Float32, Float32}) = t
    Base.isless(a::Tuple{Float32, Float32}, b::Tuple{Float32, Float32}) = a[1] < b[1] || (a[1] == b[1] && a[2] < b[2])
    Base.round(t::Tuple{Float32, Float32}, m::RoundingMode{:Nearest}; digits::Int64) = t
end

function get_data()
    train_data_path = "./data/neural_rewrter/train.json"
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
@everywhere Base.only(t::Matrix{Float32}) = size(t)[1] > 1 ? Tuple(vec(t)) : first(t)
model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
    );

target = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
    );

value_model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
    );
@everywhere function not_boosted(d, sampler, traj, mbatch=128)
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
        if hasfield(typeof(traj), :next_states)
            new_smallest_node_size = minimum(exp_size.(traj.next_states))
            return (;ds=input_values,rew=target,goal_size=new_smallest_node_size, initial_expr=d, depth=traj.pointer)
        elseif hasfield(typeof(traj), :selected_ids)
            return (;ds=input_values,rew=target,goal_size=exp_size(traj.smallest_node.ex), initial_expr=d.initial_expr, depth=traj.smallest_node.depth, softmax_ids=traj.softmax_ids, selected_ids=traj.selected_ids)
        else
            new_smallest_node_size = -1
            return (;ds=input_values,rew=target,goal_size=new_smallest_node_size, initial_expr=d.initial_expr, depth=-1)
        end
    end
end
@everywhere function boosted(d, sampler, traj)
    input_values = get_input_values(traj)
    target = get_target(traj.rewards, sampler)
    if hasfield(typeof(traj), :next_states)
        new_smallest_node_size = minimum(exp_size.(traj.next_states))
    elseif hasfield(typeof(traj), :smallest_node)
        new_smallest_node_size = exp_size(traj.smallest_node.ex)
    end
    if d.goal_size > new_smallest_node_size
        input_values = get_input_values(traj)
        target = get_target(traj.rewards, sampler)
        # goal_size = exp_size(traj.states[1]) - new_smallest_node_size
        # return(;ds=input_values,rew=target,goal_size=new_smallest_node_size, initial_expr=d.initial_expr, depth=traj.pointer)
        if hasfield(typeof(traj), :selected_ids)
            return (;ds=input_values,rew=target,goal_size=new_smallest_node_size, initial_expr=d.initial_expr, depth=traj.smallest_node.depth, softmax_ids=traj.softmax_ids, selected_ids=traj.selected_ids)
        elseif hasfield(typeof(traj), :smallest_node)
            return(;ds=input_values,rew=target,goal_size=new_smallest_node_size, initial_expr=d.initial_expr, depth=traj.smallest_node.depth)
        else
            return(;ds=input_values,rew=target,goal_size=new_smallest_node_size, initial_expr=d.initial_expr, depth=traj.pointer)
        end
    elseif (d.goal_size == new_smallest_node_size && d.depth > traj.smallest_node.depth)
        target = get_target(traj.rewards, sampler)
        input_values = get_input_values(traj)
        # return(;ds=input_values,rew=target,goal_size=new_smallest_node_size, initial_expr=d.initial_expr, depth=traj.pointer)
        # return(;ds=input_values,rew=target,goal_size=traj.goal_size, initial_expr=d.initial_expr, depth=traj.pointer)
        if hasfield(typeof(traj), :selected_ids)
            return (;ds=input_values,rew=target,goal_size=new_smallest_node_size, initial_expr=d.initial_expr, depth=traj.smallest_node.depth, softmax_ids=traj.softmax_ids, selected_ids=traj.selected_ids)
        elseif hasfield(typeof(traj), :smallest_node)
            return(;ds=input_values,rew=target,goal_size=new_smallest_node_size, initial_expr=d.initial_expr, depth=traj.smallest_node.depth)
        else
            return(;ds=input_values,rew=target,goal_size=new_smallest_node_size, initial_expr=d.initial_expr, depth=traj.pointer)
        end
    else
        return(d)
    end
end

@everywhere function compute_targets_from_existing_tree(soltree::Dict, sampler::TreeSampler)
    root_ind = findfirst(x->x.depth == 0, collect(values(soltree)))
    root = collect(values(soltree))[root_ind]
    smallest_node = MyModule.extract_smallest_node(soltree)
    
    mcache = Dict()
    all_leafs = filter(i->length(i.second.children) == 0, soltree)
    sorted_leafs = sort(all_leafs, by=x->soltree[x].depth, order=Base.Order.Reverse)
    target_from_cache!(mcache, root, soltree)
    if sampler.batch < 1
        return get_trajectory_from_mcache(root.ex, mcache, sampler)
    else
        return get_trajectory_from_mcache_planning(soltree, root, smallest_node, mcache, sampler)
    end
end
@everywhere function compute_targets_from_existing_tree(soltree::Dict, target_model, sampler::TreeSampler)
    root_ind = findfirst(x->x.depth == 0, collect(values(soltree)))
    root = collect(values(soltree))[root_ind]
    smallest_node = MyModule.extract_smallest_node(soltree)
    
    mcache = Dict()
    all_leafs = filter(i->length(i.second.children) == 0, soltree)
    sorted_leafs = sort(all_leafs, by=x->soltree[x].depth, order=Base.Order.Reverse)
    target_from_cache_value_network_fixed!(mcache, root, soltree, target_model, gamma=sampler.gamma)
    if sampler.batch < 1
        return get_trajectory_from_mcache(root.ex, mcache, sampler)
    else
        return get_trajectory_from_mcache_planning(soltree, root, smallest_node, mcache, sampler)
    end
end
function get_existing_trees(;path="data/planning_extracted/", number_of_trees=1000)
    all_soltrees = map(1:number_of_trees) do i
	soltree = deserialize(path * "solution_tree_from_planning_for_exp_$(i).bin")
	return soltree
    end
    return all_soltrees
end
function train(model, data, target_model, value_model)
    @everywhere model = $model
    @everywhere target_model = $target_model
    @everywhere value_model = $value_model
    # learner = DummyLerner(Flux.mse, model, max_iter=10)
    # learner = PolicyLerner(Flux.mse, model, max_iter=10)
    learner = PPOLerner(model, value_model)
    epochs = 2
    @everywhere is_target = true
    @everywhere is_boosted = true
    @everywhere solution = true
    @everywhere gamma = 0.99
    problem_name = "ppo_Tree"
    tmp = solution ? "solution_" : "no_solution_"
    model_name = "tree_PG_" * tmp * problem_name
    model_name *= is_boosted ? "_boosted_" : "_not_boosted_"
    gamma_name = length("$gamma") > 3 ? last("$(gamma)",2) : replace("$gamma", "." => "")
    # save_model_path = "models/ppo_" * problem_name * "_gamma$(gamma_name)/"
    save_model_path = "models/"
    @show save_model_path
    # @everywhere sampler = PolicyLinearSampler(max_steps=50, max_depth=100, n_best=-1, batch=256, gamma=gamma)
    # @everywhere sampler = PPOLinearSampler(max_steps=50, max_depth=100, n_best=-1, batch=256, gamma=gamma)
    @everywhere sampler = PPOTreeSampler(max_steps=100, max_depth=100, n_best=-1, batch=256, solution=solution, gamma=gamma)
    # @everywhere sampler = PolicyTreeSampler(max_steps=10, max_depth=100, n_best=-1, batch=256, solution=solution, gamma=gamma)
    # @everywhere sampler = QTreeSampler(max_steps=1000, max_depth=100, epsilon=1.0, eps_decay=0.80, n_best=-1, batch=256, gamma=gamma)
    # @everywhere sampler = TreeSampler(max_steps=1000, max_depth=100, epsilon=1.0, eps_decay=0.80, is_directed=false, n_best=-1, batch=128, gamma=gamma)
    # @everywhere sampler = DAGSampler(max_steps=1000, max_depth=100, epsilon=1.0, eps_decay=0.80, is_directed=false, n_best=-1, batch=128, gamma=0.95)
    # @everywhere sampler = DGSampler(max_steps=1000, max_depth=100, epsilon=1.0, eps_decay=0.80, is_directed=false, n_best=-1, batch=128, gamma=0.95)
    # @everywhere sampler = TreeSampler2Values(max_steps=1000, max_depth=100, epsilon=1.0, eps_decay=0.80, is_directed=false, n_best=-1, batch=128, gamma=0.99)
    samples = [(;ds=nothing,rew=[],goal_size=typemax(Int), initial_expr=i, depth=typemax(Int)) for i in data]
    #reading_time = @elapsed all_trees = get_existing_trees()
    #println("Have completed reading all trees in $(reading_time)")
    #@everywhere all_trees = $all_trees
    for e in 1:epochs
        # search_time = @elapsed samples = pmap(enumerate(data)) do (ind, d)
		#     reading_time = @elapsed soltree = deserialize("data/planning_extracted/" * "solution_tree_from_planning_for_exp_$(ind).bin")
        #     new_soltree = Dict()
        #     for (k,v) in soltree
        #         new_soltree[k] = Node(intern!(v.ex), v.rule_index, v.children, v.parent, v.depth, k)
        #     end
        #     traj_time = @elapsed if is_target
        #         traj = compute_targets_from_existing_tree(new_soltree, target_model, sampler)
        #     else
        #         traj = compute_targets_from_existing_tree(new_soltree, sampler)
        #     end
        #     @show reading_time, traj_time, MyModule.cache_status()
        #     not_boosted(d, sampler, traj)
	    # end
        search_time = @elapsed samples = pmap(enumerate(samples)) do (ind,d)
            empty!(MyModule.nc)
            MyModule.reset_all_function_caches()
            traj_time = @elapsed if is_target
                traj = sample_trajectory(sampler, d.initial_expr, model, value_model)
            else
                traj = sample_trajectory(sampler, d.initial_expr, model)
            end
            @show traj_time, MyModule.cache_status()
            if is_boosted
                boosted(d, sampler, traj)
            else
                not_boosted(d, sampler, traj)
            end
        end
    	#=
        search_time = @elapsed samples = pmap(enumerate(samples)) do (ind,d)
            empty!(MyModule.nc)
            MyModule.reset_all_function_caches()
            traj_time = @elapsed if is_target
                traj = sample_trajectory(sampler, d.initial_expr, model, target_model)
            else
                traj = sample_trajectory(sampler, d.initial_expr, model)
            end
            @show traj_time, MyModule.cache_status()
	    
            if is_boosted
                boosted(d, sampler, traj)
            else
                not_boosted(d, sampler, traj)
            end
        end
        =#
        flattened_samples = []
        for i in samples
            if isa(i, NamedTuple)
                push!(flattened_samples, i)
            else
                append!(flattened_samples, i)
            end
        end
        loss_over_time = 0
        update_time = @elapsed for _ in 1:1
            learning_time = 0
            loss = 0
            for (ind, s) in enumerate(flattened_samples)
                if is_target && solution
                    learning_time += @elapsed loss += compute_gradient!(s.rew, s.ds, model, target_model, value_model, learner)
                elseif is_target && !solution
                    learning_time += @elapsed loss += compute_gradient1!(s, model, target_model, value_model, learner)
                elseif !solution
                    learning_time += @elapsed loss += compute_gradient1!(s, model, learner)
                else
                    learning_time += @elapsed loss += compute_gradient!(s.rew, s.ds, model, learner)
                end
            end
            loss_over_time += loss / length(samples)
        end
        update_epsilon!(sampler)
        if mod(e, 4) == 0
            target_model = deepcopy(model)
        end
        res = mean(map(x->exp_size(intern!(x.initial_expr)) - x.goal_size, flattened_samples))
        val_time = 0.0
        #serialize(save_model_path * "trained_parallel_2Values_not_boosted_ep$(e)_batch128_for_graph_stats.bin", model)
        #serialize(save_model_path * "trained_parallel_DQN_first_DAG_not_boosted_ep$(e)_batch128_gamma95_for_graph_stats.bin", model)
        serialize("models/trained_parallel_" * model_name * "ep$(e)_batch$(sampler.batch)_gamma$(gamma_name)_for_graph_stats.bin", model)
	    #serialize("models/trained_parallel_DQN_first_Tree_from_planning_ep$(e)_batch128_gamma$(gamma_name).bin", model)
        #serialize(save_model_path * "trained_parallel_DQN_second_DG_boosted_ep$(e)_batch128_for_graph_stats.bin", model)
        #serialize(save_model_path * "trained_parallel_DQN_first_DAG_not_boosted_ep$(e)_batch128_for_graph_stats.bin", model)
        # println("Ep $(e): lres, tres = $([0, res]); loss = $(loss_over_time / 10);epsilon=$(round(sampler.epsilon, digits=2)); trajectory took --> $(round(search_time, digits=2)); update took --> $(round(update_time, digits=2)); validation took --> $(round(val_time, digits=2))")
        println("Ep $(e): lres, tres = $([0, res]); loss = $(loss_over_time / e);epsilon=No; trajectory took --> $(round(search_time, digits=2)); update took --> $(round(update_time, digits=2)); validation took --> $(round(val_time, digits=2))")
    end
end
train(model, data[1:10], target, value_model)

function both_stats(data, model, names; max_exp=100, max_depth=100)
    df = pmap(data) do ex        
        t = @elapsed soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(intern!(ex), model; max_expansions=max_exp, max_depth=max_depth, epsilon=0.0)
        all_inner_nodes = filter(i->length(i.children) != 0, collect(values(soltree)))
	filtered_stds = [1, 1]
        println("ind=, ex=$(ex), time=$(t)")
        (; s₀ = MyModule.exp_size(root.ex), sₙ = MyModule.exp_size(smallest_node.ex), se = expr(MyModule.nc,smallest_node.ex), pr = [], mean_std=mean(filtered_stds), filtered_total_nodes=length(filtered_stds), total_nodes=length(all_inner_nodes))
    end |> DataFrame
    return df
end
#max_exp = 1000
#df = both_stats(data, exp_size, "";max_exp=max_exp)
#CSV.write("stats/results_of_greedy_search_step$(max_exp).csv", df[!, [1, 2, 3, 4]])
#=
for e in 1:20
model = deserialize("models/trained_parallel_2Values_not_boosted_ep$(e)_batch128_for_graph_stats.bin")
df = both_stats(data, model, "")
CSV.write("stats/results_of_test_2Values_ep$(e)_hidden$(hidden_size).csv", df[!, [1, 2, 3, 4]])
end
=#
