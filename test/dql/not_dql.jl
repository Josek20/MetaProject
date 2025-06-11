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
using MyModule: all_expand, exp_size, Node, NodeID, intern!, expr
using Serialization
include("my_test_env.jl")
include("tree_env_setup.jl")


function visualization(soltree, online_policy, target_values::Dict)
    empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
    children = Vector[]
    text = []
    link_style = [""]
    style = [""]
    index_to_put = 0
    current_i = 1
    buff = []
    sort_by_depth = sort(collect(values(soltree)), by=x->x.depth)
    i = sort_by_depth[current_i]

    append!(buff, [soltree[j] for j in i.children])
    # push!(children, [soltree[j].ex for j in i.children])
    current_range = 2:length(i.children) + 1
    push!(children, collect(Any, current_range))
    push!(text, string(expr(MyModule.nc, i.ex)))
    # append!(text, [string(expr(MyModule.nc, soltree[j].ex)) for j in i.children])
    while !isempty(buff) 
    # for _ in 1:43
        n = popfirst!(buff)
        # @show n.ex
        # r = exp_size(soltree[n.parent].ex) - exp_size(n.ex)
        r = exp_size(i.ex) - exp_size(n.ex)
        # r = target_values[n.ex]
        push!(text, string(expr(MyModule.nc, n.ex)) * "\nPred:$(round(only(online_policy(n.ex)), digits=4))\nRew:$(r)")
        # push!(text, string(expr(MyModule.nc, n.ex)) * "\nRew:$(r)")
        current_range = current_range.stop + 1:current_range.stop + length(n.children)
        push!(children, collect(Any, current_range))
        append!(buff, [soltree[j] for j in n.children])
    end
    t = D3Tree(children, text=text, init_expand=30)
    # t = D3Tree(children[2:end], text=text, style=style, link_style=link_style, init_expand=2,  svg_node_size=(2020, 2020))
    inbrowser(t, "Mircosoft Edge")
end
function visualization(soltree, online_policy, target_values::Dict, solution)
    empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
    children = Vector[]
    text = []
    link_style = [""]
    style = [""]
    index_to_put = 0
    current_i = 1
    buff = []
    sort_by_depth = sort(collect(values(soltree)), by=x->x.depth)
    i = sort_by_depth[current_i]

    append!(buff, [soltree[j] for j in i.children])
    # push!(children, [soltree[j].ex for j in i.children])
    current_range = 2:length(i.children) + 1
    push!(children, collect(Any, current_range))
    push!(text, string(expr(MyModule.nc, i.ex)))
    # append!(text, [string(expr(MyModule.nc, soltree[j].ex)) for j in i.children])
    while !isempty(buff) 
    # for _ in 1:43
        n = popfirst!(buff)
        # @show n.ex
        # r = exp_size(soltree[n.parent].ex) - exp_size(n.ex)
        # r = exp_size(i.ex) - exp_size(n.ex)
        r = target_values[n.ex]
        if n.ex in solution
            push!(text, string(expr(MyModule.nc, n.ex)) * "\nPred:$(round(only(online_policy(n.ex)), digits=4))\nRew:$(r)")
        else
            # push!(text, "\nPred:$(round(only(online_policy(n.ex)), digits=4))\nRew:$(r)")
            push!(text, "")
        end
        # push!(text, string(expr(MyModule.nc, n.ex)) * "\nRew:$(r)")
        current_range = current_range.stop + 1:current_range.stop + length(n.children)
        push!(children, collect(Any, current_range))
        append!(buff, [soltree[j] for j in n.children])
    end
    t = D3Tree(children, text=text, init_expand=30)
    # t = D3Tree(children[2:end], text=text, style=style, link_style=link_style, init_expand=2,  svg_node_size=(2020, 2020))
    inbrowser(t, "Mircosoft Edge")
end
function update_ddqn!(buffer, online_q, target_q, policy_params_optimiser; batch_size=64, gamma=0.99, max_iter=10)
    # sampled_experiences = sample(buffer, batch_size)
    sampled_experiences = buffer
    # sampled_experiences = vcat(values(buffer)...)
    # states = [expr(MyModule.nc, i[1]) for i in sampled_experiences]
    states = [i[1] for i in sampled_experiences]
    rewards = [i[3] for i in sampled_experiences]
    next_states1 = [expr(MyModule.nc, i[2]) for i in sampled_experiences]
    # next_states1 = [i[2] for i in sampled_experiences]
    # @show next_states1
    # for (i,j) in zip(next_states1, rewards)
    #     @show i, j 
    # end
    next_possible_states = [[expr(MyModule.nc, j) for j in i[6]] for i in sampled_experiences]
    # next_possible_states = [[j for j in i[6]] for i in sampled_experiences]
    # is_dones = [i[4] for i in sampled_experiences]
    # states = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(states, sym_enc))
    # @show length(next_states1)
    
    deduped_next_states1 = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(next_states1, sym_enc))
    next_states = [isempty(i) ? 0 : MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(i)) for i in next_possible_states]
    # @show next_possible_states

    rws = [i[7] for i in sampled_experiences]
    target_values = rewards
    # target_values = [i == 0 ? r : maximum(vec(MyModule.heuristic(target_q, i)) + r_a) for (i, r_a, r, ns1) in zip(next_states, rws, rewards, next_states1)]
    # target_values = [i == 0 ? target_q(ns1) : maximum(vec(MyModule.heuristic(target_q, i)) + r_a) for (i, r_a, r, ns1) in zip(next_states, rws, rewards, next_states1)]
    # next_q = vec(MyModule.heuristic(target_q, deduped_next_states1))
    # target_values = rewards + gamma * next_q
    # @show target_values
    # println("Before=====")
    # for (i,k,r) in zip(next_states1, target_values, rewards)
    #     println("$(i) -> $(round(k, digits=4)) -- $(r) =")
    # end
    # target_values = rewards + gamma * vec(MyModule.heuristic(target_q, next_states))# .* (1 .- is_dones) 
    
    # states = hcat(states...)
    mean_sa = []
    nodes_values_progress = []
    for k in 1:max_iter
        
        sa, grad = Flux.Zygote.withgradient(online_q) do oq
            expected_values = vec(MyModule.heuristic(oq, deduped_next_states1))
            loss = mean((target_values - expected_values).^ 2)
            return loss
        end
        Optimisers.update!(policy_params_optimiser, online_q, grad[1])
        push!(mean_sa, sa)
        push!(nodes_values_progress, vec(MyModule.heuristic(online_q, deduped_next_states1)))
        # if mod(k, 10) == 0
        #     target_q = deepcopy(online_q)
        #     # next_q = vec(MyModule.heuristic(target_q, deduped_next_states1))
        #     # target_values = rewards + gamma * next_q
        #     # target_values = [i == 0 ? r : maximum(vec(MyModule.heuristic(target_q, i)) + r_a) for (i, r_a, r) in zip(next_states, rws, rewards)]
        #     # empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
        #     # target_values = [i == 0 ? target_q(ns1) : maximum(vec(MyModule.heuristic(target_q, i)) + r_a) for (i, r_a, r, ns1) in zip(next_states, rws, rewards, next_states1)]

        #     # @show target_values
        # end
    end
    # println("After=====")
    # target_values = [i == 0 ? k : maximum(vec(MyModule.heuristic(online_q, i)) + j) for (i, j, k) in zip(next_states, rws, rewards)]
    # for (i,k,r) in zip(next_states1, target_values, rewards)
    #     println("$(i) -> $(round(k, digits=4)) -- $(r)")
    # end
    # return mean_sa, nodes_values_progress
    return mean_sa, nodes_values_progress
end

function validation(data, online_policy, max_steps)
    trajectories = fetch.([Threads.@spawn sample_trajectory(MyTreeEnv(d, online_policy), online_policy, epsilon=0.0, max_steps=max_steps) for d in data])
    v = map(i->findmax(x->exp_size(i[1][1]) - exp_size(x[4]), i)[1], trajectories)
    # @show mean(v)
    return v
end
function sample_trajectory(env, policy; epsilon=1.0, max_steps=50, gamma=0.95)
    tranjectories = Tuple[]
    tranjectory_time = @elapsed for nth_step in 1:max_steps
        possible_actions = action_space(env)
        if epsilon <= rand()
            o = map(possible_actions) do i
                only(policy(i))
            end
            max_ind = argmax(o)
            a = possible_actions[max_ind]
        else
            a = rand(possible_actions)
        end
        s = state(env)
        act!(env, a)
        r = reward(env, a, s)
        # r = (exp_size(s) - exp_size(a)) ^ (gamma * nth_step)
        # rews = exp_size(s) .- exp_size.(possible_actions)
        ns = state(env)
        is_done = is_terminal(env) || isempty(action_space(env))
        possible_actions = action_space(env)
        push!(tranjectories, (s, a, r, ns, is_done, possible_actions))
        if is_done
            # empty!(tranjectories[end][end-1])
            return tranjectories
        end
    end
    # @show tranjectory_time
    # empty!(tranjectories[end][end-1])
    return tranjectories
end
isleaf(n::Node) = isempty(n.children)
function target_from_cache!(cache, leaf::Node, soltree::Dict)
    haskey(cache, leaf.ex) && return(cache[leaf.ex])
    r = exp_size(leaf.ex)
    root_id = findfirst(x->x.depth == 0, soltree)
    # root_id = leaf.parent
    v = isleaf(leaf) ? (exp_size(soltree[root_id].ex) - r) : maximum(exp_size(soltree[leaf.parent].ex) - r + target_from_cache!(cache, soltree[ch], soltree) for ch in leaf.children)
    cache[leaf.ex] = v
    return(v)
end
function target_from_cache3!(cache, leaf::Node, soltree::Dict)
    # haskey(cache, leaf.ex) && return(cache[leaf.ex])
    if haskey(cache, leaf.ex) && cache[leaf.ex][1] < leaf.depth
        r = exp_size(leaf.ex)
        root_id = findfirst(x->x.depth == 0, soltree)
        v = isleaf(leaf) ? (exp_size(soltree[root_id].ex) - r) : maximum(exp_size(soltree[leaf.parent].ex) - r + target_from_cache3!(cache, soltree[ch], soltree) for ch in leaf.children)
        cache[leaf.ex] = (leaf.depth, v)
        return(v)
    elseif haskey(cache, leaf.ex)
        return(cache[leaf.ex][2])
    end
    r = exp_size(leaf.ex)
    root_id = findfirst(x->x.depth == 0, soltree)
    # root_id = leaf.parent
    v = isleaf(leaf) ? (exp_size(soltree[root_id].ex) - r) : maximum(exp_size(soltree[leaf.parent].ex) - r + target_from_cache3!(cache, soltree[ch], soltree) for ch in leaf.children)
    cache[leaf.ex] = (leaf.depth, v)
    return(v)
end
function target_from_cache2!(cache, leaf::Node, soltree::Dict; gamma=0.99)
    # haskey(cache, leaf.ex) && return(cache[leaf.ex])
    r = exp_size(leaf.ex)
    root_id = findfirst(x->x.depth == 0, soltree)
    # root_id = leaf.parent
    v = isleaf(leaf) ? (exp_size(soltree[root_id].ex) - r) : maximum(exp_size(soltree[leaf.parent].ex) - r + target_from_cache2!(cache, soltree[ch], soltree) for ch in leaf.children)
    cache[leaf.ex] = v  * gamma ^ leaf.depth
    return(v)
end
function target_from_cache4!(cache, leaf::Node, soltree::Dict; gamma=0.99)
    # haskey(cache, leaf.ex) && return(cache[leaf.ex])
    r = exp_size(leaf.ex)
    root_id = findfirst(x->x.depth == 0, soltree)
    # root_id = leaf.parent
    v = isleaf(leaf) ? (exp_size(soltree[root_id].ex) - r) : maximum(exp_size(soltree[leaf.parent].ex) - r + target_from_cache2!(cache, soltree[ch], soltree) for ch in leaf.children)
    cache[(leaf.ex, leaf.depth)] = v
    return(v)
end
function target_from_cache1!(cache, leaf::Node, soltree::Dict, online_policy)
    haskey(cache, leaf.ex) && return(cache[leaf.ex])
    r = exp_size(leaf.ex)
    root_id = findfirst(x->x.depth == 0, soltree)
    # root_id = leaf.parent
    # v = isleaf(leaf) ? exp_size(soltree[root_id].ex) - r : maximum(r - exp_size(soltree[ch].ex) + target_from_cache1!(cache, soltree[ch], soltree, online_policy) for ch in leaf.children)
    if isleaf(leaf)
        v = exp_size(soltree[root_id].ex) - r
        # println("Leaf: $(leaf.ex): $(v)")
    else
        v = -1000
        # println("CHildren iter: $(leaf.ex)")
        for ch in leaf.children
            
            target_from_cache1!(cache, soltree[ch], soltree, online_policy)
            v_tmp = r - exp_size(soltree[ch].ex) + only(online_policy(soltree[ch].ex))
            # println("    $(soltree[ch].ex): $(v_tmp), $(only(online_policy(soltree[ch].ex)))")

            if v < v_tmp
                v = v_tmp
            end
        end
    end
    cache[leaf.ex] = v
    return(v)
end


function leaf2trajectorie(leaf::Node, soltree::Dict, root, mcache, trajectories=[])
    if leaf.node_id == leaf.parent
        return
    end
    s = soltree[leaf.parent]
    chd = [soltree[k].ex for k in soltree[leaf.parent].children]
    size_current = exp_size(leaf.ex)
    size_init = exp_size(s.ex)
	r = mcache[leaf.ex]
	# r = mcache[leaf.node_id]
    rws = [mcache[soltree[ch].ex] for ch in leaf.children]
    # s = embed_ex(soltree[leaf.parent].ex, embedding_model)
    # ns = embed_ex(leaf.ex, embedding_model)
    ns = leaf.ex
    is_done = isempty(trajectories) ? true : false
    # if haskey(mcache, s.ex) && length(s.children) != 0
    #     update
    # elseif !haskey(mcache, s.ex) && length(s.children) != 0
    #     assigne
    # end

    # get!(mcache, s.ex) do 
    #     length(s.children) == 0 ? exp_size(s.ex) : maxmimum()
    # end
    push!(trajectories, (s, ns, r, ns, is_done, [soltree[k].ex for k in soltree[leaf.parent].children], rws, leaf.depth))
    leaf2trajectorie(soltree[leaf.parent], soltree, root, mcache, trajectories)
    return trajectories
end

function tree_train(data, online_policy, target_policy; batch_size=64, gamma=0.99, epsilon=1.0, trajectory_max_steps=100, epochs=10, lr=0.001, tracked_soltree=Nothing)
    pol_optimizer = ADAM(lr)
    policy_params = Flux.setup(pol_optimizer, online_policy)
    MyModule.reset_all_function_caches()
    env = MyTreeEnv(data[1], online_policy)
    best_trajectories = Dict(i=>[] for i in data)
    stats_buffer = []
    loss_over_time = []
    # tracked_policies = []
    soltree = Dict()
    first_trees = Dict()
    # if !isnothing(tracked_soltree)
    #     tracked_soltree_progress = Dict(0=>[only(online_policy(i.ex)) for i in values(tracked_soltree)])
    # end
    target_update = 0
    for ep in 1:epochs
        rew_tmp_buffer = []
        loss_over_time_tmp = []
        training_time = @elapsed trajectories = map(data) do d
            search_time = @elapsed soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search(intern!(d), online_policy; max_expansions=trajectory_max_steps, max_depth=100)
            soltree1 = soltree
            mcache = Dict()
            root_id = findfirst(x->x.depth == 0, soltree1)
            target_from_cache!(mcache, soltree1[root_id], soltree1)
            # target_from_cache1!(mcache, root, soltree1, online_policy)
            all_leafs = filter(i->length(i.second.children) == 0, soltree1)
            @assert length(mcache) == length(soltree)
            trajectories = [leaf2trajectorie(lf, soltree1, root, mcache) for (nid, lf) in all_leafs]
            # sorted_trajectories = sort(trajectories, by=x->(exp_size(root.ex) - exp_size(x[end][2]))/x[end][end])
            sorted_trajectories = sort(trajectories, by=x->exp_size(root.ex) - exp_size(x[end][2]))
            trajectories = vcat(last(sorted_trajectories, 3)...)
            # trajectories = vcat(last(sorted_trajectories, length(sorted_trajectories))...)
            # trajectories = vcat(trajectories...)
        end
        @show training_time
            # println("Search time $(search_time); number of transitions = $(length(trajectories))); epsilon = $(epsilon)")
            # loss_values = update_ddqn!(trajectories, online_policy, target_policy, policy_params; batch_size=batch_size, gamma=gamma, max_iter=400)
        # trajectories = vcat(trajectories[(ep-1)*100 + 1:ep*100]...)
        trajectories = vcat(trajectories...)
        @show length(trajectories)
        training_time = @elapsed loss_values, nodes_progress = update_ddqn!(trajectories, online_policy, target_policy, policy_params; batch_size=batch_size, gamma=gamma, max_iter=10)
        empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
        # target_update += 1
        # if mod(target_update, 2) == 0
        #     target_update = 0
        #     target_policy = deepcopy(online_policy)
        # end
        push!(loss_over_time_tmp, mean(loss_values))
        # end
        # if !isnothing(tracked_soltree)
        #     tracked_soltree_progress[ep] = [only(online_policy(i.ex)) for i in values(tracked_soltree)]
        # end

        push!(loss_over_time, mean(loss_over_time_tmp))
        push!(stats_buffer, mean(validation(data, online_policy, trajectory_max_steps)))
        println("Epoch $(ep): took --> $(training_time); rew = $(stats_buffer[end]); loss = $(mean(loss_over_time))")
        epsilon = update_epsilon(epsilon)
    end
    soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search(intern!(data[1]), online_policy; max_expansions=trajectory_max_steps, max_depth=100)
    return stats_buffer, loss_over_time, soltree, soltree1, []
end
function parallel_train(data, online_policy, target_policy; batch_size=64, gamma=0.99, epsilon=1.0, trajectory_max_steps=100, epochs=10, lr=0.0003)
    pol_optimizer = ADAM(lr)
    policy_params = Flux.setup(pol_optimizer, online_policy)
    MyModule.reset_all_function_caches()
    env = MyTreeEnv(data[1], online_policy)
    best_trajectories = Dict(i=>[] for i in data)
    stats_buffer = []
    loss_over_time = []
    for ep in 1:epochs
        rew_tmp_buffer = []
        loss_over_time_tmp = []
        # MyModule.reset_all_function_caches()

        trajectory_collection_time = @elapsed trajectories = fetch.([Threads.@spawn sample_trajectory(MyTreeEnv(d, online_policy), online_policy, epsilon=epsilon, max_steps=trajectory_max_steps) for d in data])
        best_pos = map(i->findmax(x->exp_size(i[1][1]) - exp_size(x[4]), i)[2], trajectories)
        for (ind,(i, d)) in enumerate(zip(best_pos, data))
            if ep == 1
                best_trajectories[d] = trajectories[ind][1:i]
            else
                # s, a, r, s', is_done, possible_actions
                current_best = exp_size(best_trajectories[d][end][4])
                current = exp_size(trajectories[ind][i][4])
                if current_best > current
                    best_trajectories[d] = trajectories[ind][1:i]
                elseif current_best == current && length(best_trajectories[d]) > length(trajectories[ind][1:i])
                    best_trajectories[d] = trajectories[ind][1:i]
                end
            end
        end
        v = map(i->findmax(x->exp_size(i[1][1]) - exp_size(x[4]), i)[1], trajectories)
        @show trajectory_collection_time
        # @show length(trajectories[1])
        trajectories = vcat(trajectories...)
        # trajectories = vcat(values(best_trajectories)...)
        println("Collected trajectories in $(trajectory_collection_time); number of transitions = $(length(trajectories))); epsilon = $(epsilon)")
        @show trajectories
        # training_time = @elapsed loss_values = update_ddqn!(shuffle(trajectories)[1:batch_size], online_policy, target_policy, policy_params; batch_size=batch_size, gamma=gamma, max_iter=100)
        training_time = @elapsed loss_values, _ = update_ddqn!(trajectories, online_policy, target_policy, policy_params; batch_size=batch_size, gamma=gamma, max_iter=10)
        empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
        epsilon = update_epsilon(epsilon)
        push!(stats_buffer, mean(v))
        push!(loss_over_time, loss_values)
        # println("Epoch $(ep): took --> $(epoch_time); rew = $(mean(rew_tmp_buffer[end])); loss = $(mean(loss_over_time_tmp))")
        println("Epoch $(ep): took --> $(training_time + trajectory_collection_time); rew = $(stats_buffer[end]); loss = $(mean(loss_over_time))")
    end
    # serialize("models/trained_dqn_policy.bin", online_policy)
    return stats_buffer, loss_over_time, best_trajectories, []
end
function train(data, online_policy, target_policy; batch_size=64, gamma=0.99, epsilon=1.0, trajectory_max_steps=100, epochs=10, lr=0.003)
    pol_optimizer = ADAM(lr)
    policy_params = Flux.setup(pol_optimizer, online_policy)
    MyModule.reset_all_function_caches()
    env = MyTreeEnv(data[1], online_policy)
    stats_buffer = []
    loss_over_time = []
    for ep in 1:epochs
        rew_tmp_buffer = []
        loss_over_time_tmp = []

        epoch_time = @elapsed for d in data
            env.s_init = intern!(d)
            env = reset!(env)
            trajectory = sample_trajectory(env, online_policy, epsilon=epsilon, max_steps=10)
            v, min_ind = findmax(x->exp_size(env.s_init) - exp_size(x[4]), trajectory)
            push!(rew_tmp_buffer, v)
            # loss_values = update_policy!(online_policy, target_policy, trajectory, policy_params, batch_size=batch_size, gamma=gamma)
            loss_values, _ = update_ddqn!(trajectory, online_policy, target_policy, policy_params; batch_size=batch_size, gamma=gamma, max_iter=1)
            empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
            push!(loss_over_time_tmp, loss_values)
        end
        epsilon = update_epsilon(epsilon)
        push!(stats_buffer, mean(rew_tmp_buffer))
        push!(loss_over_time, mean(loss_over_time_tmp))
        # println("Epoch $(ep): took --> $(epoch_time); rew = $(mean(rew_tmp_buffer[end])); loss = $(mean(loss_over_time_tmp))")
        println("Epoch $(ep): took --> $(epoch_time); rew = $(rew_tmp_buffer[end]); loss = $(mean(loss_over_time_tmp))")
    end
    serialize("models/trained_dqn_policy.bin", online_policy)
    return stats_buffer, loss_over_time
end

# init the network and its parameters
hidden_size=64
# epsilone_decay = 0.9 / 2.5e2
epsilon = eps_start = 1.0
eps_end = 0.01
eps_decay = 0.95

input_size = 64

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


stats_buffer, loss_over_time, best_trajectories, soltree1, tracked_soltree_progress = tree_train(data[1:100], online_policy, target_policy, batch_size=8000, epochs=1, trajectory_max_steps=50, epsilon=0.0, lr=0.001)

# stats_buffer, loss_over_time, best_trajectories, tracked_soltree_progress = parallel_train(data[1:1], online_policy, online_policy, batch_size=8000,
#                                     epochs=100, trajectory_max_steps=20, epsilon=0.0)
# best_trajectories = soltree1
# mcache = Dict()
# root = findfirst(x->x.depth == 0, best_trajectories)
# empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
# # target_from_cache1!(mcache, best_trajectories[root], best_trajectories, online_policy)
# target_from_cache!(mcache, best_trajectories[root], best_trajectories)
# smallest_node = sort(collect(values(best_trajectories)), by=x->exp_size(x.ex))[1]
# nodes_in_proof, proof = MyModule.extract_proof(smallest_node, best_trajectories)
# visualization(best_trajectories, online_policy, mcache, [i.ex for i in nodes_in_proof])

# visualization(best_trajectories, online_policy, mcache)
# soltree1 = best_trajectories
# pol_optimizer = ADAM(0.003)
# root = soltree1[findfirst(x->x.depth==0, soltree1)]
# policy_params = Flux.setup(pol_optimizer, online_policy)
# all_leafs = filter(i->length(i.second.children) == 0, soltree1)
# trajectories = [leaf2trajectorie(lf, soltree1, root, mcache) for (nid, lf) in all_leafs]
# trajectories = vcat(trajectories...)
# for _ in 1:500

#     loss_values, nodes_progress = update_ddqn!(trajectories, online_policy, online_policy, policy_params; batch_size=8000, gamma=0.99, max_iter=10)
#     mcache = Dict()
#     empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
#     target_from_cache2!(mcache, root, best_trajectories, online_policy)
#     trajectories = [leaf2trajectorie(lf, soltree1, root, mcache) for (nid, lf) in all_leafs]
#     trajectories = vcat(trajectories...)
# end
# @show loss_values
# visualization(best_trajectories, online_policy, mcache)
# plot(hcat(nodes_progress...)', legend=false)

# Test 
# trajectories1 = [
#     [(:(v0 <= v0 - v1 + v1), :(v0 - v1 <= v0 - v1), 0, :(v0 - v1 <= v0 - v1), false, [:(min(v0 - v1, v0 - v1) == v0 - v1), :(1)], [-4, 6]), (:(v0 - v1 <= v0 - v1), :(1), 6, :(1), false, [], [])],
#     [(:(v0 <= v0 - v1 + v1), :(v0 - v1 <= v0 - v1), 0, :(v0 - v1 <= v0 - v1), false, [:(min(v0 - v1, v0 - v1) == v0 - v1), :(1)], [-4, 6]), (:(v0 - v1 <= v0 - v1), :(min(v0 - v1, v0 - v1) == v0 - v1), -4, :(min(v0 - v1, v0 - v1) == v0 - v1), false, [], [])],
#     [(:(v0 <= v0 - v1 + v1), :(v0 <= v0), 4, :(v0 <= v0), false, [], [])],
# ]
# trajectories = [
#     [(:(v0 <= v0 - v1 + v1), :(v0 <= v0), 4, :(v0 <= v0), false)],
#     [(:(v0 <= v0 - v1 + v1), :(v0 - v1 <= v0 - v1), 0, :(v0 - v1 <= v0 - v1), false), (:(v0 - v1 <= v0 - v1), :(1), 6, :(1), false)],
#     [(:(v0 <= v0 - v1 + v1), :(v0 - v1 <= v0 - v1), 0, :(v0 - v1 <= v0 - v1), false), (:(v0 - v1 <= v0 - v1), :(min(v0 - v1, v0 - v1) == v0 - v1), -4, :(min(v0 - v1, v0 - v1) == v0 - v1), false)],
# ]
# MyModule.expr(nc::MyModule.NodeCache, ex::Union{Expr, Int}) = ex

# # for i in 1:1
# online_q = Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
# target_q = Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
# target_q = deepcopy(online_q)
# online_policy = ExprModel(
#     head_model,
#     Mill.SegmentedSum(hidden_size),
#     args_model,
#     online_q
#     );

# target_policy = ExprModel(
#     head_model,
#     Mill.SegmentedSum(hidden_size),
#     args_model,
#     target_q
#     );


# pol_optimizer = ADAM()
# policy_params = Flux.setup(pol_optimizer, online_policy)
# tmp = Dict(i[end][2]=>[] for i in trajectories1)
# MyModule.reset_all_function_caches()
# for i in 1:1
#     for (ind, trajectory) in enumerate(shuffle(trajectories1))
#         # ls = update_policy!(online_policy, target_policy, trajectory, policy_params; batch_size=64, gamma=0.99, sym_enc=sym_enc, max_iter=100)
#         ls, _ = update_ddqn!(trajectory, online_policy, online_policy, policy_params; batch_size=64, gamma=0.99, max_iter=100)
#         push!(tmp[trajectory[end][2]], ls)
#     end
# end
# @show tmp
# #     o = [only(i) for i in online_policy.(all_states)]
# #     @show o
# #     @show o[1] > o[2]
# #     @show o[4] > o[3]
# # end
# all_states = [:(v0 - v1 <= v0 - v1), :(v0 <= v0), :(min(v0 - v1, v0 - v1) == v0 - v1), :(1)]
# rewards = [0, 4, -4, 6]

# o = [only(i) for i in online_policy.(all_states)]

# children = Vector[[2, 3], [4, 5], [], [], []]
# text = ["$(string(:(v0 <= v0 - v1 + v1)))", "$(string(:(v0 - v1 <= v0 - v1)))\nRew:$(rewards[1])\nPred:$(round(o[1], digits=3))", "$(string(:(v0 <= v0)))\nRew:$(rewards[2])\nPred:$(round(o[2],digits=3))", "$(string(:(min(v0 - v1, v0 - v1) == v0 - v1)))\nRew:$(rewards[3])\nPred:$(round(o[3], digits=3))", "$(string(:(1)))\nRew:$(rewards[4])\nPred:$(round(o[4], digits=3))"]
# link_style = ["", "stroke:blue", "", "", "stroke:blue"]
# style = ["", "fill:green", "", "", "fill:green"]
# # t = D3Tree(children, text=text, style=style, link_style=link_style, init_expand=2,  svg_node_size=(2020, 2020))
# t = D3Tree(children, text=text, style=style, link_style=link_style, init_expand=2,)
# inbrowser(t, "Mircosoft Edge")