function update_ddqn!(target_values, input_values, policy_network, params, loss::Function)
    sa, grad = Flux.Zygote.withgradient(policy_network) do oq
        expected_values = vec(MyModule.heuristic(oq, input_values))
        loss(target_values, expected_values)
    end
    Optimisers.update!(params, policy_network, grad[1])
    return sa
end


function update_pipeline_computed_target_values(buffer, online_policy, target_policy, params, batch_size=64, max_iter=1)
    sampled_experiences = sample(buffer, batch_size)
    next_states = [expr(MyModule.nc, i[2]) for i in sampled_experiences]
    rewards = [i[3] for i in sampled_experiences]
    deduped_next_states = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(next_states, sym_enc))
    # Todo: 
    # target_values = target_policy(...)
    for i in 1:max_iter
        loss = update_ddqn!(target_values, deduped_next_states, online_policy, params, Flux.mse)
    end
end


sample(buffer::Vector, batch_size) = buffer
function update_pipeline(buffer, online_policy, target_policy, params; batch_size=64, max_iter=1)
    sampled_experiences = sample(buffer, batch_size)
    next_states = [expr(MyModule.nc, i[2]) for i in sampled_experiences]
    rewards = [i[3] for i in sampled_experiences]
    deduped_next_states = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(next_states, sym_enc))
    target_values = rewards
    for i in 1:max_iter
        loss = update_ddqn!(target_values, deduped_next_states, online_policy, params, Flux.mse)
    end
end


function sample_trajectories(env::MyTreeEnv, policy, max_steps; epsilon=1.0, gamma=0.95)
    trajectories = Tuple[]
    trajectory_time = @elapsed for nth_step in 1:max_steps
        current_possible_actions = action_space(env)
        if epsilon <= rand()
            o = map(current_possible_actions) do i
                only(policy(i))
            end
            max_ind = argmax(o)
            a = current_possible_actions[max_ind]
        else
            a = rand(current_possible_actions)
        end
        s = state(env)
        act!(env, a)
        r = reward(env)
        ns = state(env)
        is_done = is_terminal(env) || isempty(action_space(env)) || nth_step == max_steps
        push!(trajectories, (s, a, r, ns, is_done))
        if is_done
            return trajectories
        end
    end
end


function leaf2trajectorie(leaf::Node, soltree::Dict, root, mcache, trajectories=[])
    if leaf.node_id == leaf.parent
        return
    end
    s = soltree[leaf.parent]
	r = mcache[leaf.ex]
    ns = leaf.ex
    is_done = isempty(trajectories) ? true : false
    push!(trajectories, (s, ns, r, ns, is_done))
    leaf2trajectorie(soltree[leaf.parent], soltree, root, mcache, trajectories)
    return trajectories
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


function target_from_cache2!(cache, leaf::Node, soltree::Dict; gamma=0.99)
    # haskey(cache, leaf.ex) && return(cache[leaf.ex])
    r = exp_size(leaf.ex)
    root_id = findfirst(x->x.depth == 0, soltree)
    # root_id = leaf.parent
    v = isleaf(leaf) ? (exp_size(soltree[root_id].ex) - r) : maximum(exp_size(soltree[leaf.parent].ex) - r + target_from_cache2!(cache, soltree[ch], soltree) for ch in leaf.children)
    cache[leaf.ex] = v
    return(v)
end


function sample_trajectories(d::NodeID, policy, is_dag; max_steps=50, max_depth=100, gamma=0.95, epsilon=1.0)
    # search_time = @elapsed soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search(d, policy; max_expansions=max_steps, max_depth=max_depth)
    soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(d, policy; max_expansions=max_steps, max_depth=max_depth, epsilon=epsilon)
    if is_dag
        soltree = soltree1
    end
    mcache = Dict()
    root_id = findfirst(x->x.depth == 0, soltree)
    if is_dag
        target_from_cache2!(mcache, soltree[root_id], soltree)
    else
        target_from_cache!(mcache, soltree[root_id], soltree)
    end
    all_leafs = filter(i->length(i.second.children) == 0, soltree)
    @assert length(mcache) == length(soltree)
    trajectories = [leaf2trajectorie(lf, soltree, root, mcache) for (nid, lf) in all_leafs]
    sorted_trajectories = sort(trajectories, by=x->exp_size(root.ex) - exp_size(x[1][2]))
    # return sorted_trajectories
    # @show last(sorted_trajectories)[1][2]
    return sorted_trajectories
end


function validation(data, online_policy, max_steps)
    trajectories = fetch.([Threads.@spawn sample_trajectories(MyTreeEnv(d, online_policy), online_policy, max_steps; epsilon=0.0, ) for d in data])
    v = map(i->findmax(x->exp_size(i[1][1]) - exp_size(x[4]), i)[1], trajectories)
    return v
end

update_epsilon(epsilon; eps_decay=0.95, eps_min=0.1) = max(eps_min, eps_decay * epsilon) == eps_min ? 0.0 : max(eps_min, eps_decay * epsilon)
function train(data, online_policy, target_policy; lr=0.0003, batch_size=30, max_steps=50, max_depth=100, epochs=10, update_iter=1, epsilon=1.0)
    pol_optimizer = ADAM(lr)
    policy_params = Flux.setup(pol_optimizer, online_policy)
    MyModule.reset_all_function_caches()
    env = MyTreeEnv(data[1], online_policy)
    for ep in 1:epochs
        trajectory_time = @elapsed trajectories = map(data) do d
            env.s_init = intern!(d)
            env = reset!(env)
            trajectory = sample_trajectories(env.s_init, online_policy, false; max_steps=max_steps, max_depth=max_depth, epsilon=epsilon)
            trajectory = vcat(last(trajectory, 3)...)
        end
        epsilon = update_epsilon(epsilon)
        @show trajectory_time
        trajectories = vcat(trajectories...)
        @show length(trajectories)
        update_time = @elapsed update_pipeline(trajectories, online_policy, target_policy, policy_params; batch_size=batch_size, max_iter=update_iter)
        empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
        v = validation(data, online_policy, max_steps)
        println("Ep$(ep): rew=$(mean(v)); time-->$(update_time)")
    end
end


function ()
    
end