abstract type AbstractTrajectory end

mutable struct Trajectory{P,S,A,R,NS,D} <: AbstractTrajectory
    pointer::P
    states::S
    actions::A
    rewards::R
    next_states::NS
    is_dones::D
end
Trajectory() = Trajectory(1, NodeID[], NodeID[], Float32[], NodeID[], Bool[])
Trajectory(sampler::AbstractSampler) = Trajectory(1, NodeID[], NodeID[], Float32[], NodeID[], Bool[])

function push2traj!(trj::Trajectory, data::Tuple)
    s, a, r, ns, is_done = data
    push!(trj.states, s)
    push!(trj.actions, a)
    push!(trj.rewards, r)
    push!(trj.next_states, ns)
    push!(trj.is_dones, is_done)
end


mutable struct TreeTrajectory{P,S,A,R,NS,D,TD} <: AbstractTrajectory
    pointer::P
    states::S
    actions::A
    rewards::R
    next_states::NS
    is_dones::D
    soltree::TD
end
TreeTrajectory() = TreeTrajectory(1, Vector{Vector{NodeID}}(), Vector{Vector{NodeID}}(), Vector{Vector{Float32}}(), Vector{Vector{NodeID}}(), Vector{Vector{Bool}}(), Dict())
# function push2traj!(trj::TreeTrajectory, tree::Dict, mcache::Dict)

# end

mutable struct PolicyTrajectory{P,S,A,R,NS,D} <: AbstractTrajectory
    pointer::P
    states::S
    actions::A
    rewards::R
    next_states::NS
    is_dones::D
end
function push2traj!(trj::PolicyTrajectory, data::Tuple)
    s, a, r, ns, is_done = data
    push!(trj.states, s)
    push!(trj.actions, a)
    push!(trj.rewards, r)
    push!(trj.next_states, ns)
    push!(trj.is_dones, is_done)
end

function update_epsilon!(sampler::AbstractSampler)
    !hasfield(typeof(sampler), :epsilon) && return
    new_epsilon = max(0.05, sampler.eps_decay * sampler.epsilon) <= 0.05 ? 0.05 : max(0.05, sampler.eps_decay * sampler.epsilon)
    sampler.epsilon = new_epsilon
end


# Rename to Linear
# maybe just state and value 
mutable struct RLSampler <: AbstractSampler
    max_steps::Int
    epsilon::Float32
    eps_decay::Float32
    batch::Int
end
RLSampler(;max_steps=50, epsilon=0.0, eps_decay=0.95, batch=64) = RLSampler(max_steps, epsilon, eps_decay, batch)

mutable struct RLSampler1 <: AbstractSampler
    max_steps::Int
    epsilon::Float32
    eps_decay::Float32
    batch::Int
end
RLSampler1(;max_steps=50, epsilon=0.0, eps_decay=0.95, batch=64) = RLSampler1(max_steps, epsilon, eps_decay, batch)


mutable struct TreeSampler <: AbstractSampler
    max_steps::Int
    max_depth::Int
    epsilon::Float32
    eps_decay::Float32
    is_directed::Bool
    n_best::Int
    batch::Int
    gamma::Float32
end
TreeSampler(;max_steps=50, max_depth=100, epsilon=0.0, eps_decay=0.95, is_directed=false, n_best=-1, batch=64, gamma=1) = TreeSampler(max_steps, max_depth, epsilon, eps_decay, is_directed, n_best, batch, gamma)


mutable struct PlanningTreeSampler <: AbstractSampler
    max_steps::Int
    max_depth::Int
    epsilon::Float32
    eps_decay::Float32
    is_directed::Bool
    n_best::Int
    batch::Int
end
PlanningTreeSampler(;max_steps=50, max_depth=100, epsilon=0.0, eps_decay=0.95, is_directed=false, n_best=-1, batch=64) = PlanningTreeSampler(max_steps, max_depth, epsilon, eps_decay, is_directed, n_best, batch)


mutable struct TreeSampler2Values <: AbstractSampler
    max_steps::Int
    max_depth::Int
    epsilon::Float32
    eps_decay::Float32
    is_directed::Bool
    n_best::Int
    batch::Int
    gamma::Float32
end
TreeSampler2Values(;max_steps=50, max_depth=100, epsilon=0.0, eps_decay=0.95, is_directed=false, n_best=-1, batch=64, gamma=1.0) = TreeSampler2Values(max_steps, max_depth, epsilon, eps_decay, is_directed, n_best, batch, gamma)
Trajectory(sampler::TreeSampler2Values) =  Trajectory(1, NodeID[], NodeID[], Vector{Vector{Float32}}(), NodeID[], Bool[])


mutable struct DAGSampler <: AbstractSampler
    max_steps::Int
    max_depth::Int
    epsilon::Float32
    eps_decay::Float32
    is_directed::Bool
    n_best::Int
    batch::Int
    gamma::Float32
end
DAGSampler(;max_steps=50, max_depth=100, epsilon=0.0, eps_decay=0.95, is_directed=false, n_best=-1, batch=64, gamma=1) = DAGSampler(max_steps, max_depth, epsilon, eps_decay, is_directed, n_best, batch, gamma)


mutable struct DGSampler <: AbstractSampler
    max_steps::Int
    max_depth::Int
    epsilon::Float32
    eps_decay::Float32
    is_directed::Bool
    n_best::Int
    batch::Int
    gamma::Float32
end
DGSampler(;max_steps=50, max_depth=100, epsilon=0.0, eps_decay=0.95, is_directed=false, n_best=-1, batch=64, gamma=1) = DGSampler(max_steps, max_depth, epsilon, eps_decay, is_directed, n_best, batch, gamma)


mutable struct LinearSampler <: AbstractSampler
    max_steps::Int
    max_depth::Int
    epsilon::Float32
    eps_decay::Float32
    is_directed::Bool
    n_best::Int
    batch::Int
    gamma::Float32
end
LinearSampler(;max_steps=50, max_depth=100, epsilon=0.0, eps_decay=0.95, is_directed=false, n_best=-1, batch=64, gamma=1) = LinearSampler(max_steps, max_depth, epsilon, eps_decay, is_directed, n_best, batch, gamma)
Trajectory(sampler::LinearSampler) = Trajectory(1, NodeID[], Vector{Vector{NodeID}}(), Float32[], NodeID[], Bool[])


mutable struct Tree2TreeSampler <: AbstractSampler
    max_steps::Int
    max_depth::Int
    epsilon::Float32
    eps_decay::Float32
    is_directed::Bool
    n_best::Int
    batch::Int
    gamma::Float32
end
Tree2TreeSampler(;max_steps=50, max_depth=100, epsilon=0.0, eps_decay=0.95, is_directed=false, n_best=-1, batch=64, gamma=1) = Tree2TreeSampler(max_steps, max_depth, epsilon, eps_decay, is_directed, n_best, batch, gamma)


mutable struct PolicyTreeSampler <: AbstractSampler
    max_steps::Int
    max_depth::Int
    n_best::Int
    batch::Int
    solution::Bool
    gamma::Float32
end
PolicyTreeSampler(;max_steps=50, max_depth=100, n_best=-1, batch=64, solution=true, gamma=1) = PolicyTreeSampler(max_steps, max_depth, n_best, batch, solution, gamma)

mutable struct PPOTreeSampler <: AbstractSampler
    max_steps::Int
    max_depth::Int
    n_best::Int
    batch::Int
    solution::Bool
    gamma::Float32
end
PPOTreeSampler(;max_steps=50, max_depth=100, n_best=-1, batch=64, solution=true, gamma=1) = PPOTreeSampler(max_steps, max_depth, n_best, batch, solution, gamma)

mutable struct PolicyLinearSampler <: AbstractSampler
    max_steps::Int
    max_depth::Int
    n_best::Int
    batch::Int
    gamma::Float32
end
PolicyLinearSampler(;max_steps=50, max_depth=100, n_best=-1, batch=64, gamma=1) = PolicyLinearSampler(max_steps, max_depth, n_best, batch, gamma)

mutable struct PPOLinearSampler <: AbstractSampler
    max_steps::Int
    max_depth::Int
    n_best::Int
    batch::Int
    gamma::Float32
end
PPOLinearSampler(;max_steps=50, max_depth=100, n_best=-1, batch=64, gamma=1) = PPOLinearSampler(max_steps, max_depth, n_best, batch, gamma)

mutable struct QTreeSampler <: AbstractSampler
    max_steps::Int
    max_depth::Int
    epsilon::Float32
    eps_decay::Float32
    n_best::Int
    batch::Int
    gamma::Float32
end
QTreeSampler(;max_steps=50, max_depth=100, epsilon=0.0, eps_decay=0.95, n_best=-1, batch=64, gamma=1) = QTreeSampler(max_steps, max_depth, epsilon, eps_decay, n_best, batch, gamma)


mutable struct A2CSampler <: AbstractSampler
    max_steps::Int
    max_depth::Int
    batch::Int
    gamma::Float32
end
A2CSampler(;max_steps=50, max_depth=100, batch=64, gamma=1) = A2CSampler(max_steps, max_depth, batch, gamma)


function compute_linear_targets(traj::Trajectory, target_model::AbstractModel; gamma=1.0)
    results = Dict()
    root = traj.states[1]
    results[traj.next_states[end]] = exp_size(root) - exp_size(traj.next_states[end]) 
    for (s, pa) in zip(reverse(traj.states), reverse(traj.actions))
       results[s] = max(0, maximum(exp_size(root) - exp_size(ns) + gamma * only(target_model(ns)) for ns in pa)) 
    end
    # @show length(results), length(Set(traj.states))
    # @assert length(results) == length(Set(traj.states)) + 1
    trj = Trajectory()
    for (ind,(i, v)) in enumerate(results)
        push!(trj.states, i)
        push!(trj.actions, i)
        push!(trj.rewards, v)
        push!(trj.next_states, i)
        push!(trj.is_dones, ind == length(results))
    end
    return trj
end


function compute_linear_targets(traj::Trajectory; gamma=1.0)
    results = Dict()
    root = traj.states[1]
    results[traj.next_states[end]] = exp_size(root) - exp_size(traj.next_states[end]) 
    for (s, pa) in zip(reverse(traj.states), reverse(traj.actions))
    #    results[s] = max(0, maximum(exp_size(root) - exp_size(ns) + gamma * get!(results, ns, exp_size(root) - exp_size(ns)) for ns in pa)) 
        v = map(pa) do ns
            r = exp_size(root) - exp_size(ns)
            if haskey(results, ns)
                return(r + gamma * results[ns])
            else
                return(r + gamma * r)
            end
        end
        results[s] = max(0, maximum(v))
    end
    # @show length(results), length(Set(traj.states))
    # @assert length(results) == length(Set(traj.states)) + 1
    trj = Trajectory()
    for (ind,(i, v)) in enumerate(results)
        push!(trj.states, i)
        push!(trj.actions, i)
        push!(trj.rewards, v)
        push!(trj.next_states, i)
        push!(trj.is_dones, ind == length(results))
    end
    return trj
end

function sample_trajectory(sampler::A2CSampler, env::AbstractEnvironment, policy_model::AbstractModel, value_model::AbstractModel)::Trajectory
    traj = Trajectory()
    inputs_actions = []
    for t in 1:sampler.max_steps
        possible_positions = position_space(env)
        pos = [only(value_model(p)) for p in possible_positions]
        # possible_actions = action_space(env, pos)
        embeded_subtree = 2
        weights = vec(policy_model(embeded_subtree))
        node_index = StatsBase.sample(1:length(weights), Weights(softmax(weights)))
        a = possible_actions[node_index]
        s = state(env)
        act!(env, a)
        ns = state(env)
        r = reward(env, a, s)
        is_done = isempty(action_space(env)) || is_terminal(env)
        push2traj!(traj, (s, a, r, ns, is_done))
        other_indexes = setdiff(1:length(possible_actions), [node_index])
        # @show possible_actions[other_indexes]
        # @show a
        push!(inputs_actions, vcat(a, possible_actions[other_indexes]))
        if is_done
            break
        end
    end
    reset!(env)

end

function sample_trajectory(sampler::LinearSampler, env::AbstractEnvironment, model::AbstractModel)::Trajectory
    reset!(env)
    soltree = Dict()
    traj = Trajectory(sampler)
    for t in 1:sampler.max_steps
        possible_actions = action_space(env)
        if rand() >= sampler.epsilon
            o = [exp_size(env.s_init) - exp_size(pa) + sampler.gamma * only(model(pa)) for pa in possible_actions]
            a = possible_actions[argmax(o)]
        else
            a = rand(possible_actions)
        end
        # @show length(possible_actions)
        s = state(env)
        act!(env, a)
        ns = state(env)
        r = reward(env, a, s)
        is_done = isempty(action_space(env)) || is_terminal(env)
        push2traj!(traj, (s,possible_actions,r,ns,is_done))
        if is_done
            break
        end
    end
    if length(traj.states) >= sampler.batch
        traj.states = traj.states[1:sampler.batch]
        traj.next_states = traj.next_states[1:sampler.batch]
        traj.actions = traj.actions[1:sampler.batch]
    end
    return compute_linear_targets(traj, gamma=sampler.gamma)
end


function sample_trajectory(sampler::LinearSampler, env::AbstractEnvironment, model::AbstractModel, target_model::AbstractModel)::Trajectory
    reset!(env)
    soltree = Dict()
    traj = Trajectory(sampler)
    for t in 1:sampler.max_steps
        possible_actions = action_space(env)
        if rand() >= sampler.epsilon
            o = [exp_size(env.s_init) - exp_size(pa) + sampler.gamma * only(model(pa)) for pa in possible_actions]
            a = possible_actions[argmax(o)]
        else
            a = rand(possible_actions)
        end
        s = state(env)
        act!(env, a)
        ns = state(env)
        r = reward(env, a, s)
        is_done = isempty(action_space(env)) || is_terminal(env)
        push2traj!(traj, (s,possible_actions,r,ns,is_done))
        if is_done
            break
        end
    end
    reset!(env)
    # if length(traj.states) >= sampler.batch
    #     traj.states = traj.states[1:sampler.batch]
    #     traj.next_states = traj.next_states[1:sampler.batch]
    #     traj.actions = traj.actions[1:sampler.batch]
    # end
    return compute_linear_targets(traj, target_model, gamma=sampler.gamma)
end


function sample_trajectory(sampler::RLSampler1, env::AbstractEnvironment, model::AbstractModel)::PolicyTrajectory
    reset!(env)
    soltree = Dict()
    traj = PolicyTrajectory(1, NodeID[], Vector{Vector{NodeID}}(), Float32[], NodeID[], Bool[])
    for t in 1:sampler.max_steps
        possible_actions = action_space(env)
        if rand() >= sampler.epsilon
            o = [vec(model(pa)) for pa in possible_actions]
            a = possible_actions[argmax(o)]
        else
            a = rand(possible_actions)
        end
        s = state(env)
        act!(env, a)
        ns = state(env)
        r = reward(env, a, s)
        is_done = isempty(action_space(env)) || is_terminal(env)
        push2traj!(traj, (s,possible_actions,r,ns,is_done))
        if is_done
            break
        end
    end
    reset!(env)
    # if traj.rewards[end] == 0
    #     traj.rewards[end] = -100
    # end
    # best_node_id = argmax(x->exp_size(env.s_init) - exp_size(x), traj.next_states)
    # best_node_id = argmax([exp_size(env.s_init) - exp_size(x) for x in traj.next_states])

    # best_node_id = argmin(exp_size.(traj.next_states))
    # traj.states = traj.states[1:best_node_id]
    # traj.actions = traj.actions[1:best_node_id]
    # traj.rewards = traj.rewards[1:best_node_id]
    # traj.next_states = traj.next_states[1:best_node_id]
    # traj.is_dones = traj.is_dones[1:best_node_id]
    traj.rewards[traj.rewards .== 0] .= -1
    return traj
end


sample_trajectory(sampler::PPOLinearSampler, env::Expr, model::AbstractModel, value_model::AbstractModel) = sample_trajectory(sampler, MyTreeEnv(env, model), model, value_model)
function sample_trajectory(sampler::PPOLinearSampler, env::AbstractEnvironment, model::AbstractModel, value_model::AbstractModel)
    reset!(env)
    traj = Trajectory()
    inputs_actions = [[env.s_init]]
    start_time = time()
    total_action_time = 0
    rollout_time = @elapsed for t in 1:sampler.max_steps
        action_time = @elapsed possible_actions = action_space(env)
	#@show length(possible_actions), action_time
	total_action_time += action_time
        weights = [only(model(x)) for x in possible_actions]
        node_index = StatsBase.sample(1:length(weights), Weights(softmax(weights)))
        a = possible_actions[node_index]
        s = state(env)
        act!(env, a)
        ns = state(env)
        r = reward(env, a, s)
        is_done = isempty(action_space(env)) || is_terminal(env)
        push2traj!(traj, (s, a, r, ns, is_done))
        other_indexes = setdiff(1:length(possible_actions), [node_index])
        # @show possible_actions[other_indexes]
        # @show a
        push!(inputs_actions, vcat(a, possible_actions[other_indexes]))
        if is_done
            break
        end
        if time() - start_time > 10.0
            break
        end
    end
    println("Have finished the search took $(total_action_time), roll time $(rollout_time)")
    # possible_actions = action_space(env)
    # push!(inputs_actions, possible_actions)
    reset!(env)

    # @show traj.rewards
    returns = zeros(Float32, length(traj.rewards))
    inputs_set = []
    for t in length(traj.rewards):-1:1
        next_vl = map(x->only(value_model(x)), inputs_actions[t+1])
        current_vl = map(x->only(value_model(x)), inputs_actions[t])
        Aₜ = traj.rewards[t] + sampler.gamma * maximum(next_vl) - maximum(current_vl)
        returns[t] = Aₜ
        push!(inputs_set, inputs_actions[t][argmax(current_vl)])
    end
    tmp = [MyModule.expr(MyModule.nc, x) for x in inputs_set]
    took_time = @elapsed vl_ds = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(tmp, sym_enc))
    println("Have finished value targets processing $(length(tmp)), took $(took_time)")

    # prepare the actions
    inputs_set = []
    for i in vcat(inputs_actions...)
        if i in inputs_set
            continue
        else
            push!(inputs_set, i)
        end
    end

    tmp = [MyModule.expr(MyModule.nc, x) for x in inputs_set]
    took_time = @elapsed ds = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(tmp, sym_enc))
    inputs_ids = []
    for i in inputs_actions
        tmp = []
        for j in i
            push!(tmp, findfirst(==(j), inputs_set))
        end
        push!(inputs_ids, tmp)
    end
    println("Have finished actor targets processing $(length(tmp)), took $(took_time)")
    tmp = vcat(traj.states[1], traj.next_states)
    depth = argmin(exp_size.(tmp))
    node_ex = tmp[depth]
    # return (;rewards=returns, inputs=inputs, smallest_node=MyModule.Node(node_ex, (), UInt64[], hash(node_ex), depth, hash(node_ex)))
    return (;rewards=returns, inputs=(ds, vl_ds), smallest_node=MyModule.Node(node_ex, (), UInt64[], hash(node_ex), depth, hash(node_ex)), softmax_ids=inputs_ids, selected_ids=ones(Int32, length(inputs_ids)))
end


sample_trajectory(sampler::PolicyLinearSampler, env::Expr, model::AbstractModel) = sample_trajectory(sampler, MyTreeEnv(env, model), model)
function sample_trajectory(sampler::PolicyLinearSampler, env::AbstractEnvironment, model::AbstractModel)
    reset!(env)
    traj = Trajectory()
    inputs_actions = []
    for t in 1:sampler.max_steps
        @show env.s_init, env.s_current
        possible_actions = action_space(env)
        weights = [only(model(x)) for x in possible_actions]
        node_index = StatsBase.sample(1:length(weights), Weights(softmax(weights)))
        a = possible_actions[node_index]
        s = state(env)
        act!(env, a)
        ns = state(env)
        r = reward(env, a, s)
        is_done = isempty(action_space(env)) || is_terminal(env)
        push2traj!(traj, (s, a, r, ns, is_done))
        other_indexes = setdiff(1:length(possible_actions), [node_index])
        # @show possible_actions[other_indexes]
        # @show a
        push!(inputs_actions, vcat(deepcopy(a), possible_actions[other_indexes]))
        if is_done
            break
        end
    end
    reset!(env)

    # compute returns
    G = 0
    # @show traj.rewards
    returns = zeros(Float32, length(traj.rewards))
    for t in length(traj.rewards):-1:1
        # @show traj.rewards, length(traj.rewards)
        G = traj.rewards[t] + sampler.gamma * G
        returns[t] = G
    end
    # prepare the actions
    inputs_set = []
    for i in vcat(inputs_actions...)
        if i in inputs_set
            continue
        else
            push!(inputs_set, i)
        end
    end

    tmp = [MyModule.expr(MyModule.nc, x) for x in inputs_set]
    ds = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(tmp, sym_enc))
    inputs_ids = []
    for i in inputs_actions
        tmp = []
        for j in i
            push!(tmp, findfirst(==(j), inputs_set))
        end
        push!(inputs_ids, tmp)
    end
    tmp = vcat(traj.states[1], traj.next_states)
    depth = argmin(exp_size.(tmp))
    node_ex = tmp[depth]
    # return (;rewards=returns, inputs=inputs, smallest_node=MyModule.Node(node_ex, (), UInt64[], hash(node_ex), depth, hash(node_ex)))
    return (;rewards=returns, inputs=ds, smallest_node=MyModule.Node(node_ex, (), UInt64[], hash(node_ex), depth, hash(node_ex)), softmax_ids=inputs_ids, selected_ids=ones(Int32, length(inputs_ids)))
end


function sample_trajectory(sampler::AbstractSampler, env::AbstractEnvironment, model::AbstractModel)::Trajectory
    reset!(env)
    soltree = Dict()
    traj = Trajectory()
    for t in 1:sampler.max_steps
        possible_actions = action_space(env)
        if rand() >= sampler.epsilon
            o = [vec(model(pa)) for pa in possible_actions]
            a = possible_actions[argmax(o)]
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
    reset!(env)
    # if traj.rewards[end] == 0
    #     traj.rewards[end][traj.rewards[end]] = -100
    # end
    traj.rewards[traj.rewards .== 0] .= -100
    return traj
end


function leaf2trajectorie(leaf::Node, soltree::Dict, root, mcache, traj = [])
    if leaf.node_id == leaf.parent
        return
    end
    s = soltree[leaf.parent].ex
	r = mcache[leaf.ex]
    ns = leaf.ex
    is_done = isempty(leaf.children) ? true : false
    push!(traj, (s, ns, r, ns, is_done))
    leaf2trajectorie(soltree[leaf.parent], soltree, root, mcache, traj)
    return traj
end


isleaf(n::Node) = isempty(n.children)
function target_from_cache!(cache, leaf::Node, soltree::Dict; gamma=1.0)
    haskey(cache, leaf.ex) && return(cache[leaf.ex])
    r = exp_size(leaf.ex)
    root_id = findfirst(x->x.depth == 0, soltree)
    # root_id = leaf.parent
    if isleaf(leaf)
        v = (exp_size(soltree[root_id].ex) - r)
    else
        v = maximum(exp_size(soltree[root_id].ex) - r + gamma * target_from_cache!(cache, soltree[ch], soltree) for ch in leaf.children)
        v = max(0, v)
    end
    cache[leaf.ex] = v
    return(v)
end


function target_from_cache2!(cache, leaf::Node, soltree::Dict, root_node::Node; gamma=1.0)
    # haskey(cache, leaf.ex) && return(cache[leaf.ex])
    r = exp_size(leaf.ex)
    # root_id = findfirst(x->x.depth == 0, soltree)
    # root_id = leaf.parent
    if isleaf(leaf)
        v = (exp_size(root_node.ex) - r)
    else
        v = maximum(exp_size(root_node.ex) - r + gamma * target_from_cache2!(cache, soltree[ch], soltree, root_node) for ch in leaf.children)
        v = max(0, v)
    end
    # v = isleaf(leaf) ? (exp_size(soltree[leaf.parent].ex) - r) : maximum(exp_size(soltree[leaf.parent].ex) - r + gamma * target_from_cache2!(cache, soltree[ch], soltree, root_node) for ch in leaf.children)
    cache[leaf.ex] = v
    return(v)
end


function target_from_cache_value_network!(cache, leaf::Node, soltree::Dict, model::ExprModel; gamma=1.0)
    r = exp_size(leaf.ex)
    root_id = findfirst(x->x.depth == 0, soltree)
    if leaf.depth == 0
        v = maximum(exp_size(soltree[root_id].ex) - r + gamma * only(model(soltree[x].ex)) for x in leaf.children)
        v = max(0, v)
        cache[leaf.ex] = v
        return(v)
    end
    if isleaf(leaf)
        v = (exp_size(soltree[root_id].ex) - r)
    else
        # check if has upper
        v = maximum(exp_size(soltree[root_id].ex) - r + gamma * only(model(soltree[x].ex)) for x in leaf.children)
        v = max(0, v)
    end
    cache[leaf.ex] = v
    target_from_cache_value_network!(cache, soltree[leaf.parent], soltree, model)
end

function target_from_cache_value_network_fixed!(cache, root::Node, soltree::Dict, model::ExprModel; gamma=1.0)
    current_nodes = [root]
    current_id = 1
    while true
        if current_id > length(current_nodes)
            break
        end
        current_root = current_nodes[current_id]
        r = exp_size(current_root.ex)
        if length(current_root.children) == 0 && haskey(cache, current_root.ex)
            current_id += 1
            continue
        elseif length(current_root.children) == 0
            current_id += 1
            cache[current_root.ex] = exp_size(root.ex) - r
            continue
        end
        v = maximum(exp_size(root.ex) - r + gamma * only(model(soltree[x].ex)) for x in current_root.children)
        cache[current_root.ex] = max(0, v)
        for i in current_root.children
            push!(current_nodes, soltree[i])
        end
        current_id += 1
    end
end


function is_upper(leaf_node, soltree1, initial_leaf)
    if leaf_node.ex == initial_leaf.ex && leaf_node.depth != initial_leaf.depth
        return(true)
    end
    if leaf_node.depth == 0
        return(false)
    end
    return(is_upper(soltree1[leaf_node.parent], soltree1, initial_leaf))
end


function update_cache!(cache, leaf, soltree; gamma=1.0)
    if leaf.depth == 0 || leaf.node_id == leaf.parent
        return
    end
    parent = soltree[leaf.parent]
    children = parent.children
    r = exp_size(leaf.ex)
    v = maximum(exp_size(parent.ex) - r + gamma * cache[soltree[x].ex] for x in children)
    cache[parent.ex] = max(0, v)
    update_cache!(cache, parent, soltree, gamma=gamma)
end
function update_cache!(cache, leaf, soltree, target_model; gamma=1.0)
    if leaf.depth == 0 || leaf.node_id == leaf.parent
        return
    end
    parent = soltree[leaf.parent]
    children = parent.children
    r = exp_size(leaf.ex)
    v = maximum(exp_size(parent.ex) - r + gamma * only(target_model(soltree[x].ex)) for x in children)
    cache[parent.ex] = max(0, v)
    update_cache!(cache, parent, soltree, target_model, gamma=gamma)
end


function get_trajectory_from_mcache(initial_expr::NodeID, mcache::Dict)
    trj = Trajectory()
    push!(trj.states, initial_expr)
    push!(trj.actions, initial_expr)
    push!(trj.rewards, mcache[initial_expr])
    push!(trj.next_states, initial_expr)
    push!(trj.is_dones, false)
    for (ind,(i, j)) in enumerate(mcache)
        if i == initial_expr
            continue
        end
        push!(trj.states, i)
        push!(trj.actions, i)
        push!(trj.rewards, j)
        push!(trj.next_states, i)
        push!(trj.is_dones, ind == length(mcache))
    end
    return trj
end


function get_trajectory_from_mcache_planning(soltree::Dict, root::Node, smallest_node::Node, mcache::Dict, sampler::AbstractSampler)
    nodes_in_proof, proof = MyModule.extract_proof(smallest_node, soltree)
    nodes_in_proof = vcat(root, nodes_in_proof)
    # if sampler.batch > length(nodes_in_proof) 
    #     d2p = [ for n in nodes_in_proof]
    # else
    d2p = MyModule.distance_from_proof(soltree, nodes_in_proof)
    for n in 1:10
        tmp = filter(v -> v[2] ≤ n, d2p) # ids of nodes of interest
        if length(tmp) >= sampler.batch
            d2p = tmp
            break
        end
    end
    trj = Trajectory(sampler)
    push!(trj.states, root.ex)
    push!(trj.actions, root.ex)
    push!(trj.rewards, mcache[root.ex])
    push!(trj.next_states, root.ex)
    push!(trj.is_dones, false)
    for (ind,(i, _)) in enumerate(d2p)
        if i == root.ex
            continue
        end
        push!(trj.states, soltree[i].ex)
        push!(trj.actions, soltree[i].ex)
        push!(trj.rewards, mcache[soltree[i].ex])
        push!(trj.next_states, soltree[i].ex)
        push!(trj.is_dones, ind == length(mcache))
    end
    return trj
end

 function get_solutions_traj_actions!(node::Node, soltree, targets)
    if node.depth == 0
        return
    end
    current_node_id = node.parent
    get_full_tree = UInt64[]
    possible_actions = UInt64[]
    for d in soltree[node.parent].depth:-1:1
        tmp = current_node_id
        current_node_id = soltree[current_node_id].parent
        tmp = filter(x->tmp != x, soltree[current_node_id].children)
        append!(possible_actions, tmp)
        append!(get_full_tree, soltree[current_node_id].children)
    end
    push!(get_full_tree, current_node_id)
    # @show length(possible_actions)
    # @show length(get_full_tree)
    if isa(targets, Dict)
        targets[get_full_tree] = Dict(i=>0 for i in possible_actions)
    else
        push!(targets, (node.parent, possible_actions, get_full_tree))
    end
    get_solutions_traj_actions!(soltree[node.parent], soltree, targets)
end


sample_trajectory(sampler::PPOTreeSampler, env::AbstractEnvironment, policy::AbstractModel, value_model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(state(env)), policy, value_model)
sample_trajectory(sampler::PPOTreeSampler, env::Expr, policy::AbstractModel, value_model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(env), policy, value_model)
function sample_trajectory(sampler::PPOTreeSampler, env, policy::AbstractModel, value_model::AbstractModel)
    soltree, smallest_node, root, tree_history = MyModule.initialize_policy_tree_search(env, policy, max_expansions=sampler.max_steps, max_depth=sampler.max_depth, gamma=sampler.gamma)
    if sampler.solution
        targets = []
        get_solutions_traj_actions!(smallest_node, soltree, targets)
        Gₜ = 0
        inputs_vec = []
        value_inputs_vec = []
        rews = Float32[]
	    v_returns = Float32[]
        for (ind, (pos_action, possible_actions, tree)) in enumerate(targets)
            if ind == 1
                rew = minimum(x->exp_size(soltree[x].ex), tree) - exp_size(smallest_node.ex)
            else
                rew = minimum(x->exp_size(soltree[x].ex), tree) - minimum(x->exp_size(soltree[x].ex), targets[ind - 1][3])
            end
            # @show Gₜ
            # Aₜ = Gₜ - only(value_model(soltree[pos_action].ex))
            current_values = map(x->only(value_model(soltree[x].ex)), tree)
            if ind == 1
                Aₜ = rew - maximum(current_values)
	    	    G = rew
            else
                next_a =targets[ind - 1][1]
                next_values = map(x->only(value_model(soltree[x].ex)), targets[ind-1][3])
                Aₜ = rew + sampler.gamma * maximum(next_values) - maximum(current_values)
		        G = rew + sampler.gamma * maximum(next_values)
            end
            # @show Aₜ
            input_values = [expr(MyModule.nc, soltree[i].ex) for i in vcat(pos_action, possible_actions)]
            input_values = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_values, sym_enc))
            push!(inputs_vec, input_values)
            push!(value_inputs_vec, tree[argmax(current_values)])
            push!(rews, Aₜ)
            push!(v_returns, G)
        end
        # input_values = [expr(MyModule.nc, soltree[i].ex) for (i, _, _) in targets]
        input_values = [expr(MyModule.nc, soltree[i].ex) for i in value_inputs_vec]
        v_inputs = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_values, sym_enc))
        return (;rewards=(rews, v_returns), inputs=(inputs_vec, v_inputs), smallest_node=smallest_node)
    else
        Gₜ = 0
        inputs_vec = []
        value_inputs_vec = []
        softmax_ids = []
        selected_ids = []
        rews = Float32[]
	    v_returns = Float32[]
        reversed_tree_history = reverse(tree_history)
        for (ind, (tree_node_ids, possible_actions, selected_action_index)) in enumerate(reversed_tree_history)
            if ind == 1 
                rew = minimum(x->exp_size(soltree[x].ex), tree_node_ids) - minimum(x->exp_size(x.ex), collect(values(soltree)))
            else
                rew = minimum(x->exp_size(soltree[x].ex), tree_node_ids) - minimum(x->exp_size(soltree[x].ex), reversed_tree_history[ind - 1][1])
            end
            current_values = map(x->only(value_model(soltree[x].ex)), tree_node_ids)
            if ind == 1
                # Aₜ = rew - only(value_model(soltree[possible_actions[selected_action_index]].ex))
                Aₜ = rew - maximum(current_values)
	    	    G = rew
            else
                # _, next_possible_actions, next_selected_action_index = reversed_tree_history[ind - 1]
                # next_a = next_possible_actions[next_selected_action_index]
                next_values = map(x->only(value_model(soltree[x].ex)), reversed_tree_history[ind - 1][1])
                # Aₜ = rew + sampler.gamma * only(value_model(soltree[next_a].ex)) - only(value_model(soltree[possible_actions[selected_action_index]].ex))
                Aₜ = rew + sampler.gamma * maximum(next_values) - maximum(current_values)
		        G = rew + sampler.gamma * maximum(next_values)
            end
            push!(value_inputs_vec, tree_node_ids[argmax(current_values)])
            # @show Gₜ
            if ind == 1
                append!(inputs_vec, possible_actions)
                push!(softmax_ids, collect(1:length(possible_actions)))
                push!(selected_ids, selected_action_index)
            else
                push!(softmax_ids, [])
                for (ind,a) in enumerate(possible_actions)
                    if ind == selected_action_index && a in inputs_vec
                        tmp_ind = findfirst(==(a), inputs_vec)
                        push!(softmax_ids[end], tmp_ind)
                        push!(selected_ids, selected_action_index)
                    elseif ind == selected_action_index && !(a in inputs_vec)
                        push!(inputs_vec, a)
                        push!(softmax_ids[end], length(inputs_vec))
                        push!(selected_ids, selected_action_index)
                    elseif ind != selected_action_index && !(a in inputs_vec)
                        push!(inputs_vec, a)
                        push!(softmax_ids[end], length(inputs_vec))
                    elseif ind != selected_action_index && a in inputs_vec
                        tmp_ind = findfirst(==(a), inputs_vec)
                        push!(softmax_ids[end], tmp_ind)
                    # else
                    #     push!(inputs_vec, a)
                    #     push!(softmax_ids[end], )
                    #     push!(selected_ids, selected_action_index)
                    end
                end
            end
            push!(rews, Aₜ)
            push!(v_returns, G)
        end
        @timeit TO "get nodes for inputs" begin
            input_values = [expr(MyModule.nc, soltree[i].ex) for i in inputs_vec]
            input_values = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_values, sym_enc))
        end
        @assert length(rews) == length(softmax_ids)
        # v_inputs = [expr(MyModule.nc, soltree[pa[sid]].ex) for (_,pa,sid) in reversed_tree_history]
        v_inputs = [expr(MyModule.nc, soltree[i].ex) for i in value_inputs_vec]
        v_inputs = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(v_inputs, sym_enc))
        return (;rewards=(rews, v_returns), inputs=(input_values, v_inputs), smallest_node=smallest_node, softmax_ids=softmax_ids, selected_ids=selected_ids)
    end
end


sample_trajectory(sampler::PolicyTreeSampler, env::AbstractEnvironment, model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(state(env)), model)
sample_trajectory(sampler::PolicyTreeSampler, env::Expr, model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(env), model)
function sample_trajectory(sampler::PolicyTreeSampler, env::NodeID, model::AbstractModel)
    @timeit TO "sample tree search" soltree, smallest_node, root, tree_history = MyModule.initialize_policy_tree_search(env, model, max_expansions=sampler.max_steps, max_depth=sampler.max_depth, gamma=sampler.gamma)
    if sampler.solution
        targets = []
        get_solutions_traj_actions!(smallest_node, soltree, targets)
        Gₜ = 0
        inputs_vec = []
        rews = []
        for (ind, (pos_action, possible_actions, tree)) in enumerate(targets)
            if ind == 1
                rew = minimum(x->exp_size(soltree[x].ex), tree) - exp_size(smallest_node.ex)
            else
                rew = minimum(x->exp_size(soltree[x].ex), tree) - minimum(x->exp_size(soltree[x].ex), targets[ind - 1][3])
            end
            Gₜ = rew + sampler.gamma * Gₜ
            # @show Gₜ
            input_values = [expr(MyModule.nc, soltree[i].ex) for i in vcat(pos_action, possible_actions)]
            input_values = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_values, sym_enc))
            push!(inputs_vec, input_values)
            push!(rews, Gₜ)
        end
        return (;rewards=rews, inputs=inputs_vec, smallest_node=smallest_node)
    else
        Gₜ = 0
        inputs_vec = []
        softmax_ids = []
        selected_ids = []
        rews = []
        reversed_tree_history = reverse(tree_history)
        for (ind, (tree_node_ids, possible_actions, selected_action_index)) in enumerate(reversed_tree_history)
            if ind == 1
                rew = minimum(x->exp_size(soltree[x].ex), tree_node_ids) - minimum(x->exp_size(x.ex), collect(values(soltree)))
            else
                rew = minimum(x->exp_size(soltree[x].ex), tree_node_ids) - minimum(x->exp_size(soltree[x].ex), reversed_tree_history[ind - 1][1])
            end
            Gₜ = rew + sampler.gamma * Gₜ
            # @show Gₜ
            if ind == 1
                append!(inputs_vec, possible_actions)
                push!(softmax_ids, collect(1:length(possible_actions)))
                push!(selected_ids, selected_action_index)
            else
                push!(softmax_ids, [])
                for (ind,a) in enumerate(possible_actions)
                    if ind == selected_action_index && a in inputs_vec
                        tmp_ind = findfirst(==(a), inputs_vec)
                        push!(softmax_ids[end], tmp_ind)
                        push!(selected_ids, selected_action_index)
                    elseif ind == selected_action_index && !(a in inputs_vec)
                        push!(inputs_vec, a)
                        push!(softmax_ids[end], length(inputs_vec))
                        push!(selected_ids, selected_action_index)
                    elseif ind != selected_action_index && !(a in inputs_vec)
                        push!(inputs_vec, a)
                        push!(softmax_ids[end], length(inputs_vec))
                    elseif ind != selected_action_index && a in inputs_vec
                        tmp_ind = findfirst(==(a), inputs_vec)
                        push!(softmax_ids[end], tmp_ind)
                    # else
                    #     push!(inputs_vec, a)
                    #     push!(softmax_ids[end], )
                    #     push!(selected_ids, selected_action_index)
                    end
                end
            end
            push!(rews, Gₜ)
        end
        @timeit TO "get nodes for inputs" begin
            input_values = [expr(MyModule.nc, soltree[i].ex) for i in inputs_vec]
            input_values = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_values, sym_enc))
        end
        @assert length(rews) == length(softmax_ids)
        return (;rewards=rews, inputs=input_values, smallest_node=smallest_node, softmax_ids=softmax_ids, selected_ids=selected_ids)
    end
end


function get_Q_targets!(node::Node, soltree::Dict, model::ExprModel, targets::Dict; γ=0.99)
    next_children = soltree[node.node_id].children
    if isempty(next_children)
        targets[node.node_id] = 0f0
        return
    end
    if node.depth == 0
        get_full_tree = [node.node_id]
    else
        current_node_id = node.node_id
        get_full_tree = []
        for d in node.depth:-1:1
            current_node_id = soltree[current_node_id].parent
            append!(get_full_tree, soltree[current_node_id].children)
        end
        push!(get_full_tree, current_node_id)
    end
    # @show get_full_tree
    rew = minimum(x->exp_size(soltree[x].ex), get_full_tree) - minimum(x->exp_size(soltree[x].ex), next_children)
    new_target = rew + γ * maximum(x->only(model(soltree[x].ex)), next_children)
    targets[node.node_id] = new_target
    for child in next_children
        get_Q_targets!(soltree[child], soltree, model, targets)
    end
    return
end


sample_trajectory(sampler::QTreeSampler, env::AbstractEnvironment, model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(state(env)), model)
sample_trajectory(sampler::QTreeSampler, env::Expr, model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(env), model)
function sample_trajectory(sampler::QTreeSampler, env, model::AbstractModel)
    # soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search(env, model, max_expansions=sampler.max_steps, max_depth=sampler.max_depth)
    soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(env, model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon, gamma=sampler.gamma)
    # nodes_in_proof, proof = MyModule.extract_proof(smallest_node, soltree)
    # nodes_in_proof = vcat(root, nodes_in_proof)
    
    targets = Dict()
    get_Q_targets!(root, soltree, model, targets, γ=sampler.gamma)
    mcache = Dict(soltree[k].ex=>v for (k,v) in targets)
    if sampler.batch < 1
        return get_trajectory_from_mcache(root.ex, mcache, sampler)
    else
        return get_trajectory_from_mcache_planning(soltree, root, smallest_node, mcache, sampler)
    end 
end


sample_trajectory(sampler::QTreeSampler, env::AbstractEnvironment, model::AbstractModel, target_model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(state(env)), model, target_model)
sample_trajectory(sampler::QTreeSampler, env::Expr, model::AbstractModel, target_model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(env), model, target_model)
function sample_trajectory(sampler::QTreeSampler, env, model::AbstractModel, target_model::AbstractModel)
    # soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search(env, model, max_expansions=sampler.max_steps, max_depth=sampler.max_depth)
    soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(env, model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon, gamma=sampler.gamma)
    # nodes_in_proof, proof = MyModule.extract_proof(smallest_node, soltree)
    # nodes_in_proof = vcat(root, nodes_in_proof)
    
    targets = Dict()
    get_Q_targets!(root, soltree, target_model, targets, γ=sampler.gamma)
    mcache = Dict(soltree[k].ex=>v for (k,v) in targets)
    if sampler.batch < 1
        return get_trajectory_from_mcache(root.ex, mcache, sampler)
    else
        return get_trajectory_from_mcache_planning(soltree, root, smallest_node, mcache, sampler)
    end 
end


sample_trajectory(sampler::Tree2TreeSampler, env::AbstractEnvironment, model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(state(env)), model)
sample_trajectory(sampler::Tree2TreeSampler, env::Expr, model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(env), model)
function sample_trajectory(sampler::Tree2TreeSampler, env, model::AbstractModel)
    soltree, smallest_node, root, all_trees_history = MyModule.initialize_tree_search_tree(env, model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon, gamma=sampler.gamma)
    # sannity check 
    tmp = collect(keys(soltree))
    for i in all_trees_history
        for j in i
            @assert j in tmp
        end
    end
    # my idea to make all the other nodes to be E(root) - E(s) and the max(V(s)) to be max(r(t, t') + sampler.gamma * max(V(s')))
    # r(t,t1) = min(t) - min(t1) # => minₛ(E(s)) - min₁(E(1))
    # max(V(s)) = max(r(t, t') + sampler.gamma * max(V(s')))
    # nodes_in_proof, proof = MyModule.extract_proof(smallest_node, soltree)
    # nodes_in_proof = vcat(root, nodes_in_proof)
    # last_history_tree = findfirst(x->smallest_node.node_id in x, all_trees_history)
    r(t::Vector{UInt64}, tₙ::Vector{UInt64}) = minimum(map(x->exp_size(soltree[x].ex), t)) - minimum(map(x->exp_size(soltree[x].ex), tₙ))
    res = map(enumerate(all_trees_history[1: end - 1])) do (ind, t)
        # maybe compute values of the difference in the 
        target_v = maximum(r(t, all_trees_history[ind + 1]) + sampler.gamma * maximum(map(sₙ -> only(model(soltree[sₙ].ex)), all_trees_history[ind + 1])))
        best_s_id = argmax([only(model(soltree[i].ex)) for i in t])
        # input_values = [expr(MyModule.nc, soltree[t[best_s_id]].ex)]
        # input_values = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_values, sym_enc))
        return (;rewards=target_v, input_values=soltree[t[best_s_id]].ex)
    end
    tmp1 = [i.input_values for i in res]
    tmp2 = [i.rewards for i in res]
    return (;input_values=tmp1, rewards=tmp2, pointer=exp_size(smallest_node.ex))
end

 
sample_trajectory(sampler::DGSampler, env::AbstractEnvironment, model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(state(env)), model)
sample_trajectory(sampler::DGSampler, env::Expr, model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(env), model)
function sample_trajectory(sampler::DGSampler, env, model::AbstractModel)::Trajectory
    soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(env, model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon, gamma=sampler.gamma)
    mcache = Dict()
    # Get cache from the Tree
    target_from_cache!(mcache, root, soltree)
    # Recompute cache for edges that are repetead 
    all_leafs = filter(i->length(i.children) == 0, collect(values(soltree1)))
    all_inner_nodes = filter(i->length(i.children) != 0, collect(values(soltree1)))
    all_inner_nodes_expr = Set(map(x->x.ex, all_inner_nodes))
    number_of_inner_nodes = length(soltree1) - length(all_leafs)
    sorted_leafs = sort(all_leafs, by=x->(exp_size(x.ex), x.depth))
    # filter inner
    filtered_sorted_leafs = filter(x->!(x.ex in all_inner_nodes_expr), sorted_leafs)
    # filter upper leafs
    unfiltered_sorted_leafs = filter(x->x.ex in all_inner_nodes_expr, sorted_leafs)
    for leaf_node in unfiltered_sorted_leafs
        update_cache!(mcache, leaf_node, soltree1, gamma=sampler.gamma)
    end
    if sampler.batch < 1
        return get_trajectory_from_mcache(root.ex, mcache, sampler)
    else
        return get_trajectory_from_mcache_planning(soltree, root, smallest_node, mcache, sampler)
    end 
end


sample_trajectory(sampler::DAGSampler, env::AbstractEnvironment, model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(state(env)), model)
sample_trajectory(sampler::DAGSampler, env::Expr, model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(env), model)
function sample_trajectory(sampler::DAGSampler, env, model::AbstractModel)::Trajectory
    soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(env, model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon, gamma=sampler.gamma)
    mcache = Dict()
    target_from_cache!(mcache, root, soltree)
    all_leafs = filter(i->length(i.children) == 0, collect(values(soltree1)))
    all_inner_nodes = filter(i->length(i.children) != 0, collect(values(soltree1)))
    all_inner_nodes_expr = Set(map(x->x.ex, all_inner_nodes))
    unfiltered_sorted_leafs = filter(x->x.ex in all_inner_nodes_expr, all_leafs)
    # filter upper from tree
    upper_nodes = filter(x->is_upper(x, soltree1, x), unfiltered_sorted_leafs)
    not_upper_nodes = filter(x->!is_upper(x, soltree1, x), unfiltered_sorted_leafs)
    for leaf_node in upper_nodes
        delete!(soltree1, leaf_node.node_id)
        soltree1[leaf_node.parent].children = filter(x->x != leaf_node.node_id, soltree1[leaf_node.parent].children)
    end
    for leaf_node in not_upper_nodes
        update_cache!(mcache, leaf_node, soltree1, gamma=sampler.gamma)
    end
    if sampler.batch < 1
        return get_trajectory_from_mcache(root.ex, mcache, sampler)
    else
        return get_trajectory_from_mcache_planning(soltree, root, smallest_node, mcache, sampler)
    end 
end


sample_trajectory(sampler::DAGSampler, env::AbstractEnvironment, model::AbstractModel, target_model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(state(env)), model, target_model)
sample_trajectory(sampler::DAGSampler, env::Expr, model::AbstractModel, target_model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(env), model, target_model)
function sample_trajectory(sampler::DAGSampler, env, model::AbstractModel, target_model::AbstractModel)::Trajectory
    soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(env, model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon, gamma=sampler.gamma)
    mcache = Dict()
    all_leafs = filter(i->length(i.children) == 0, collect(values(soltree1)))
    all_inner_nodes = filter(i->length(i.children) != 0, collect(values(soltree1)))
    all_inner_nodes_expr = Set(map(x->x.ex, all_inner_nodes))
    unfiltered_sorted_leafs = filter(x->x.ex in all_inner_nodes_expr, all_leafs)
    # filter upper from tree
    upper_nodes = filter(x->is_upper(x, soltree1, x), unfiltered_sorted_leafs)
    # not_upper_nodes = filter(x->!is_upper(x, soltree1, x), unfiltered_sorted_leafs)
    for leaf_node in upper_nodes
        delete!(soltree1, leaf_node.node_id)
        soltree1[leaf_node.parent].children = filter(x->x != leaf_node.node_id, soltree1[leaf_node.parent].children)
    end
    # for leaf_node in not_upper_nodes
    #     # update_cache!(mcache, leaf_node, soltree1)
    #     target_from_cache_value_network!(mcache, leaf_node, soltree1, target_model, gamma=sampler.gamma)
    # end
    target_from_cache_value_network_fixed!(mcache1, root, soltree1, target_model, gamma=sampler.gamma)
    if sampler.batch < 1
        return get_trajectory_from_mcache(root.ex, mcache, sampler)
    else
        return get_trajectory_from_mcache_planning(soltree, root, smallest_node, mcache, sampler)
    end 
end


sample_trajectory(sampler::TreeSampler, env::AbstractEnvironment, model::AbstractModel, target_model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(state(env)), model, target_model)
sample_trajectory(sampler::TreeSampler, env::Expr, model::AbstractModel, target_model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(env), model, target_model)
function sample_trajectory(sampler::TreeSampler, env, model::AbstractModel, target_model::AbstractModel)::Trajectory
    # search_time = @elapsed soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search(d, policy; max_expansions=max_steps, max_depth=max_depth)
    soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(env, model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon, gamma=sampler.gamma)
    if sampler.is_directed
        stree = soltree1
        root_id = findfirst(x->x.depth == 0, stree)
    else
        stree = soltree
        root_id = root.node_id
    end
    mcache = Dict()
    all_leafs = filter(i->length(i.second.children) == 0, stree)
    sorted_leafs = sort(all_leafs, by=x->stree[x].depth, order=Base.Order.Reverse)
    # for (nid, lf) in sorted_leafs
    #     target_from_cache_value_network!(mcache, lf, stree, target_model)
    # end
    target_from_cache_value_network_fixed!(mcache, stree[root_id], stree, target_model, gamma=sampler.gamma)
    if length(mcache) != length(soltree)
        for (i,j) in soltree
            if !haskey(mcache, j.ex)
                @show i, j.ex
            end
        end
        @assert length(mcache) == length(soltree)
    end
    if sampler.batch < 1
        return get_trajectory_from_mcache(root.ex, mcache, sampler)
    else
        return get_trajectory_from_mcache_planning(soltree, root, smallest_node, mcache, sampler)
    end
end

sample_trajectory(sampler::TreeSampler, env::AbstractEnvironment, model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(state(env)), model)
sample_trajectory(sampler::TreeSampler, env::Expr, model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(env), model)
function sample_trajectory(sampler::TreeSampler, env, model::AbstractModel)::Trajectory
    # search_time = @elapsed soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search(d, policy; max_expansions=max_steps, max_depth=max_depth)
    @timeit TO "sample tree search" soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(env, model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon, gamma=sampler.gamma)
    @show length(soltree)
    if sampler.is_directed
        stree = soltree1
    else
        stree = soltree
    end
    @timeit TO "get target values and Traj" begin        
        mcache = Dict()
        root_id = findfirst(x->x.depth == 0, stree)
        if sampler.is_directed
            target_from_cache2!(mcache, stree[root_id], stree, stree[root_id])
        else
            target_from_cache!(mcache, stree[root_id], stree)
        end
        # @show length(mcache), length(soltree)
        if length(mcache) != length(soltree)
            for (i,j) in soltree
                if !haskey(mcache, j.ex)
                    @show i, j.ex
                end
            end
            @assert length(mcache) == length(soltree)
        end
        if sampler.batch < 1
            return get_trajectory_from_mcache(root.ex, mcache, sampler)
        else
            return get_trajectory_from_mcache_planning(soltree, root, smallest_node, mcache, sampler)
        end
    end
end

sample_trajectory(sampler::TreeSampler2Values, env::AbstractEnvironment, model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(state(env)), model)
sample_trajectory(sampler::TreeSampler2Values, env::Expr, model::AbstractModel) = sample_trajectory(sampler, MyModule.intern!(env), model)
function sample_trajectory(sampler::TreeSampler2Values, env, model::AbstractModel)::Trajectory
    soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(env, model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon)
    if sampler.is_directed
        stree = soltree1
    else
        stree = soltree
    end
    all_leafs = filter(i->length(i.children) == 0, collect(values(stree)))
    all_inner_nodes = filter(i->length(i.children) != 0, collect(values(stree)))
    all_inner_nodes_expr = Set(map(x->x.ex, all_inner_nodes))
    # @assert length(all_inner_nodes_expr) + length(all_leafs) == length(soltree)
    # @show typeof(all_leafs)
    number_of_inner_nodes = length(stree) - length(all_leafs)
    sorted_leafs = sort(all_leafs, by=x->(exp_size(root.ex) - exp_size(x.ex), x.depth))
    # filter inner
    filtered_sorted_leafs = filter(x->!(x.ex in all_inner_nodes_expr), sorted_leafs)
    unfiltered_sorted_leafs = filter(x->x.ex in all_inner_nodes_expr, sorted_leafs)
    if !(sampler.is_directed)
        @assert length(unfiltered_sorted_leafs) == 0
    end
    final_rewards = []
    final_next_states = NodeID[]
    inner_to_leaf = Dict{NodeID, Vector{Float32}}()
    for n in filtered_sorted_leafs
        tmp = [exp_size(root.ex) - exp_size(n.ex), 0]
        inner_to_leaf[n.ex] = tmp        
        for _ in 1:n.depth
            pr = stree[n.parent]
            # @show n
            tmp[2] += 1
            if !haskey(inner_to_leaf, pr.ex)
                inner_to_leaf[pr.ex] = tmp
            else
                break
            end
            n = pr
        end
    end
    for n in unfiltered_sorted_leafs
        tmp = inner_to_leaf[n.ex]
        for _ in 1:n.depth
            pr = stree[n.parent]
            tmp[2] += 1
            if !haskey(inner_to_leaf, pr.ex)
                inner_to_leaf[pr.ex] = tmp
            elseif inner_to_leaf[pr.ex][1] > tmp[1] || (inner_to_leaf[pr.ex][1] == tmp[1] && inner_to_leaf[pr.ex][2] > tmp[2])
                inner_to_leaf[pr.ex] = tmp
            else
                break
            end
            n = pr
        end
    end
    # @show inner_to_leaf 
    # @show length(inner_to_leaf), length(soltree)
    # @show length(inner_to_leaf), length(soltree)
    # set_diff = setdiff(Set(map(x->x.ex,values(soltree))), Set(keys(inner_to_leaf)))
    # if !isempty(set_diff)
    #     @show set_diff 
    #     @show [i in all_inner_nodes_expr for i in set_diff]
    #     MyModule.check_soltree_consistancy(stree)
    #     tmp = [i for i in values(stree) if i.ex in set_diff]
    #     @show tmp
    # end
    @assert length(inner_to_leaf) == length(soltree)
    # final_next_states = collect(keys(inner_to_leaf))
    # final_rewards = collect(values(inner_to_leaf))
    # trj = Trajectory(1, final_next_states, final_next_states, final_rewards, final_next_states, falses(length(final_next_states)))

    if sampler.batch < 1
        return get_trajectory_from_mcache(root.ex, inner_to_leaf, sampler)
    else
        traj = get_trajectory_from_mcache_planning(soltree, root, smallest_node, inner_to_leaf, sampler)
        @assert length(traj.rewards) * length(traj.rewards[1]) == length(traj.next_states) * 2
        return traj
    end
end

# get_target_values(trj::AbstractTrajectory) = error("get_target_values not implemeted")
# get_input_values(trj::AbstractTrajectory) = error("get_target_values not implemeted")
get_target_values(trj::Tuple{Int, Trajectory}) = get_target_values(trj[2])
get_input_values(trj::Tuple{Int, Trajectory}) = get_input_values(trj[2])
function get_target_values(trj::Tuple{Int, TreeTrajectory})
    return trj[2].rewards[trj[1]]
end
function get_input_values(trj::Tuple{Int, TreeTrajectory})
    input_values = [expr(MyModule.nc, i) for i in trj[2].next_states[trj[1]]]
    input_values = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_values, sym_enc))
    return input_values
end

get_target_values(trj::Trajectory) = trj.rewards
# function get_target_values(trj::Trajectory)
#     n = length(trj.rewards)
#     target_values = zeros(Float32, n)
#     G = 0.0
#     for t in reverse(1:n)
#         G = trj.rewards[t] + 0.9 * G
#         target_values[t] = G
#     end
#     return target_values
# end
function get_input_values(trj::PolicyTrajectory)
    input_values = [map(x->expr(MyModule.nc, x), i) for i in trj.actions]
    concatenated_input_values = vcat(input_values...)
    ids_masks = Int[]
    # matrix_mask = ones(length(concatenated_input_values), length(trj.actions)) * -Inf
    matrix_mask = zeros(Bool, length(concatenated_input_values), length(trj.actions))
    for (ind1, (a, ns)) in enumerate(zip(trj.actions, trj.next_states))
        tmp = 0
        for (ind, j) in enumerate(a)
            if j == ns
                tmp = ind
                break
            end
        end
        if ind1 == 1
            matrix_mask[1:length(a), ind1] .= 1
            ids = tmp
        else
            old = sum(length.(trj.actions[1:ind1 - 1]))
            ids = old + tmp + length(concatenated_input_values) * (ind1 - 1)
            matrix_mask[old + 1: old + length(a), ind1] .= 1
        end
        # ids = ind1 == 1 ? (length(a), tmp) : (length(a), length(trj.actions[ind1 - 1]) + tmp)
        push!(ids_masks, ids)
    end
    input_values = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(concatenated_input_values))
    return (input_values, (matrix_mask, ids_masks))
end
function get_target_values(trj::PolicyTrajectory)
    n = length(trj.rewards)
    target_values = zeros(Float32, n)
    G = 0.0
    for t in reverse(1:n)
        G = trj.rewards[t] + 0.9 * G
        target_values[t] = G
    end
    return target_values
end
# function get_target_values(trj::Trajectory)
#     rew = []
#     cum_rew = 0
#     root_ex = trj.states[1]
#     for (ind, n) in enumerate(reverse(trj.next_states))
#         if ind == 1
#             cum_rew = exp_size(root_ex) - exp_size(n)
#         else
#             cum_rew += exp_size(trj.states[end - ind + 1]) - exp_size(n)
#         end
#         push!(rew, cum_rew)
#     end
#     return rew
# end
function get_input_values(trj::Trajectory, sym_enc=sym_enc)
    input_values = [expr(MyModule.nc, i) for i in trj.next_states]
    input_values = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_values, sym_enc))
    return input_values
end
function get_input_values(trj::Vector{NodeID}, sym_enc=sym_enc)
    input_values = [expr(MyModule.nc, i) for i in trj]
    input_values = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_values, sym_enc))
    return input_values
end
get_input_values(trj::NamedTuple{(:rewards, :inputs, :smallest_node, :softmax_ids, :selected_ids)}) = trj.inputs
get_input_values(trj::NamedTuple{(:input_values, :rewards, :pointer)}) = get_input_values(trj.input_values)
get_input_values(trj::NamedTuple{(:rewards, :inputs)}) = trj.inputs
get_input_values(trj::NamedTuple{(:rewards, :inputs, :smallest_node)}) = trj.inputs
preprocess_trajectory(trj::Trajectory) = get_target_values(trj), get_input_values(trj)
# preprocess_trajectory(trj::TupleTrajectory) = get_target_values(trj), get_input_values(trj)
# preprocess_trajectory(trj::TreeTrajectory) = get_target_values(trj), get_input_values(trj)
preprocess_trajectory(trj::Tuple{Int, Trajectory}) = get_target_values(trj), get_input_values(trj)
preprocess_trajectory(trj::Tuple{Int, TreeTrajectory}) = get_target_values(trj), get_input_values(trj)
preprocess_trajectory(trj::NamedTuple) = trj.labels, get_input_values(trj.input_values)
preprocess_trajectory(trj::Tuple{Int, PolicyTrajectory}) = get_target_values(trj[2]), get_input_values(trj[2])
get_target(rew, sampler::AbstractSampler) = rew
get_target(rew, sampler::TreeSampler2Values) = vec(hcat(rew...))
# get_input_values(trj::Vector{NamedTuple}) = trj.inp
# function preprocess_trajectory(trj::AbstractTrajectory)
#     target_values = get_target_values(trj)
#     input_values = get_input_values(trj)
#     return target_values, input_values
# end
# function preprocess_trajectory(trj::Tuple{Int, AbstractTrajectory})
#     target_values = get_target_values(trj)
#     input_values = get_input_values(trj)
#     return target_values, input_values
# end




function validation1(pipeline)
    validation_sampler = RLSampler(max_steps=50, epsilon=0.0, eps_decay=0.95, batch=64)
    traj = sample_trajectory(validation_sampler, pipeline.env, pipeline.model)
    return validation(traj)
end
function validation2(pipeline, traj)
    return validation1(pipeline), validation(traj)
end
validation(traj::AbstractTrajectory) = 0
function validation(traj::Trajectory)
    smallest_node = minimum(exp_size.(traj.next_states))
    return max(0, exp_size(traj.states[1]) - smallest_node)
end
function validation(traj::TreeTrajectory)
    return max(0, exp_size(traj.states[end][end]) - exp_size(traj.next_states[end][1]))
end
