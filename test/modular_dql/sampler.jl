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

function update_epsilon!(sampler::AbstractSampler)
    new_epsilon = max(0.1, sampler.eps_decay * sampler.epsilon) == 0.1 ? 0.0 : max(0.1, sampler.eps_decay * sampler.epsilon)
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


mutable struct TreeSampler <: AbstractSampler
    max_steps::Int
    max_depth::Int
    epsilon::Float32
    eps_decay::Float32
    is_directed::Bool
    n_best::Int
    batch::Int
end
TreeSampler(;max_steps=50, max_depth=100, epsilon=0.0, eps_decay=0.95, is_directed=false, n_best=-1, batch=64) = TreeSampler(max_steps, max_depth, epsilon, eps_decay, is_directed, n_best, batch)


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
function target_from_cache!(cache, leaf::Node, soltree::Dict)
    haskey(cache, leaf.ex) && return(cache[leaf.ex])
    r = exp_size(leaf.ex)
    root_id = findfirst(x->x.depth == 0, soltree)
    # root_id = leaf.parent
    v = isleaf(leaf) ? (exp_size(soltree[root_id].ex) - r) : maximum(exp_size(soltree[leaf.parent].ex) - r + target_from_cache!(cache, soltree[ch], soltree) for ch in leaf.children)
    cache[leaf.ex] = v
    return(v)
end


function target_from_cache2!(cache, leaf::Node, soltree::Dict, root_node::Node; gamma=0.99)
    # haskey(cache, leaf.ex) && return(cache[leaf.ex])
    r = exp_size(leaf.ex)
    # root_id = findfirst(x->x.depth == 0, soltree)
    # root_id = leaf.parent
    v = isleaf(leaf) ? (exp_size(root_node.ex) - r) : maximum(exp_size(soltree[leaf.parent].ex) - r + target_from_cache2!(cache, soltree[ch], soltree) for ch in leaf.children)
    cache[leaf.ex] = v
    return(v)
end


function sample_trajectory(sampler::TreeSampler, env::AbstractEnvironment, model::AbstractModel)::TreeTrajectory
    # search_time = @elapsed soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search(d, policy; max_expansions=max_steps, max_depth=max_depth)
    soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(state(env), model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon)
    if sampler.is_directed
        stree = soltree1
    else
        stree = soltree
    end
    mcache = Dict()
    root_id = findfirst(x->x.depth == 0, stree)
    if sampler.is_directed
        target_from_cache2!(mcache, stree[root_id], stree, stree[root_id])
    else
        target_from_cache!(mcache, stree[root_id], stree)
    end
    # @show length(mcache), length(soltree)
    @assert length(mcache) == length(soltree)
    all_leafs = filter(i->length(i.second.children) == 0, stree)
    trajectories = [leaf2trajectorie(lf, stree, root, mcache) for (nid, lf) in all_leafs]
    sorted_trajectories = sort(trajectories, by=x->exp_size(root.ex) - exp_size(x[1][2]))

    if sampler.n_best > 0
        n_best_trajectories = last(sorted_trajectories, sampler.n_best)
    else
        n_best_trajectories = sorted_trajectories
    end
    trj = TreeTrajectory()
    # @show length(n_best_trajectories[1])
    trj.soltree = stree
    for sorted_trajectory in n_best_trajectories
        states = map(x->x[1], sorted_trajectory)
        actions = map(x->x[2], sorted_trajectory)
        rewards = map(x->x[3], sorted_trajectory)
        next_states = map(x->x[4], sorted_trajectory)
        is_dones = map(x->x[5], sorted_trajectory)
        push!(trj.states, states)
        push!(trj.actions, actions)
        push!(trj.rewards, rewards)
        push!(trj.next_states, next_states)
        push!(trj.is_dones, is_dones)
    end
    # @show length(trj.states[1])
    @assert length(trj.states[1]) == length(n_best_trajectories[1])
    # return sorted_trajectories
    # @show last(sorted_trajectories)[1][2]
    return trj
end
# function sample_trajectory(sampler::PlanningSampler, env::AbstractEnvironment, model::AbstractModel)::PlanningTrajectory
#     soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(state(env), model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon)

# end


get_target_values(trj::AbstractTrajectory) = error("get_target_values not implemeted")
get_input_values(trj::AbstractTrajectory) = error("get_target_values not implemeted")
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

# get_target_values(trj::Trajectory) = trj.rewards
function get_target_values(trj::Trajectory)
    rew = []
    cum_rew = 0
    root_ex = trj.states[1]
    for (ind, n) in enumerate(reverse(trj.next_states))
        if ind == 1
            cum_rew = exp_size(root_ex) - exp_size(n)
        else
            cum_rew += exp_size(trj.states[end - ind + 1]) - exp_size(n)
        end
        push!(rew, cum_rew)
    end
    return rew
end
function get_input_values(trj::Trajectory, sym_enc=sym_enc)
    input_values = [expr(MyModule.nc, i) for i in trj.next_states]
    input_values = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_values, sym_enc))
    return input_values
end

function preprocess_trajectory(trj::AbstractTrajectory)
    target_values = get_target_values(trj)
    input_values = get_input_values(trj)
    return target_values, input_values
end
function preprocess_trajectory(trj::Tuple{Int, AbstractTrajectory})
    target_values = get_target_values(trj)
    input_values = get_input_values(trj)
    return target_values, input_values
end




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
