abstract type AbstractTrajectory end

mutable struct Trajectory <: AbstractTrajectory
    pointer::Int
    states::Vector
    actions::Vector
    rewards::Vector
    next_states::Vector
    is_dones::Vector
end

Trajectory() = Trajectory(1, [], [], [], [], [])
function push2traj!(trj::Trajectory, data::Tuple)
    s, a, r, ns, is_done = data
    push!(trj.states, s)
    push!(trj.actions, a)
    push!(trj.rewards, r)
    push!(trj.next_states, ns)
    push!(trj.is_dones, is_done)
end
function update_epsilon!(sampler::AbstractSampler)
    new_epsilon = max(0.1, sampler.eps_decay * sampler.epsilon) == 0.1 ? 0.0 : max(0.1, sampler.eps_decay * sampler.epsilon)
    sampler.epsilon = new_epsilon
end
struct RLSampler <: AbstractSampler
    max_steps::Int
    epsilon::Float32
    eps_decay::Float32
end
RLSampler(;max_steps=50, epsilon=0.0, eps_decay=0.95) = RLSampler(max_steps, epsilon, eps_decay)
struct TreeSampler <: AbstractSampler
    max_steps::Int
    max_depth::Int
    epsilon::Float32
    eps_decay::Float32
    is_dag::Bool
end
TreeSampler(;max_steps=50, max_depth=100, epsilon=0.0, eps_decay=0.95, is_dag=false) = TreeSampler(max_steps, max_depth, epsilon, eps_decay, is_dag)

function sample_trajectory(sampler::RLSampler, env::AbstractEnvironment, model::AbstractModel)::AbstractTrajectory
    reset!(env)
    trj = Trajectory()
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
        push2traj!(trj, (s,a,r,ns,is_done))
        if is_done
            break
        end
    end
    return traj
end

function leaf2trajectorie(leaf::Node, soltree::Dict, root, mcache, traj::Trajectory)
    if leaf.node_id == leaf.parent
        return
    end
    s = soltree[leaf.parent]
	r = mcache[leaf.ex]
    ns = leaf.ex
    is_done = isempty(trajectories) ? true : false
    push2traj!(traj, (s, ns, r, ns, is_done))
    leaf2trajectorie(soltree[leaf.parent], soltree, root, mcache, traj::Trajectory)
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


function target_from_cache2!(cache, leaf::Node, soltree::Dict; gamma=0.99)
    # haskey(cache, leaf.ex) && return(cache[leaf.ex])
    r = exp_size(leaf.ex)
    root_id = findfirst(x->x.depth == 0, soltree)
    # root_id = leaf.parent
    v = isleaf(leaf) ? (exp_size(soltree[root_id].ex) - r) : maximum(exp_size(soltree[leaf.parent].ex) - r + target_from_cache2!(cache, soltree[ch], soltree) for ch in leaf.children)
    cache[leaf.ex] = v
    return(v)
end

# function sample_trajectory(sampler::AbstractSampler, env::AbstractEnvironment, model::AbstractModel)::AbstractTrajectory
function sample_trajectory(sampler::TreeSampler, env::AbstractEnvironment, model::AbstractModel)::AbstractTrajectory
    # search_time = @elapsed soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search(d, policy; max_expansions=max_steps, max_depth=max_depth)
    soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(state(env), model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon)
    if sampler.is_dag
        stree = soltree1
    else
        stree = soltree
    end
    mcache = Dict()
    root_id = findfirst(x->x.depth == 0, stree)
    if sampler.is_dag
        target_from_cache2!(mcache, stree[root_id], stree)
    else
        target_from_cache!(mcache, stree[root_id], stree)
    end
    @assert length(mcache) == length(stree)
    all_leafs = filter(i->length(i.second.children) == 0, stree)
    trajectories = [leaf2trajectorie(lf, stree, root, mcache) for (nid, lf) in all_leafs]
    sorted_trajectories = sort(trajectories, by=x->exp_size(root.ex) - exp_size(x[1][2]))
    # return sorted_trajectories
    # @show last(sorted_trajectories)[1][2]
    return sorted_trajectories
end


get_target_values(trj::AbstractTrajectory) = error("get_target_values not implemeted")
get_input_values(trj::AbstractTrajectory) = error("get_target_values not implemeted")

get_target_values(trj::Trajectory) = trj.rewards
function get_input_values(trj::Trajectory, sym_enc=sym_enc)
    input_values = [expr(MyModule.nc, i) for i in trj.next_states]
    input_values = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_values, sym_enc))
    return input_values
end

function preprocess_trajectory(trj::AbstractTrajectory, type)
    target_values = get_target_values(trj)
    input_values = get_input_values(trj)
    return target_values, input_values
end