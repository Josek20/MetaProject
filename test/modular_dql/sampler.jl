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

mutable struct DQNSampler <: AbstractSampler
    max_steps::Int
    epsilon::Float32
    eps_decay::Float32
    batch::Int
end
DQNSampler(;max_steps=50, epsilon=0.0, eps_decay=0.95, batch=64) = DQNSampler(max_steps, epsilon, eps_decay, batch)

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
    traj.rewards[traj.rewards .== 0] .= -100
    return traj
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
function target_from_cache!(cache, leaf::Node, soltree::Dict; gamma=0.9)
    haskey(cache, leaf.ex) && return(cache[leaf.ex])
    r = exp_size(leaf.ex)
    root_id = findfirst(x->x.depth == 0, soltree)
    # root_id = leaf.parent
    v = isleaf(leaf) ? (exp_size(soltree[root_id].ex) - r) : maximum(exp_size(soltree[leaf.parent].ex) - r + gamma * target_from_cache!(cache, soltree[ch], soltree) for ch in leaf.children)
    cache[leaf.ex] = v
    return(v)
end


function target_from_cache2!(cache, leaf::Node, soltree::Dict, root_node::Node; gamma=0.9)
    # haskey(cache, leaf.ex) && return(cache[leaf.ex])
    r = exp_size(leaf.ex)
    # root_id = findfirst(x->x.depth == 0, soltree)
    # root_id = leaf.parent
    v = isleaf(leaf) ? (exp_size(root_node.ex) - r) : maximum(exp_size(soltree[leaf.parent].ex) - r + gamma * target_from_cache2!(cache, soltree[ch], soltree, root_node) for ch in leaf.children)
    # v = isleaf(leaf) ? (exp_size(soltree[leaf.parent].ex) - r) : maximum(exp_size(soltree[leaf.parent].ex) - r + gamma * target_from_cache2!(cache, soltree[ch], soltree, root_node) for ch in leaf.children)
    cache[leaf.ex] = v
    return(v)
end

function target_from_cache_value_network!(cache, leaf::Node, soltree::Dict, model::ExprModel; gamma=0.9)
    r = exp_size(leaf.ex)
    if leaf.depth == 0
        v = maximum(exp_size(soltree[leaf.parent].ex) - r + gamma * only(model(soltree[x].ex)) for x in leaf.children)
        cache[leaf.ex] = v
        return(v)
    end
    if isleaf(leaf)
        v = (exp_size(soltree[leaf.parent].ex) - r)
    else
        # check if has upper
        v = maximum(exp_size(soltree[leaf.parent].ex) - r + gamma * only(model(soltree[x].ex)) for x in leaf.children)
    end
    cache[leaf.ex] = v
    target_from_cache_value_network!(cache, soltree[leaf.parent], soltree, model)
end

function sample_trajectory(sampler::TreeSampler, env::AbstractEnvironment, model::AbstractModel, target_model::AbstractModel)::Trajectory
    # search_time = @elapsed soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search(d, policy; max_expansions=max_steps, max_depth=max_depth)
    soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(state(env), model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon)
    if sampler.is_directed
        stree = soltree1
    else
        stree = soltree
    end
    mcache = Dict()
    all_leafs = filter(i->length(i.second.children) == 0, stree)
    sorted_leafs = sort(all_leafs, by=x->stree[x].depth, order=Base.Order.Reverse)
    for (nid, lf) in sorted_leafs
        target_from_cache_value_network!(mcache, lf, stree, target_model)
    end
    if length(mcache) != length(soltree)
        for (i,j) in soltree
            if !haskey(mcache, j.ex)
                @show i, j.ex
            end
        end
        @assert length(mcache) == length(soltree)
    end
    trj = Trajectory()
    push!(trj.states, state(env))
    push!(trj.actions, state(env))
    push!(trj.rewards, mcache[state(env)])
    push!(trj.next_states, state(env))
    push!(trj.is_dones, false)
    for (ind,(i, j)) in enumerate(mcache)
        if i == state(env)
            continue
        end
        push!(trj.states, i)
        push!(trj.actions, i)
        push!(trj.rewards, j)
        push!(trj.next_states, i)
        push!(trj.is_dones, ind == length(mcache))
    end
    trj_rand = Trajectory()
    for i in rand(1:length(trj.states), sampler.batch)
        push!(trj_rand.states, trj.states[i])
        push!(trj_rand.actions, trj.actions[i])
        push!(trj_rand.rewards, trj.rewards[i])
        push!(trj_rand.next_states, trj.next_states[i])
        push!(trj_rand.is_dones, trj.is_dones[i])
    end
    if sampler.batch > 0
        trj_rand.pointer = exp_size(smallest_node.ex)
        return trj_rand
    else
        trj.pointer = exp_size(smallest_node.ex)
        return trj
    end
end


function sample_trajectory(sampler::TreeSampler, env::AbstractEnvironment, model::AbstractModel)::Trajectory
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
    if length(mcache) != length(soltree)
        for (i,j) in soltree
            if !haskey(mcache, j.ex)
                @show i, j.ex
            end
        end
        @assert length(mcache) == length(soltree)
    end
    trj = Trajectory()
    push!(trj.states, state(env))
    push!(trj.actions, state(env))
    push!(trj.rewards, mcache[state(env)])
    push!(trj.next_states, state(env))
    push!(trj.is_dones, false)
    for (ind,(i, j)) in enumerate(mcache)
        push!(trj.states, i)
        push!(trj.actions, i)
        push!(trj.rewards, j)
        push!(trj.next_states, i)
        push!(trj.is_dones, ind == length(mcache))
    end
    trj_rand = Trajectory()
    for i in rand(1:length(trj.states), sampler.batch)
        push!(trj_rand.states, trj.states[i])
        push!(trj_rand.actions, trj.actions[i])
        push!(trj_rand.rewards, trj.rewards[i])
        push!(trj_rand.next_states, trj.next_states[i])
        push!(trj_rand.is_dones, trj.is_dones[i])
    end
    if sampler.batch > 0
        trj_rand.pointer = exp_size(smallest_node.ex)
        return trj_rand
    else
        trj.pointer = exp_size(smallest_node.ex)
        return trj
    end
end

function sample_trajectory(sampler::PlanningTreeSampler, env::AbstractEnvironment, model::AbstractModel)::Trajectory
    soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(state(env), model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon)
    if sampler.is_directed
        stree = soltree1
    else
        stree = soltree
    end
    all_leafs = filter(i->length(i.children) == 0, collect(values(stree)))
    all_inner_nodes = filter(i->length(i.children) != 0, collect(values(stree)))
    all_inner_nodes_expr = Set(map(x->x.ex, all_inner_nodes))
    # @show typeof(all_leafs)
    number_of_inner_nodes = length(stree) - length(all_leafs)
    sorted_leafs = sort(all_leafs, by=x->(exp_size(x.ex), x.depth))
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
        if length(inner_to_leaf) == number_of_inner_nodes
            continue
        end
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
    @assert length(inner_to_leaf) == length(soltree)
    final_next_states = collect(keys(inner_to_leaf))
    final_rewards = collect(values(inner_to_leaf))
    trj = Trajectory(1, final_next_states, final_next_states, final_rewards, final_next_states, falses(length(final_next_states)))

    return trj
end
# function sample_trajectory(sampler::PlanningSampler, env::AbstractEnvironment, model::AbstractModel)::PlanningTrajectory
#     soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(state(env), model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon)

# end


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
preprocess_trajectory(trj::Trajectory) = get_target_values(trj), get_input_values(trj)
# preprocess_trajectory(trj::TupleTrajectory) = get_target_values(trj), get_input_values(trj)
# preprocess_trajectory(trj::TreeTrajectory) = get_target_values(trj), get_input_values(trj)
preprocess_trajectory(trj::Tuple{Int, Trajectory}) = get_target_values(trj), get_input_values(trj)
preprocess_trajectory(trj::Tuple{Int, TreeTrajectory}) = get_target_values(trj), get_input_values(trj)
preprocess_trajectory(trj::NamedTuple) = trj.labels, get_input_values(trj.input_values)
preprocess_trajectory(trj::Tuple{Int, PolicyTrajectory}) = get_target_values(trj[2]), get_input_values(trj[2])

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