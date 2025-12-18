mutable struct QTreeSampler <: AbstractSampler
    max_steps::Int
    max_depth::Int
    epsilon::Float32
    eps_decay::Float32
    batch::Int
    gamma::Float32
end
QTreeSampler(;max_steps=50, max_depth=100, epsilon=0.0, eps_decay=0.95, batch=64, gamma=1) = QTreeSampler(max_steps, max_depth, epsilon, eps_decay, batch, gamma)



function run_search(sampler::DGSampler, env, model)
    soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(env, model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon, gamma=sampler.gamma)
    return soltree, smallest_node, root, soltree1
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


function preprocess(sampler::QTreeSampler, raw)
    soltree, smallest_node, root, soltree1 = raw
    targets = Dict()
    get_Q_targets!(root, soltree, model, targets, γ=sampler.gamma)
    mcache = Dict(soltree[k].ex=>v for (k,v) in targets)
    if sampler.batch < 1
        return get_trajectory_from_mcache(root.ex, mcache, sampler)
    else
        return get_trajectory_from_mcache_planning(soltree, root, smallest_node, mcache, sampler)
    end 
end
