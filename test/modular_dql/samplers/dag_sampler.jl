mutable struct DAGSampler <: AbstractSampler
    max_steps::Int
    max_depth::Int
    epsilon::Float32
    eps_decay::Float32
    batch::Int
    gamma::Float32
end
DAGSampler(;max_steps=50, max_depth=100, epsilon=0.0, eps_decay=0.95, batch=64, gamma=1) = DAGSampler(max_steps, max_depth, epsilon, eps_decay, batch, gamma)


function run_search(sampler::DAGSampler, env, model)
    soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(env, model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon, gamma=sampler.gamma)
    return soltree, smallest_node, root, soltree1
end

function preprocess(sampler::DAGSampler, raw)
    soltree, smallest_node, root, soltree1 = raw
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

finalize_trajectory(sampler::DAGSampler, processed) = processed