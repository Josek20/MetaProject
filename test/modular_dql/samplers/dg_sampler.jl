mutable struct DGSampler <: AbstractSampler
    max_steps::Int
    max_depth::Int
    epsilon::Float32
    eps_decay::Float32
    batch::Int
    gamma::Float32
end
DGSampler(;max_steps=50, max_depth=100, epsilon=0.0, eps_decay=0.95, batch=64, gamma=1) = DGSampler(max_steps, max_depth, epsilon, eps_decay, batch, gamma)


function run_search(sampler::DGSampler, env, model)
    soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(env, model; max_expansions=sampler.max_steps, max_depth=sampler.max_depth, epsilon=sampler.epsilon, gamma=sampler.gamma)
    return soltree, smallest_node, root, soltree1
end

function preprocess(sampler::DGSampler, raw)
    soltree, smallest_node, root, soltree1 = raw
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

finalize_trajectory(sampler::DGSampler, processed) = processed