function rollout(ex, model; max_expansions=10)
    soltree = Dict{UInt64, Node}()
    current = Node(ex, (), hash(ex), 0)
    soltree[current.node_id] = current
    smallest_node = current
    smallest_node_size = exp_size(ex)
    for step in 1:max_expansions
        ex = current.ex
        new_ex, _ = all_expand(ex, theory)
        new_nodes = map(x->Node(x, (), current.node_id, current.depth + 1), new_ex)
        filtered_nodes = filter(x->push_to_tree!(soltree, x), new_nodes)
        isempty(filtered_nodes) && break
        # @show length(filtered_nodes)
        o = map(x->model(x.ex), filtered_nodes)
        min_index = argmax(o)
        current = filtered_nodes[min_index]
        current_size = exp_size(current.ex)
        if smallest_node_size > current_size
            smallest_node_size = current_size
            smallest_node = current
        end
    end
    return soltree, smallest_node, []
end


function rollout_expand!(parent::Node, soltree, open_list, model; theory=theory)
    ex = parent.ex
    new_ex, rules_applied = all_expand(ex, theory)
    new_nodes = map(x->Node(x[1], x[2], parent.node_id, parent.depth + 1), zip(new_ex, rules_applied))
    new_nodes = filter(x->push_to_tree!(soltree, x), new_nodes)
    isempty(new_nodes) && return
    o = model(ex)
    for ((p, rid), n) in zip(rules_applied, new_nodes)
        enqueue!(open_list, n, o[rid])
    end
    nodes_ids = map(x->x.node_id, new_nodes)
    append!(parent.children, nodes_ids)
end
