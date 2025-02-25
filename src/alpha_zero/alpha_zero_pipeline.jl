function get_all_subtrees!(ex::NodeID, all_subtrees::Vector, parent=[])
    node = MyModule.nc[ex]
    for (ind,i) in enumerate([node.left, node.right])
        if !(MyModule.nc[i].iscall) && MyModule.nc[i].head ∉ [:&&, :||]
            continue
        end
        if i == MyModule.nullid
            continue
        end
        pos = vcat(parent, ind)
        push!(all_subtrees, (i, pos))
        get_all_subtrees!(i, all_subtrees, pos)
    end
end

function my_rewrite!(ex::NodeID, pos, new_exp_part::NodeID)
    if isempty(pos)
        return new_exp_part
    end
    node = MyModule.nc[ex]

    if pos[1] == 1
        new_part = my_rewrite!(node.left, pos[2:end], new_exp_part)
        # node.left = new_part
        new_node = OnlyNode(node.head, node.iscall, node.v, new_part, node.right)
    else        
        new_part = my_rewrite!(node.right, pos[2:end], new_exp_part)
        # node.right = new_part
        new_node = OnlyNode(node.head, node.iscall, node.v, node.left, new_part)
    end
    return get!(MyModule.nc, new_node)
end

function rollout(ex, model; max_expansions=10)
    soltree = Dict{UInt64, Node}()
    current = Node(ex, (), hash(ex), 0)
    soltree[current.node_id] = current
    smallest_node = current
    smallest_node_size = exp_size(ex)
    for step in 1:max_expansions
        ex = current.ex
        subtrees = [(ex, [])]
        get_all_subtrees!(ex, subtrees)
        tree_action = []
        for (st, pos) in subtrees
            new_ex = [(intern!(r(st)), pos, ind, st) for (ind,r) in enumerate(theory)]
            # filter empty rewrites
            filtered_ex = filter(x->!isnothing(x[1]), new_ex)
            new_ex = map(x->(x..., my_rewrite!(ex, x[2], x[1])), filtered_ex)
            # filter repeated 
            filtered_ex = filter(x->!haskey(soltree, hash(x[5])), new_ex)
            append!(tree_action, filtered_ex)
        end
        isempty(tree_action) && break
        root_embedding = MyModule.general_cached_inference(ex, Expr, model)
        o = map(tree_action) do (_, _, rule_id, subtree, _)
            subtree_embedding = MyModule.general_cached_inference(subtree, Expr, model)
            model.heuristic(subtree_embedding)[rule_id]
        end
        min_index = argmax(o)
        _, pos, rule_id, subtree, current_exp = tree_action[min_index]
        current_size = exp_size(current_exp)
        current = Node(current_exp, (pos, rule_id), current.node_id, current.depth + 1)
        soltree[current.node_id] = current
        if smallest_node_size > current_size
            smallest_node_size = current_size
            smallest_node = current
        end
    end
    return soltree, smallest_node, []
end


function backpropagate_stats!(n, nodes_stats, soltree)
    if n.parent == n.node_id
        return
    end
    nodes_stats[n.parent][1] += 1
    nodes_stats[n.parent][2] += nodes_stats[n.node_id][2] + nodes_stats[n.node_id][3]
    backpropagate_stats!(soltree[n.parent], nodes_stats, soltree)
end


function monte_carlo_expand!(parent, soltree, open_list)
    new_ex, rules_applied = all_expand(parent.ex, theory)
    filtered_new_ex = filter(x->parent.ex != x, new_ex)
    
    new_nodes = map(x->Node(x[1], x[2], parent.node_id, parent.depth + 1), zip(filtered_new_ex, rules_applied))
    # tmp = filter(x->push_to_tree!(soltree, x), new_nodes)
    for x in new_nodes
        res = push_to_tree!(soltree, x)
        # if res && x.node_id != parent.node_id && x.node_id ∉ parent.children && x.parent == parent.node_id
        if res && x.node_id ∉ parent.children
            push!(parent.children, x.node_id)
            enqueue!(open_list, x, 0)
            return 
        end
    end
    # if isempty(filtered_new_ex)
    n,v = dequeue_pair!(open_list)
    monte_carlo_expand!(first(open_list)[1], soltree, open_list)
    # end
end