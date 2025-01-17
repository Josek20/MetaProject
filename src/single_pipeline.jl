
function push_to_tree!(soltree::Dict, new_node::Node)
    node_id = new_node.node_id
    if haskey(soltree, node_id)
        old_node = soltree[node_id]
        if new_node.depth < old_node.depth
            soltree[node_id].depth = new_node.depth
            soltree[node_id].rule_index = new_node.rule_index
            push!(soltree[new_node.parent].children, new_node.node_id)
            filter!(x->x!=old_node.node_id, soltree[old_node.parent].children) 
            soltree[node_id].parent = new_node.parent
        end
        return (false)
    else
        soltree[node_id] = new_node
        return (true)
    end
end


function expand_node!(parent::Node, soltree, heuristic, open_list, config::NamedTuple) 
    ex = parent.ex
    succesors = execute(ex, config.theory, config.exp_cache)
    new_nodes = map(x-> Node(x[2], x[1], parent.node_id, parent.depth + 1, config.expr_cache), succesors)
    filtered_new_nodes = filter(x-> push_to_tree!(soltree, x), new_nodes)
    isempty(filtered_new_nodes) && return(false)
    exprs = map(x->x.ex, filtered_new_nodes)
    o = map(x->heuristic(x.ex, config.heuristic_cache), filtered_new_nodes)
    for (v,n) in zip(o, filtered_new_nodes)
        enqueue!(open_list, n, v)
    end
    nodes_ids = map(x->x.node_id, filtered_new_nodes)
    append!(parent.children, nodes_ids)
    if in(:1, exprs)
        return true
    end
    return false
end


function build_tree!(soltree::Dict{UInt64, Node}, heuristic, open_list::PriorityQueue, close_list::Set{UInt64}, config::NamedTuple)
    step = 0
    reached_goal = false
    while length(open_list) > 0
        if config.max_steps <= step
            break
        end
        nodes, prob = dequeue_pair!(open_list)
        push!(close_list, nodes.node_id)
        # if nodes.depth >= max_depth
        #     continue
        # end
        step += 1
        reached_goal = expand_node!(nodes, soltree, heuristic, open_list, config)
        if reached_goal
            return true
        end
    end
    return false
end


function initialize_tree_search(heuristic, ex::Expr, config::NamedTuple)
    soltree = Dict{UInt64, Node}()
    open_list = PriorityQueue{Node, Float32}()
    close_list = Set{UInt64}()
    root = Node(ex, (0,0), nothing, 0, config.expr_cache)
    o = heuristic(root.ex, cache)
    soltree[root.node_id] = root
    enqueue!(open_list, root, only(o))
    reached_goal = build_tree!(soltree, heuristic, open_list, close_list, config)
    smallest_node = extract_smallest_terminal_node(soltree, close_list, config.size_cache)
    simplified_expression = smallest_node.ex
    big_vector, hp, hn, proof_vector, cr = extract_training_data(smallest_node, soltree)
    return (;simplified_expression=simplified_expression, bd=big_vector, hp=hp, hn=hn, proof=proof_vector, saturated=length(open_list) == 0 || reached_goal)
end

