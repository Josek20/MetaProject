mutable struct Node{E}
    ex::E
    rule_index::Tuple
    children::Vector{UInt64}
    parent::UInt64
    depth::Int
    node_id::UInt64
end


Node(ex, rule_applied, parent_id::UInt64, depth::Int) = Node(ex, rule_applied, UInt64[], parent_id, depth, hash(ex))


function push_to_tree!(soltree::Dict, new_node::Node)
    node_id = new_node.node_id
    if haskey(soltree, node_id)
        old_node = soltree[node_id]
        if new_node.depth < old_node.depth
            soltree[node_id].depth = new_node.depth
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


function expand_node!(parent::Node, soltree, open_list, model; theory=theory)
    new_ex, _ = all_expand(parent.ex, theory)
    new_nodes = map(x->Node(x, (), parent.node_id, parent.depth + 1), new_ex)
    new_nodes = filter(x->push_to_tree!(soltree, x), new_nodes)
    isempty(new_nodes) && return
    o = map(x->model(x.ex), new_nodes)
    for (v,n) in zip(o, new_nodes)
        enqueue!(open_list, (n, v))
    end
end


function build_tree!(soltree, open_list, close_list, model; max_expansions=1000, max_depth=10)
    expansions = 0
    while !isempty(open_list)
        expansions != max_expansions && break
        node, _ = dequeue_pair!(open_list)

        node.depth == max_depth && continue
        
        expand_node!(node, soltree, open_list, model)
        push!(close_list, node.node_id)
        expansions += 1
    end
end


function extract_smallest_node(soltree)
    smallest_node = nothing
    smallest_node_size = -1
    for (k, n) in soltree
        ex_size = exp_size(n.ex)
        if isnothing(smallest_node) || (ex_size <= smallest_node_size && n.depth < smallest_node.depth)
            smallest_node = n
            smallest_node_size = ex_size
        end
    end
    return(smallest_node)
end


function initialize_tree_search(ex, model; max_expansions=1000, max_depth=10)
    open_list = PriorityQueue{Node, Float32}()
    close_list = Set{UInt64}()

    soltree = Dict{UInt64, Node}()
    root = Node(ex, (), hash(ex), 0)
    soltree[root.node_id] = root
    o = model(root.ex)
    enqueue!(open_list, root, o)
    build_tree!(soltree, open_list, close_list, model, max_expansions=max_expansions, max_depth=max_depth)

    smallest_node = extract_smallest_node(soltree)
    return(soltree, smallest_node, root)
end