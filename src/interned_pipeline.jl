Node(ex::NodeID, rule_index::Tuple, parent, depth::Int64, ee::Union{ProductNode, Nothing}) = Node(ex, rule_index, UInt64[],  parent, depth, hash(nc[][ex]), ee)

function exp_size(x::NodeID, size_cache)
    get!(size_cache, x) do
        node = nc[][x]
        if node == nullnode
            return 0f0
        elseif !(node.iscall)
            return 1f0
        end
        return exp_size(node.left, size_cache) + exp_size(node.right, size_cache) + 1
    end
end


function all_expand(node_id::NodeID, theory, expr_cache)
    node = nc[][node_id]
    !(node.iscall) && return(NodeID[])
    get!(expr_cache, node_id) do
        self = [r(node_id) for r in theory]
        self = [isa(x, Int) ? intern!(x) : x for x in self]
        self = convert(Vector{NodeID}, filter(!isnothing, self))
        # self = filter(x->x!=InternedExpr.nullnode, self)
        lefts = all_expand(node.left, theory, expr_cache)
        rights = all_expand(node.right, theory, expr_cache)
        lefts = isempty(lefts) ? [node.left] : lefts
        rights = isempty(rights) ? [node.right] : rights
        if length(lefts) > 1 && length(rights) > 1
            childs_left = map(Iterators.product(lefts, [node.right])) do (l, r)
                intern!(OnlyNode(node.head, node.iscall, node.v, l, r))
            end |> vec
            childs_right = map(Iterators.product([node.left], rights)) do (l, r)
                intern!(OnlyNode(node.head, node.iscall, node.v, l, r))
            end |> vec
            childs = vcat(childs_left, childs_right)
        else 
            childs = map(Iterators.product(lefts, rights)) do (l, r)
                intern!(OnlyNode(node.head, node.iscall, node.v, l, r))
            end |> vec
        end
        return vcat(self, childs)
    end
end


function interned_expand_node!(parent::Node, soltree, heuristic, open_list, all_symbols, symbols_to_index, theory, cache, exp_cache, size_cache, expr_cache, alpha=0.9, lambda=-100.4) 
    ex = parent.ex
    succesors = filter(x->x!=ex, all_expand(ex, theory, exp_cache))
    new_nodes = map(x-> Node(x, (0,0), parent.node_id, parent.depth + 1, nothing), succesors)
    filtered_new_nodes = filter(x-> push_to_tree!(soltree, x), new_nodes)
    isempty(filtered_new_nodes) && return(false)
    # o = map(x->alpha * exp_size(x.ex, size_cache) + (1 - alpha) * only(heuristic(x.ex, cache)), filtered_new_nodes)
    o = map(x->exp_size(x.ex, size_cache), filtered_new_nodes)
    # o = map(x->node_count(x.ex.ex), filtered_new_nodes)
    # o = map(x->only(heuristic(x.ex, cache)), filtered_new_nodes)
    # o = heuristic(exprs, cache)
    for (v,n) in zip(o, filtered_new_nodes)
        enqueue!(open_list, n, v)
    end
    nodes_ids = map(x->x.node_id, filtered_new_nodes)
    append!(parent.children, nodes_ids)
    # if in(:1, exprs)
    #     return true
    # end
    return false
end


function interned_build_tree!(soltree::Dict{UInt64, Node}, heuristic, open_list::PriorityQueue, close_list::Set{UInt64}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int64}, max_steps, max_depth, theory, cache, exp_cache, size_cache, expr_cache, alpha)
    step = 0
    reached_goal = false
    epsilon = 0.05
    expand_n = 25
    expanded_depth = []
    while length(open_list) > 0
        if max_steps <= step
            break
        end
        nodes, prob = dequeue_pair!(open_list)
        step += 1
        reached_goal = interned_expand_node!(nodes, soltree, heuristic, open_list, all_symbols, symbols_to_index, theory, cache, exp_cache, size_cache, expr_cache, alpha)
        if reached_goal
            return true 
        end
    end
    return false
end


function interned_initialize_tree_search(heuristic, ex::Expr, max_steps, max_depth, all_symbols, theory, cache, exp_cache, size_cache, expr_cache, alpha)
    soltree = Dict{UInt64, MyModule.Node}()
    open_list = PriorityQueue{MyModule.Node, Float32}()
    close_list = Set{UInt64}()
    ex = intern!(ex)
    root = MyModule.Node(ex, (0,0), nothing, 0, nothing)
    o = MyModule.exp_size(root.ex, cache)
    soltree[root.node_id] = root
    enqueue!(open_list, root, only(o))
    reached_goal = MyModule.interned_build_tree!(soltree, heuristic, open_list, close_list, all_symbols, symbols_to_index, 1000, 10, theory, cache, exp_cache, size_cache, expr_cache, 0.5)
    smallest_node = extract_smallest_terminal_node(soltree, close_list, size_cache)
    simplified_expression = smallest_node.ex
    big_vector, hp, hn, proof_vector = extract_training_data(smallest_node, soltree)
    tmp = []
    return simplified_expression, [], big_vector, length(open_list) == 0 || reached_goal, hp, hn, root, proof_vector, tmp
end
