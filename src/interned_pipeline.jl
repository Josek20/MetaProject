Node(ex::NodeID, rule_index::Tuple, parent, depth::Int64, ee::Union{ProductNode, Nothing}) = Node(ex, rule_index, UInt64[],  parent, depth, hash(nc[ex]), ee)

function interned_extract_training_data(node, soltree, sym_enc=sym_enc)
    training_exp=[]
    hp=Vector[]
    hn=Vector[]
    proof_vector=[]
    extract_training_data!(node, soltree, training_exp, hp, hn, proof_vector)
    hp = reduce(vcat, hp)
    hn = reduce(vcat, hn)
    training_exp = [expr(nc, i) for i in training_exp]
    tdata = MyModule.no_reduce_multiple_fast_ex2mill(training_exp, sym_enc)
    return tdata, hp, hn, reverse(proof_vector), 0
end


@my_cache function exp_size(x::NodeID)
    # get!(size_cache, x) do
    node = nc[x]
    if node == nullnode
        return 0f0
    elseif !(node.iscall)
        return 1f0
    end
    return exp_size(node.left) + exp_size(node.right) + 1
    # end
end


@my_cache function all_expand(node_id::NodeID, theory)
    node = nc[node_id]
    !(node.iscall) && return(NodeID[])
    # get!(expr_cache, node_id) do
    self = [r(node_id) for r in theory]
    self = [isa(x, Int) ? intern!(x) : x for x in self]
    self = convert(Vector{NodeID}, filter(!isnothing, self))
    # self = filter(x->x!=InternedExpr.nullnode, self)
    lefts = all_expand(node.left, theory)
    rights = all_expand(node.right, theory)
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
    # end
end


function interned_expand_node!(parent::Node, soltree, heuristic, open_list, all_symbols, symbols_to_index, theory) 
    ex = parent.ex
    succesors = filter(x->x!=ex, all_expand(ex, theory))
    new_nodes = map(x-> Node(x, (0,0), parent.node_id, parent.depth + 1, nothing), succesors)
    filtered_new_nodes = filter(x-> push_to_tree!(soltree, x), new_nodes)
    isempty(filtered_new_nodes) && return(false)
    # o = map(x->exp_size(x.ex), filtered_new_nodes)
    o = map(x->only(heuristic(x.ex)), filtered_new_nodes)
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


function interned_build_tree!(soltree::Dict{UInt64, Node}, heuristic, open_list::PriorityQueue, close_list::Set{UInt64}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int64}, max_steps, max_depth, theory)
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
        push!(close_list, nodes.node_id)
        step += 1
        reached_goal = interned_expand_node!(nodes, soltree, heuristic, open_list, all_symbols, symbols_to_index, theory)

        # if reached_goal
        #     return true 
        # end
    end
    return false
end


function interned_initialize_tree_search(heuristic, ex::Expr, max_steps, max_depth, all_symbols, symbols_to_index, theory)
    soltree = Dict{UInt64, Node}()
    open_list = PriorityQueue{Node, Float32}()
    close_list = Set{UInt64}()
    ex = intern!(ex)
    root = Node(ex, (0,0), nothing, 0, nothing)
    o = heuristic(root.ex)
    soltree[root.node_id] = root
    enqueue!(open_list, root, only(o))
    reached_goal = interned_build_tree!(soltree, heuristic, open_list, close_list, all_symbols, symbols_to_index, max_steps, max_depth, theory)
    smallest_node = extract_smallest_terminal_node(soltree, close_list)
    simplified_expression = smallest_node.ex
    # @show length(soltree)
    big_vector, hp, hn, proof_vector, _ = interned_extract_training_data(smallest_node, soltree)
    # tmp = []
    return simplified_expression, [], big_vector, length(open_list) == 0 || reached_goal, hp, hn, root, proof_vector, []
end
