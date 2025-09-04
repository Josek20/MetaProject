mutable struct Node{E}
    ex::E
    rule_index::Tuple
    children::Vector{UInt64}
    parent::UInt64
    depth::Int
    node_id::UInt64
end


Node(ex, rule_applied, parent_id::UInt64, depth::Int) = Node(ex, rule_applied, UInt64[], parent_id, depth, hash(ex))
# Node1(ex, rule_applied, parent_id::UInt64, depth::Int) = Node(ex, rule_applied, UInt64[], parent_id, depth, hash(ex, hash(depth)))
Node1(ex, rule_applied, parent_ex, parent_id::UInt64, depth::Int) = Node(ex, rule_applied, UInt64[], parent_id, depth, hash(ex, hash(parent_ex)))


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


function expand_node1!(parent::Node, soltree, soltree1, open_list, model; theory=theory)
    @timeit TO "extracting all children" new_ex, rules_applied = all_expand(parent.ex, theory)
    new_nodes = map(x->Node(x[1], x[2], parent.node_id, parent.depth + 1), zip(new_ex, rules_applied))
    # new_nodes1 = map(x->Node1(x[1], x[2], hash(parent.ex, hash(parent.depth)), parent.depth + 1), zip(new_ex, rules_applied))
    # @show hash(parent.ex, hash(soltree[parent.parent].ex)
    # parent_id1 = parent.depth == 0 ? hash(parent.ex, hash(parent.ex)) : hash(parent.ex, hash(soltree[parent.parent].ex))
    # parent_id1 = parent.depth == 0 ? hash(parent.ex, hash(parent.ex)) : findfirst(x->x.ex == parent.ex, soltree)
    parent_id1 = parent.depth == 0 ? hash(parent.ex, hash(parent.ex)) : hash(parent.ex, hash(soltree[parent.parent].ex))
    # @show parent_id1
    new_nodes1 = map(x->Node1(x[1], x[2], parent.ex, parent_id1, parent.depth + 1), zip(new_ex, rules_applied))
    new_nodes = filter(x->push_to_tree!(soltree, x), new_nodes)
    # new_nodes1 = filter(x->push_to_tree!(soltree1, x), new_nodes1)
    # new_nodes1 = filter(x->push_to_tree1!(soltree1, x), new_nodes1)
    new_nodes1_indices = []
    for (ind, i) in enumerate(new_nodes1)
        if i.ex != parent.ex
            push!(new_nodes1_indices, ind)
            soltree1[i.node_id] = i
        end
    end
    if isempty(new_nodes)
        nodes_ids2 = map(x->x.node_id, new_nodes1[new_nodes1_indices])
        append!(soltree1[parent_id1].children, nodes_ids2)
        return
    end
    @timeit TO "new children inference cached" o = map(x->only(model(x.ex)), new_nodes)
    # @timeit TO "transfrom new children to Expr" expr_data = map(x->expr(MyModule.nc, x.ex), new_nodes)
    # @timeit TO "transfrom new children into Mill structure" input_data = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(expr_data, sym_enc))
    # @timeit TO "new_children inference batched" o = MyModule.heuristic(model, input_data)
    
    for (v,n) in zip(o, new_nodes)
        enqueue!(open_list, n, v)
    end
    nodes_ids = map(x->x.node_id, new_nodes)
    append!(parent.children, nodes_ids)
    nodes_ids2 = map(x->x.node_id, new_nodes1[new_nodes1_indices])
    # append!(soltree1[hash(parent.ex, hash(parent.depth))].children, nodes_ids2)
    # @show parent.ex
    # @show soltree1
    # @show new_nodes1[new_nodes1_indices][1].parent
    append!(soltree1[parent_id1].children, nodes_ids2)
    # append!(soltree1[hash(parent.ex, hash(soltree[parent.parent].ex))].children, nodes_ids2)
end

function expand_node2!(parent::Node, soltree, open_list, model; theory=theory)
    new_ex, rules_applied = all_expand(parent.ex, theory)
    not_filtered_new_nodes = map(x->Node1(x[1], x[2], parent.node_id, parent.depth + 1), zip(new_ex, rules_applied))
    new_nodes = filter(x->push_to_tree!(soltree, x), not_filtered_new_nodes)
    isempty(new_nodes) && return
    o = map(x->only(model(x.ex)), new_nodes)
    for (v,n) in zip(o, new_nodes)
        enqueue!(open_list, n, v)
    end
    nodes_ids = map(x->x.node_id, not_filtered_new_nodes)
    append!(parent.children, nodes_ids)
end

function expand_node!(parent::Node, soltree, open_list, model; theory=theory)
    new_ex, rules_applied = all_expand(parent.ex, theory)
    new_nodes = map(x->Node(x[1], x[2], parent.node_id, parent.depth + 1), zip(new_ex, rules_applied))
    new_nodes = filter(x->push_to_tree!(soltree, x), new_nodes)
    isempty(new_nodes) && return
    o = map(x->only(model(x.ex)), new_nodes)
    for (v,n) in zip(o, new_nodes)
        enqueue!(open_list, n, v)
    end
    nodes_ids = map(x->x.node_id, new_nodes)
    append!(parent.children, nodes_ids)
end


function build_tree_epsilon_greedy!(soltree, soltree1, open_list, close_list, model; max_expansions=1000, max_depth=10, epsilon=1.0)
    expansions = 0
    greedy_set = Set()
    while !isempty(open_list)
        expansions == max_expansions && break
            if rand() > epsilon
                @timeit TO "choose node from queue" begin
                    node, _ = dequeue_pair!(open_list)
                end
            else
                @timeit TO "choose node random sampled" begin
                    node, _ = rand(open_list.xs)
                    dequeue!(open_list, node)
                end

            end
        push!(close_list, node.node_id)
        # push!(greedy_set, node)
        node.depth == max_depth && continue
        
        @timeit TO "expand node" expand_node1!(node, soltree, soltree1, open_list, model)
        expansions += 1
    end
end


function build_tree!(soltree, open_list, close_list, model; max_expansions=1000, max_depth=10)
    expansions = 0
    while !isempty(open_list)
        expansions == max_expansions && break
        node, _ = dequeue_pair!(open_list)
        push!(close_list, node.node_id)

        node.depth == max_depth && continue
        
        expand_node!(node, soltree, open_list, model)
        expansions += 1
    end
end


function build_tree1!(soltree, soltree1, open_list, close_list, model; max_expansions=1000, max_depth=10)
    expansions = 0
    while !isempty(open_list)
        expansions == max_expansions && break
        node, _ = dequeue_pair!(open_list)
        push!(close_list, node.node_id)

        node.depth == max_depth && continue
        
        expand_node1!(node, soltree, soltree1, open_list, model)
        expansions += 1
    end
end

function extract_smallest_node(soltree)
    smallest_node = nothing
    smallest_node_size = typemax(Int)
    for (k, n) in soltree
        n.depth == 0 && continue
        # @show n.ex
        ex_size = exp_size(n.ex)
        if ex_size < smallest_node_size
            smallest_node = n
            smallest_node_size = ex_size
        elseif ex_size == smallest_node_size && n.depth < smallest_node.depth
            smallest_node = n
        end
    end
    return(smallest_node)
end


function initialize_tree_search(ex, model; max_expansions=1000, max_depth=10, epsilon=1.0)
    open_list = PriorityQueue{Node, Float32}()
    close_list = Set{UInt64}()

    soltree = Dict{UInt64, Node}()
    soltree1 = Dict{UInt64, Node}()
    root = Node(ex, (), hash(ex), 0)
    # root1 = Node1(ex, (), hash(ex, hash(ex)), 0)
    root1 = Node(ex, (), UInt64[], hash(ex, hash(ex)), 0, hash(ex, hash(ex)))
    soltree[root.node_id] = root
    soltree1[root1.node_id] = root1
    o = only(model(root.ex))
    enqueue!(open_list, root, o)
    build_tree!(soltree, open_list, close_list, model, max_expansions=max_expansions, max_depth=max_depth)
    # build_tree1!(soltree, soltree1, open_list, close_list, model, max_expansions=max_expansions, max_depth=max_depth)
    smallest_node = extract_smallest_node(soltree)
    # return(soltree, smallest_node, root, soltree1)
    return(soltree, smallest_node, root)
end


function initialize_tree_search_epsilon(ex, model; max_expansions=1000, max_depth=10, epsilon=1.0)
    # if isa(model, ExprModel) || isa(model, Function)
    #     open_list = PriorityQueue{Node, Float32}(Base.Order.Reverse)
    # else
    if isa(model, Function)
        open_list = PriorityQueue{Node, Float32}(Base.Order.Reverse)
    elseif size(model.heuristic.layers[end].weight)[1] == 2
        open_list = PriorityQueue{Node, Tuple{Float32, Float32}}(Base.Order.Reverse)
    else
        open_list = PriorityQueue{Node, Float32}(Base.Order.Reverse)
    end
    # end
    close_list = Set{UInt64}()

    soltree = Dict{UInt64, Node}()
    soltree1 = Dict{UInt64, Node}()
    root = Node(ex, (), hash(ex), 0)
    # root1 = Node1(ex, (), hash(ex, hash(0)), 0)
    root1 = Node(ex, (), UInt64[], hash(ex, hash(ex)), 0, hash(ex, hash(ex)))
    soltree[root.node_id] = root
    soltree1[root1.node_id] = root1
    o = only(model(root.ex))
    enqueue!(open_list, root, o)
    @timeit TO "build tree epsilon" build_tree_epsilon_greedy!(soltree, soltree1, open_list, close_list, model; max_expansions=max_expansions, max_depth=max_depth, epsilon=epsilon)
    @timeit TO "extract smallest node" smallest_node = extract_smallest_node(soltree)
    return(soltree, smallest_node, root, soltree1)
end


function initialize_tree_search_epsilon_for_parallel(ex, model; max_expansions=1000, max_depth=10, epsilon=1.0)
    # if isa(model, ExprModel) || isa(model, Function)
    #     open_list = PriorityQueue{Node, Float32}(Base.Order.Reverse)
    # else
    open_list = PriorityQueue{Node, Float32}(Base.Order.Reverse)
    # open_list = PriorityQueue{Node, Tuple{Float32, Float32}}(Base.Order.Reverse)
    # end
    close_list = Set{UInt64}()

    soltree = Dict{UInt64, Node}()
    soltree1 = Dict{UInt64, Node}()
    root = Node(ex, (), hash(ex), 0)
    # root1 = Node1(ex, (), hash(ex, hash(0)), 0)
    root1 = Node(ex, (), UInt64[], hash(ex, hash(ex)), 0, hash(ex, hash(ex)))
    soltree[root.node_id] = root
    soltree1[root1.node_id] = root1
    o = only(model(root.ex))
    enqueue!(open_list, root, o)
    @timeit TO "build tree epsilon" build_tree_epsilon_greedy!(soltree, soltree1, open_list, close_list, model; max_expansions=max_expansions, max_depth=max_depth, epsilon=epsilon)
    @timeit TO "extract smallest node" smallest_node = extract_smallest_node(soltree)
    # Change interned to Expr
    for (i,j) in soltree
        tmp = MyModule.expr(MyModule.nc, j.ex)
        soltree[i] = Node(tmp, j.rule_index, j.children, j.parent, j.depth, j.node_id)
    end
    for (i,j) in soltree1
        tmp = MyModule.expr(MyModule.nc, j.ex)
        soltree1[i] = Node(tmp, j.rule_index, j.children, j.parent, j.depth, j.node_id)
    end 
    root = soltree[root.node_id]
    smallest_node = soltree[smallest_node.node_id]
    return(soltree, smallest_node, root, soltree1)
end