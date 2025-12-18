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


function expand_node1!(parent::Node, root::Node, soltree, soltree1, open_list, model; theory=theory, gamma=1.0)
    @timeit TO "extracting all children" new_ex, rules_applied = all_expand(parent.ex, theory)
    # @show parent.rule_index, length(soltree)
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
    # @timeit TO "new children inference cached" o = map(x->exp_size(root.ex) - exp_size(x.ex) + gamma * only(model(x.ex)), new_nodes)
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


function expand_node_policy!(node, root, soltree, soltree1,  open_list_nodes, open_list_weights, model; gamma=1.0)
    new_ex, rules_applied = all_expand(node.ex, theory)
    new_nodes = map(x->Node(x[1], x[2], node.node_id, node.depth + 1), zip(new_ex, rules_applied))
    new_nodes = filter(x->push_to_tree!(soltree, x), new_nodes)
    isempty(new_nodes) && return
    o = map(x->only(model(x.ex)), new_nodes)
    # o = map(x->exp_size(root.ex) - exp_size(x.ex) + gamma * only(model(x.ex)), new_nodes)
    for (v,n) in zip(o, new_nodes)
        push!(open_list_nodes, n.node_id)
        push!(open_list_weights, v)
    end
    nodes_ids = map(x->x.node_id, new_nodes)
    append!(node.children, nodes_ids)
end


function build_tree_epsilon_greedy!(soltree, soltree1, open_list, close_list, model; max_expansions=1000, max_depth=10, epsilon=1.0, gamma=1.0)
    expansions = 0
    greedy_set = Set()
    start_time = time()
    root = first(soltree)[2]
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
        if time() - start_time > 10.0
            break
        end
        push!(close_list, node.node_id)
        # push!(greedy_set, node)
        node.depth == max_depth && continue
        
        @timeit TO "expand node" expand_node1!(node, root, soltree, soltree1, open_list, model, gamma=gamma)
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


function build_tree_bfs!(soltree, soltree1, open_list, close_list, model;  max_expansions=1000, max_depth=10)
    root = first(soltree)[2]
    for dp in 1:max_depth
        new_open_list = empty(open_list)
        while !isempty(open_list)
            node, _ = dequeue_pair!(open_list)
            # push!(close_list, node.node_id)
            expand_node!(node, soltree, new_open_list, model)
            # expand_node1!(node, root, soltree, soltree1, new_open_list, model)
        end
        open_list = new_open_list
    end
end


function build_tree_search_tree!(soltree, soltree1, open_list, close_list, model, all_trees_history; max_expansions=1000, max_depth=10, epsilon=1.0, gamma=1.0)
    expansions = 0
    greedy_set = Set()
    start_time = time()
    root = first(soltree)[2]
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
        if time() - start_time > 10.0
            break
        end
        push!(close_list, node.node_id)
        # push!(greedy_set, node)
        node.depth == max_depth && continue
        
        @timeit TO "expand node" expand_node1!(node, root, soltree, soltree1, open_list, model, gamma=gamma)
        push!(all_trees_history, collect(keys(soltree)))
        expansions += 1
    end
end


function build_tree_policy!(soltree, tree_history, open_list_nodes, open_list_weights, close_list, model; max_expansions=max_expansions, max_depth=max_depth, gamma=gamma)
    expansions = 0
    greedy_set = Set()
    start_time = time()
    root = first(soltree)[2]
    tmp = deepcopy(open_list_nodes)
    while true
        expansions == max_expansions && break
        
        # mask values
        # not_expanded = setdiff(open_list_nodes, close_list)
        mask = .!in.(open_list_nodes, Ref(close_list))
        not_masked_nodes = open_list_nodes[mask]
        isempty(not_masked_nodes) && break
        not_masked_weights = open_list_weights[mask]
        # sample from not masked
        node_index = StatsBase.sample(1:length(not_masked_weights), Weights(softmax(not_masked_weights)))
        node = soltree[not_masked_nodes[node_index]]

        if time() - start_time > 10.0
            break
        end
        push!(close_list, node.node_id)
        push!(tree_history, (deepcopy(open_list_nodes), not_masked_nodes, node_index))
        node.depth == max_depth && continue
        @timeit TO "expand node policy" expand_node_policy!(node, root, soltree, tree_history, open_list_nodes, open_list_weights, model, gamma=gamma)
        expansions += 1
    end
    # push!(tree_history, (deepcopy(open_list_nodes), -1, -1))

    @show expansions
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
    smallest_node = extract_smallest_node(soltree)
    return(soltree, smallest_node, root)
end


function initialize_tree_bfs(ex, model; max_expansions=1000, max_depth=10)
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
    build_tree_bfs!(soltree, soltree1, open_list, close_list, model, max_expansions=max_expansions, max_depth=max_depth)
    smallest_node = extract_smallest_node(soltree)
    return(soltree, smallest_node, root, soltree1)
end


function initialize_tree_search_tree(ex, model; max_expansions=1000, max_depth=10, epsilon=1.0, gamma=1)
    if isa(model, Function)
        open_list = PriorityQueue{Node, Float32}(Base.Order.Reverse)
    elseif size(model.heuristic.layers[end].weight)[1] == 2
        open_list = PriorityQueue{Node, Tuple{Float32, Float32}}(Base.Order.Reverse)
    else
        open_list = PriorityQueue{Node, Float32}(Base.Order.Reverse)
    end
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
    all_trees_history = Vector{Vector{UInt64}}()
    push!(all_trees_history, [root.node_id])
    build_tree_search_tree!(soltree, soltree1, open_list, close_list, model, all_trees_history, max_expansions=max_expansions, max_depth=max_depth, epsilon=epsilon, gamma=gamma)
    smallest_node = extract_smallest_node(soltree)
    return(soltree, smallest_node, root, all_trees_history)
end


function initialize_tree_search_epsilon(ex, model; max_expansions=1000, max_depth=10, epsilon=1.0, gamma=1.0)
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
    @timeit TO "build tree epsilon" build_tree_epsilon_greedy!(soltree, soltree1, open_list, close_list, model; max_expansions=max_expansions, max_depth=max_depth, epsilon=epsilon, gamma=gamma)
    @timeit TO "extract smallest node" smallest_node = extract_smallest_node(soltree)
    return(soltree, smallest_node, root, soltree1)
end


function initialize_policy_tree_search(ex, model; max_expansions=1000, max_depth=10, gamma=1.0)
    # open_list = PriorityQueue{Node, Float32}(Base.Order.Reverse) 
    
    close_list = Set{UInt64}()

    soltree = Dict{UInt64, Node}()
    # soltree1 = Dict{UInt64, Node}()
    root = Node(ex, (), hash(ex), 0)
    # root1 = Node1(ex, (), hash(ex, hash(0)), 0)
    # root1 = Node(ex, (), UInt64[], hash(ex, hash(ex)), 0, hash(ex, hash(ex)))
    soltree[root.node_id] = root
    # soltree1[root1.node_id] = root1
    o = only(model(root.ex))
    # enqueue!(open_list, root, o)
    tree_history = Vector{Tuple}()
    open_list_nodes = UInt64[root.node_id]
    open_list_weights = Float32[o]
    @timeit TO "build tree policy" build_tree_policy!(soltree, tree_history, open_list_nodes, open_list_weights, close_list, model; max_expansions=max_expansions, max_depth=max_depth, gamma=gamma)
    @timeit TO "extract smallest node" smallest_node = extract_smallest_node(soltree)
    return(soltree, smallest_node, root, tree_history)
end