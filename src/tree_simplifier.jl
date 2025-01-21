mutable struct TreeSeachConfig 
    soltree
    open_list
    second_open_list
    third_open_list
    close_list
    expansion_history
    encodings_buffer
end


mutable struct ExprWithHash
    ex::Union{Expr, Symbol, Number}
	head::Union{Symbol, Number}
	args::Vector
    # left::ExprWithHash
    # right::ExprWithHash
	hash::UInt
end


@my_cache function ExprWithHash(ex::Expr)
    # hashed_ex = hash(ex)
    # get!(expr_cache, hashed_ex) do
    if length(ex.args) >= 3
        head = ex.args[1]
        args = map(x->ExprWithHash(x), ex.args[2:end])
    else
        if ex.head == :call
            head = ex.args[1]
            args = [ExprWithHash(ex.args[end])]
        else
            head = ex.head
            args = map(x->ExprWithHash(x), ex.args)
        end
    end
    h = hash(hash(head), hash(args))
    ExprWithHash(ex, head, args, h)
    # end
end


@my_cache function ExprWithHash(ex::Symbol)
    # hashed_ex = hash(ex)
    # get!(expr_cache, hashed_ex) do
    args = []
    head = ex
    h = hash(ex)
    ExprWithHash(ex, head, args, h)
    # end 
end


@my_cache function ExprWithHash(ex::Number)
    # hashed_ex = hash(ex)
    # get!(expr_cache, hashed_ex) do
    args = []
    head = ex
    h = hash(ex)
    ExprWithHash(ex, head, args, h)
    # end
end


function reconstruct(ex::ExprWithHash, all_symbols, reconstruct_cache)
    if isempty(ex.args)
        return ex.head
    end
    get!(reconstruct_cache, ex) do
        if ex.head in [:&&, :||]
            expression = Expr(ex.head, [reconstruct(x, all_symbols, reconstruct_cache) for x in ex.args]...)
        else
            expression = Expr(:call, ex.head, [reconstruct(x, all_symbols, reconstruct_cache) for x in ex.args]...)
        end
        expression
    end
end


mutable struct Node{E, P}
    ex::E
    rule_index::Tuple
    children::Vector{UInt64}
    parent::P
    depth::Int64
    node_id::UInt64
    expression_encoding::Union{ProductNode, Nothing}
end


Node(ex::Int, rule_index, parent, depth, ee::Nothing) = Node(ex, rule_index, UInt64[],  parent, depth, hash(ex), ee)
# Node(ex, rule_index, parent, depth, ee::Nothing) = Node(ex, rule_index, UInt64[],  parent, depth, hash(ex), ee)
Node(ex::Expr, rule_index::Tuple, parent, depth::Int, ee::Nothing) = Node(ex, rule_index, UInt64[],  parent, depth, hash(ex), ee)
Node(ex::Expr, rule_index::Tuple, parent::UInt64, depth::Int64, ee::ProductNode) = Node(ex, rule_index, UInt64[],  parent, depth, hash(ex), ee)
function Node(ex::Union{Expr,Int}, rule_index, parent, depth)
    ex_hash = ExprWithHash(ex)
    Node(ex_hash, rule_index, UInt64[],  parent, depth, ex_hash.hash, nothing)
end


Base.hash(e::ExprWithHash) = e.hash
Base.hash(e::ExprWithHash, h::UInt) = hash(e.hash, h)
function Base.:(==)(e1::ExprWithHash, e2::ExprWithHash) 
    e1.hash != e2.hash && return(false)
    e1.head != e2.head && return(false)
    length(e1.args) != length(e2.args) && return(false)
    all(i == j for (i,j) in zip(e1.args, e2.args))
end


# exp_size(node::Node) = exp_size(node.ex)
# exp_size(ex::Expr) = sum(exp_size(a) for a in ex.args)
# exp_size(ex::Symbol) = 1f0
# exp_size(ex::Int) = 1f0
# exp_size(ex::Float64) = 1f0
# exp_size(ex::Float32) = 1f0


function show_proof(init_ex, proof)
    ex = init_ex
    for i in proof
        position, rule = i
        new_ex = my_rewriter!(position, ex, theory[rule])
        println("$(ex)->$(theory[rule])->$(new_ex)")
        ex = new_ex
    end
end


function exp_size1(ex::ExprWithHash, size_cache)
    get!(size_cache, ex) do
        return istree(ex.ex) ? reduce(+, exp_size1(x, size_cache) for x in arguments(ex.ex); init=0) + 1 : 1
    end
end


function exp_size1(ex, size_cache)
    get!(size_cache, ex) do
        return istree(ex) ? reduce(+, exp_size1(x, size_cache) for x in arguments(ex); init=0) + 1 : 1
    end
end


@my_cache function exp_size(ex::ExprWithHash)
    # get!(size_cache, ex) do
    if isempty(ex.args)
        return 1f0
    end
    return sum(exp_size(a) for a in ex.args) + 1
    # end
end


@my_cache function exp_size(ex::Union{Expr, Symbol, Int})
    # get!(size_cache, ex) do
    if isa(ex, Symbol) || isa(ex, Number)
        return 1f0
    end
    res = sum(exp_size(a) for a in ex.args)
    return ex.head in [:&&, :||] ? res + 1 : res
    # end
end


function my_rewriter!(position::Vector{Int}, ex::Expr, rule)
    if isempty(position)
        return rule(ex) 
    end
    ind = position[1]
    ret = my_rewriter!(position[2:end], ex.args[ind], rule)
    if !isnothing(ret)
        ex.args[ind] = ret
    end
    return nothing
end


function old_traverse_expr1!(ex::ExprWithHash, matchers::Vector, tree_ind::Int, trav_indexs::Vector{Int}, tmp::Vector{Tuple{Vector{Int}, Int}}, caching)
    if !isa(ex.ex, Expr)
        return
    end
    if haskey(caching, ex)
        b = copy(trav_indexs)
        append!(tmp, [(b, i) for i in caching[ex]])
        c = length(ex.ex.args)
        m = length(ex.args)
        for (ind, i) in enumerate(ex.args)
            # Update the traversal index path with the current index
            ind = c >= 3 ? ind + 1 : ind
            ind = m == 1 ? ind + 1 : ind
            push!(trav_indexs, ind)
            # Recursively traverse the sub-expression
            old_traverse_expr!(i, matchers, tree_ind + 1, trav_indexs, tmp, caching)
            # After recursion, pop the last index to backtrack to the correct level
            pop!(trav_indexs)
        end
    end
    get!(caching, ex) do
        a = filter(em->!isnothing(em[2]), collect(enumerate(rt(ex.ex) for rt in matchers)))
        if !isempty(a)
            b = copy(trav_indexs)
            append!(tmp, [(b, i[1]) for i in a])
        end

        # Traverse sub-expressions
        c = length(ex.ex.args)
        m = length(ex.args)
        for (ind, i) in enumerate(ex.args)
            # Update the traversal index path with the current index
            ind = c >= 3 ? ind + 1 : ind
            ind = m == 1 ? ind + 1 : ind
            push!(trav_indexs, ind)
            # Recursively traverse the sub-expression
            old_traverse_expr!(i, matchers, tree_ind + 1, trav_indexs, tmp, caching)
            # After recursion, pop the last index to backtrack to the correct level
            pop!(trav_indexs)
        end
        return isempty(a) ? [] : [i[1] for i in a]
    end
end


function old_traverse_expr!(ex::ExprWithHash, matchers::Vector, tree_ind::Int, trav_indexs::Vector{Int}, tmp::Vector{Tuple{Vector{Int}, Int}}, caching)
    if !isa(ex.ex, Expr)
        return
    end
    if haskey(caching, ex)
        b = copy(trav_indexs)
        append!(tmp, [(b, i) for i in caching[ex]])
        c = length(ex.ex.args)
        m = length(ex.args)
        for (ind, i) in enumerate(ex.args)
            # Update the traversal index path with the current index
            ind = c >= 3 ? ind + 1 : ind
            ind = m == 1 ? ind + 1 : ind
            push!(trav_indexs, ind)
            # Recursively traverse the sub-expression
            old_traverse_expr!(i, matchers, tree_ind + 1, trav_indexs, tmp, caching)
            # After recursion, pop the last index to backtrack to the correct level
            pop!(trav_indexs)
        end
    end
    get!(caching, ex) do
        a = filter(em->!isnothing(em[2]), collect(enumerate(rt(ex.ex) for rt in matchers)))
        if !isempty(a)
            b = copy(trav_indexs)
            append!(tmp, [(b, i[1]) for i in a])
        end

        # Traverse sub-expressions
        c = length(ex.ex.args)
        m = length(ex.args)
        for (ind, i) in enumerate(ex.args)
            # Update the traversal index path with the current index
            ind = c >= 3 ? ind + 1 : ind
            ind = m == 1 ? ind + 1 : ind
            push!(trav_indexs, ind)
            # Recursively traverse the sub-expression
            old_traverse_expr!(i, matchers, tree_ind + 1, trav_indexs, tmp, caching)
            # After recursion, pop the last index to backtrack to the correct level
            pop!(trav_indexs)
        end
        return isempty(a) ? [] : [i[1] for i in a]
    end
end


function old_traverse_expr!(ex::Union{Expr,Symbol,Number}, matchers::Vector, tree_ind::Int, trav_indexs::Vector{Int}, tmp::Vector{Tuple{Vector{Int}, Int}}, caching::LRU{Expr, Vector})
    if !isa(ex, Expr)
        return
    end
    if haskey(caching, ex)
        b = copy(trav_indexs)
        append!(tmp, [(b, i) for i in caching[ex]])
        for (ind, i) in enumerate(ex.args)
            # Update the traversal index path with the current index
            push!(trav_indexs, ind)

            # Recursively traverse the sub-expression
            old_traverse_expr!(i, matchers, tree_ind, trav_indexs, tmp, caching)

            # After recursion, pop the last index to backtrack to the correct level
            pop!(trav_indexs)
        end
    end
    get!(caching, ex) do
        a = filter(em->!isnothing(em[2]), collect(enumerate(rt(ex) for rt in matchers)))
        if !isempty(a)
            b = copy(trav_indexs)
            append!(tmp, [(b, i[1]) for i in a])
        end

        # Traverse sub-expressions
        for (ind, i) in enumerate(ex.args)
            # Update the traversal index path with the current index
            push!(trav_indexs, ind)

            # Recursively traverse the sub-expression
            old_traverse_expr!(i, matchers, tree_ind, trav_indexs, tmp, caching)

            # After recursion, pop the last index to backtrack to the correct level
            pop!(trav_indexs)
        end
        return isempty(a) ? [] : [i[1] for i in a]
    end
end


function execute(ex::ExprWithHash, theory::Vector, caching)
    res = []
    tmp = Tuple{Vector{Int}, Int}[]
    old_traverse_expr!(ex, theory, 1, Int64[], tmp, caching)
    # old_exs = [ex.ex for _ in 1:length(tmp)]
    # @show ex.ex
    for (ind,(pl, r)) in enumerate(tmp)
        old_ex = copy(ex.ex)
        o = my_rewriter!(pl, old_ex, theory[r])
        if isnothing(o)
            push!(res, ((pl, r), old_ex))
        else
            push!(res, ((pl, r), o))
        end
    end
    return res
end


function execute(ex::Union{Expr, Number}, theory::Vector, caching::Dict{Expr, Vector}) 
    res = []
    tmp = Tuple{Vector{Int}, Int}[]
    old_traverse_expr!(ex, theory, 1, Int64[], tmp, caching) 
    for (pl, r) in tmp
        old_ex = copy(ex)
        o = my_rewriter!(pl, old_ex, theory[r])
        if isnothing(o)
            push!(res, ((pl, r), old_ex))
        else
            push!(res, ((pl, r), o))
        end
    end
    return res
end


function execute(ex::Union{Expr, Number}, theory::Vector, caching::LRU{Expr, Vector}) 
    res = []
    tmp = Tuple{Vector{Int}, Int}[]
    old_traverse_expr!(ex, theory, 1, Int64[], tmp, caching) 
    for (pl, r) in tmp
        old_ex = copy(ex)
        o = my_rewriter!(pl, old_ex, theory[r])
        if isnothing(o)
            push!(res, ((pl, r), old_ex))
        else
            push!(res, ((pl, r), o))
        end
    end
    return res
end


function batched_execute(vectored_ex::Vector, theory, caching)
    res = []
    all_tmp = Dict()
    for ex in vectored_ex
        tmp = Tuple{Vector{Int}, Int}[]
        old_traverse_expr!(ex, theory, 1, Int64[], tmp, caching) 
        all_tmp[ex] = tmp
    end
    for (k,v) in all_tmp
        for (pl, r) in v
            old_ex = copy(k)
            o = my_rewriter!(pl, old_ex, theory[r])
            if isnothing(o)
                push!(res, ((pl, r), old_ex))
            else
                push!(res, ((pl, r), o))
            end
        end
    end
    return res
end 


function push_to_tree!(soltree::Dict, new_node::Node)
    node_id = new_node.node_id
    if haskey(soltree, node_id)
        old_node = soltree[node_id]
        if new_node.depth < old_node.depth
            soltree[node_id].depth = new_node.depth
            push!(soltree[new_node.parent].children, new_node.node_id)
            filter!(x->x!=old_node.node_id, soltree[old_node.parent].children) 
            soltree[node_id].parent = new_node.parent
        # else
        #     soltree[node_id] = old_node
        end
        return (false)
    else
        soltree[node_id] = new_node
        return (true)
    end
end


function expand_node2!(parent::Node, soltree, heuristic, open_list, encodings_buffer, all_symbols, symbols_to_index, theory, variable_names, cache, exp_cache, size_cache, expr_cache, alpha=0.9, lambda=-100.4) 
    ex = parent.ex
    succesors = execute(ex, theory, exp_cache)
    new_nodes = map(x-> Node(x[2], x[1], parent.node_id, parent.depth + 1, nothing), succesors)
    # new_nodes = map(x-> Node(x[2], x[1], parent.node_id, parent.depth + 1, expr_cache), succesors)
   
    filtered_new_nodes = filter(x-> push_to_tree!(soltree, x), new_nodes)
    isempty(filtered_new_nodes) && return(false)
    exprs = map(x->x.ex, filtered_new_nodes)
    
    if isempty(open_list)
        o = Flux.softmax([exp_size(e, size_cache) * lambda for e in exprs])
        for (v,n) in zip(o, filtered_new_nodes)
            enqueue!(open_list, n, v)
        end
    else
        open_keys = keys(open_list)
        new_nodes = [exp_size(e, size_cache) * lambda for e in exprs]
        open_nodes = [exp_size(e.ex, size_cache) * lambda for e in open_keys]
        append!(open_nodes, new_nodes)
        o = Flux.softmax(open_nodes)
        for (k, i) in zip(open_keys,o)
            open_list[k] = i
        end
        for (v,n) in zip(o[end - length(new_nodes) + 1:end], filtered_new_nodes)
            enqueue!(open_list, n, v)
        end
    end
    nodes_ids = map(x->x.node_id, filtered_new_nodes)
    append!(parent.children, nodes_ids)
    if in(:1, exprs)
        return true
    end
    return false
end


function expand_node3!(parent::Node, soltree, heuristic, open_list, encodings_buffer, all_symbols, symbols_to_index, theory, variable_names, cache, exp_cache, size_cache, expr_cache, alpha=0.9, lambda=-100.4) 
    ex = parent.ex
    succesors = execute(ex, theory, exp_cache)
    new_nodes = map(x-> Node(x[2], x[1], parent.node_id, parent.depth + 1, expr_cache), succesors)
   
    filtered_new_nodes = filter(x-> push_to_tree!(soltree, x), new_nodes)
    isempty(filtered_new_nodes) && return(false)
    exprs = map(x->x.ex, filtered_new_nodes)
    
    if isempty(open_list)
        o = Flux.softmax([exp_size(e, size_cache) * lambda for e in exprs])
        for (v,n) in zip(o, filtered_new_nodes)
            push!(open_list, (n, v))
        end
    else
        # open_keys = keys(open_list)
        new_nodes = [exp_size(e, size_cache) * lambda for e in exprs]
        open_nodes = [exp_size(e[1].ex, size_cache) * lambda for e in open_list]
        append!(open_nodes, new_nodes)
        o = Flux.softmax(open_nodes)
        # for (k, i) in zip(open_list,o)
        # @show length(open_nodes)
        for i in 1:length(open_list)
            open_list[i] = (open_list[i][1], o[i])
        end
        for (v,n) in zip(o[end - length(new_nodes) + 1:end], filtered_new_nodes)
            push!(open_list, (n, v))
        end
    end
    nodes_ids = map(x->x.node_id, filtered_new_nodes)
    append!(parent.children, nodes_ids)
    if in(:1, exprs)
        return true
    end
    return false
end


function expand_node!(parent::Node, pipeline_config::SearchTreePipelineConfig, tree_config::TreeSeachConfig) 
    ex = parent.ex
    succesors = execute(ex, pipeline_config.theory, pipeline_config.matching_expr_cache)
    if isa(ex, ExprWithHash)
        new_nodes = map(x-> Node(x[2], x[1], parent.node_id, parent.depth + 1), succesors)
    else
        new_nodes = map(x-> Node(x[2], x[1], parent.node_id, parent.depth + 1, nothing), succesors)
    end
    filtered_new_nodes = filter(x-> push_to_tree!(tree_config.soltree, x), new_nodes)
    # @show length(filtered_new_nodes)
    isempty(filtered_new_nodes) && return(false)
    exprs = map(x->x.ex, filtered_new_nodes)
    o = map(x->only(pipeline_config.heuristic(x.ex)), filtered_new_nodes)
    for (v,n) in zip(o, filtered_new_nodes)
        enqueue!(tree_config.open_list, n, v)
    end
    nodes_ids = map(x->x.node_id, filtered_new_nodes)
    append!(parent.children, nodes_ids)
    if in(:1, exprs)
        return true
    end
    return false
end


function expand_node!(parent::Node, soltree, heuristic, open_list, all_symbols, symbols_to_index, theory, exp_cache) 
    ex = parent.ex
    succesors = execute(ex, theory, exp_cache)
    if isa(ex, ExprWithHash)
        new_nodes = map(x-> Node(x[2], x[1], parent.node_id, parent.depth + 1), succesors)
    else
        new_nodes = map(x-> Node(x[2], x[1], parent.node_id, parent.depth + 1, nothing), succesors)
    end
    filtered_new_nodes = filter(x-> push_to_tree!(soltree, x), new_nodes)
    # @show length(filtered_new_nodes)
    isempty(filtered_new_nodes) && return(false)
    exprs = map(x->x.ex, filtered_new_nodes)
    # o = map(x->alpha * exp_size(x.ex, size_cache) + (1 - alpha) * only(heuristic(x.ex, cache)), filtered_new_nodes)
    # o = map(x->exp_size(x.ex), filtered_new_nodes)
    # o = map(x->node_count(x.ex.ex), filtered_new_nodes)
    o = map(x->only(heuristic(x.ex)), filtered_new_nodes)
    # o = heuristic(exprs, cache)
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


function expand_node_from_multiple!(parent::Node, soltree, heuristic, open_list, second_open_list, theory, cache, exp_cache, size_cache, expr_cache, alpha=0.9, lambda=-100.4) 
    ex = parent.ex
    succesors = execute(ex, theory, exp_cache)
    if isa(ex, ExprWithHash)
        new_nodes = map(x-> Node(x[2], x[1], parent.node_id, parent.depth + 1, expr_cache), succesors)
    else
        new_nodes = map(x-> Node(x[2], x[1], parent.node_id, parent.depth + 1, nothing), succesors)
    end
    filtered_new_nodes = filter(x-> push_to_tree!(soltree, x), new_nodes)
    isempty(filtered_new_nodes) && return(false)
    exprs = map(x->x.ex.ex, filtered_new_nodes)
    o1 = map(x->only(heuristic(x.ex)), filtered_new_nodes)
    o2 = map(x->exp_size(x.ex), filtered_new_nodes)
    for (v1,v2,n) in zip(o1, o2, filtered_new_nodes)
        enqueue!(open_list, n, v1)
        enqueue!(second_open_list, n, v2)
    end

    nodes_ids = map(x->x.node_id, filtered_new_nodes)
    append!(parent.children, nodes_ids)
    if in(:1, exprs)
        return true
    end
    return false
end


function expand_multiple_node!(parents::Vector, soltree, heuristic, open_list, encodings_buffer, all_symbols, symbols_to_index, theory, variable_names, cache, exp_cache) 
    new_nodes = map(parents) do parent
        succesors = execute(parent.ex, theory, exp_cache)
        new_nodes = map(x-> Node(x[2], x[1], parent.node_id, parent.depth + 1, nothing), succesors)
    end

    new_nodes = vcat(new_nodes...)
    filtered_new_nodes = filter(x-> push_to_tree!(soltree, x), new_nodes)
    isempty(filtered_new_nodes) && return(false)
    exprs = map(x->x.ex, filtered_new_nodes)
    o = map(x->alpha * x.depth + (1 - alpha) * only(heuristic(x.ex, cache)), filtered_new_nodes)
    for (v,n) in zip(o, filtered_new_nodes)
        enqueue!(open_list, n, v)
    end
    for n in filtered_new_nodes
        push!(soltree[n.parent].children, n.node_id)
    end
    if in(:1, exprs)
        return true
    end
    return false
end


function build_tree_with_reward_function1!(soltree::Dict{UInt64, Node}, heuristic, open_list, close_list::Set{UInt64}, encodings_buffer::Dict{UInt64, ProductNode}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int64}, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, alpha, lambda=-100.4, epsilon = 0.05)
    step = 0
    reached_goal = false
    epsilon = 0.5
    expand_n = 25
    expanded_depth = []
    while length(open_list) > 0
        if max_steps <= step
            break
        end
        i = rand() < epsilon ? rand(1:length(open_list)) : argmin(i->open_list[i][2], 1:length(open_list))
        # i = argmin(i->open_list[i][2], 1:length(open_list))
        nodes, prob = open_list[i]
        open_list[i] = open_list[end]
        pop!(open_list)
        step += 1
        reached_goal = expand_node3!(nodes, soltree, heuristic, open_list, encodings_buffer, all_symbols, symbols_to_index, theory, variable_names, cache, exp_cache, size_cache, expr_cache, alpha, lambda)
        if reached_goal
            return true 
        end
    end
    return false
end


function build_tree_with_reward_function!(soltree::Dict{UInt64, Node}, heuristic, open_list::PriorityQueue, close_list::Set{UInt64}, encodings_buffer::Dict{UInt64, ProductNode}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int64}, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, alpha, lambda=-100.4)
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
        reached_goal = expand_node2!(nodes, soltree, heuristic, open_list, encodings_buffer, all_symbols, symbols_to_index, theory, variable_names, cache, exp_cache, size_cache, expr_cache, alpha, lambda)
        if reached_goal
            return true 
        end
    end
    return false
end


function build_tree_beam_search!(soltree::Dict{UInt64, Node}, heuristic, open_list::PriorityQueue, close_list::Set{UInt64}, encodings_buffer::Dict{UInt64, ProductNode}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int64}, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, alpha)
    step = 0
    reached_goal = false
    epsilon = 0.05
    expand_n = 25
    expanded_nodes = []
    expanded_nodes_heuristic = []
    while length(open_list) > 0
        if max_steps <= step
            break
        end
        if step == 0
            nodes, prob = dequeue_pair!(open_list)
        else
            min_node_index = argmin(expanded_nodes_heuristic)
            nodes = expanded_nodes[min_node_index]
            expanded_nodes = []
            expanded_nodes_heuristic = []
        end
        # nodes, prob = dequeue_pair!(open_list)
        # if nodes.depth >= max_depth
        #     continue
        # end
        step += 1
        reached_goal = expand_node_beam_search!(nodes, soltree, heuristic, expanded_nodes, expanded_nodes_heuristic, encodings_buffer, all_symbols, symbols_to_index, theory, variable_names, cache, exp_cache, size_cache, expr_cache, alpha)
        if reached_goal
            return true 
        end
    end
    return false
end


function build_tree!(pipeline_config::SearchTreePipelineConfig, tree_config::TreeSeachConfig)
    local_step = 0
    reached_goal = false
    while length(tree_config.open_list) > 0
        if pipeline_config.max_steps <= local_step
            break
        end
        nodes, prob = dequeue_pair!(tree_config.open_list)
        local_step += 1
        reached_goal = expand_node!(nodes, pipeline_config, tree_config)
        if reached_goal
            return true 
        end
    end
    return false
end


function build_tree!(soltree::Dict{UInt64, Node}, heuristic, open_list::PriorityQueue, close_list::Set{UInt64}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int64}, max_steps, max_depth, theory, exp_cache)
    step = 0
    reached_goal = false
    while length(open_list) > 0
        if max_steps <= step
            break
        end
        nodes, prob = dequeue_pair!(open_list)
        # if nodes.depth >= max_depth
        #     continue
        # end
        step += 1
        @show @elapsed reached_goal = expand_node!(nodes, soltree, heuristic, open_list, all_symbols, symbols_to_index, theory, exp_cache)
        if reached_goal
            return true 
        end
    end
    return false
end


function build_tree_with_multiple_queues!(soltree::Dict{UInt64, Node}, heuristic, open_list, second_open_list, close_list, max_steps, theory, cache, exp_cache, size_cache, expr_cache, alpha)
    step = 0
    reached_goal = false
    epsilon = 0.5
    expand_n = 25
    expanded_depth = []
    alpha = 0.5
    nodes = first(values(soltree))
    while !isempty(open_list) || !isempty(second_open_list)
        if max_steps <= step
            break
        end 
        selected_queue = rand() < alpha ? open_list : second_open_list
        selected_queue = length(selected_queue) == 0 && selected_queue == open_list ? second_open_list : selected_queue 
        while true
            nodes, prob = dequeue_pair!(selected_queue)
            if !(nodes.node_id in close_list) || length(selected_queue) == 0
                push!(close_list, nodes.node_id)
                break
            end
        end
        step += 1
        reached_goal = expand_node_from_multiple!(nodes, soltree, heuristic, open_list, second_open_list, theory, cache, exp_cache, size_cache, expr_cache, alpha)
        if reached_goal
            return true 
        end
    end
    return false
end


function extract_smallest_node_proof(node::Node, soltree::Dict{UInt64, Node}, rules_in_proof=[])
    if node.parent == node.node_id
        return reverse(rules_in_proof)
    end
    push!(nodes_in_proof, node.rule_index)
    return extract_smallest_node_proof(soltree[node.parent], soltree, rules_in_proof)
end


function extract_training_data(node, soltree, sym_enc=sym_enc)
    training_exp=[]
    hp=Vector[]
    hn=Vector[]
    proof_vector=[]
    extract_training_data!(node, soltree, training_exp, hp, hn, proof_vector)
    hp = reduce(vcat, hp)
    hn = reduce(vcat, hn)
    tdata = MyModule.no_reduce_multiple_fast_ex2mill(training_exp, sym_enc)
    return tdata, hp, hn, reverse(proof_vector), 0
end


"""
    # neighborhood = (1, 2, 3, ∞)
"""
function extract_training_data!(node, soltree, training_exp, hp, hn, proof_vector)
    if isnothing(node.parent)
        n = length(hn)
        a = sum(length, hn)
        hp[1] = fill(only(hp[1]), a)
        for i in 2:n
            a = sum(length, hn[i:n])
            hp[i] = fill(only(hp[i]), a)
            append!(hn, hn[i:n])
        end
        return
    end
    push!(proof_vector, node.rule_index)
    if node.depth == 1
        tmp = vcat([soltree[i].children for i in soltree[node.parent].children if i != node.node_id]...)
        extended_children_list = vcat(soltree[node.parent].children, tmp)
    else
        tmp = [soltree[i].children for i in soltree[soltree[node.parent].parent].children]
        extended_children_list = vcat(tmp...)
    end
    # children_list = soltree[node.parent].children
    l = length(training_exp)
    pos_index = 0
    push!(hn, [])
    push!(hp, [])
    for (ind,i) in enumerate(extended_children_list)
        if node.node_id == i
            pos_index = ind + l
        else
            push!(hn[end], ind + l)
        end
        push!(training_exp, soltree[i].ex)
    end
    push!(hp[end], pos_index)
    extract_training_data!(soltree[node.parent], soltree, training_exp, hp, hn, proof_vector)
end

function extract_smallest_terminal_node1(soltree::Dict{UInt64, Node}, close_list::Set{UInt64}, nbest=5)
    all_nodes = values(soltree)
    all_nodes = filter(x->x.depth!=0, collect(all_nodes))
    tmp1 = exp_size.(all_nodes)
    tmp2 = minimum(tmp1)
    tmp3 = findall(x-> x == tmp2, tmp1)
    if length(tmp3) == 1
        return only(collect(all_nodes)[tmp3])
    end
    filtered_nodes = collect(all_nodes)[tmp3]
    all_depth = map(x->x.depth, filtered_nodes)
    # all_depth = filter(x->x!=0, all_depth)
    tmp4 = minimum(all_depth)
    tmp5 = findall(x->x.depth == tmp4, filtered_nodes)
    if length(tmp5) < nbest
        return filtered_nodes[tmp5[begin:length(tmp5)]]
    end
    return filtered_nodes[tmp5[begin:nbest]]
end


function extract_smallest_terminal_node(soltree::Dict{UInt64, Node}, close_list::Set{UInt64})
    min_node = nothing
    min_size = typemax(Int)
    min_depth = typemax(Int)
    
    for n in values(soltree)
        if n.depth == 0
            continue
        end
        
        size_n = exp_size(n.ex)

        if size_n < min_size
            min_node = n
            min_size = size_n
            min_depth = n.depth
        elseif size_n == min_size
            if n.depth < min_depth
                min_node = n
                min_depth = n.depth
            end
        end
    end
    
    return min_node
end


function initialize_tree_search(heuristic, ex::Expr, max_steps, max_depth, all_symbols, symbols_to_index, theory, exp_cache)
    soltree = Dict{UInt64, Node}()
    open_list = PriorityQueue{Node, Float32}()
    second_open_list = PriorityQueue{Node, Float32}()
    close_list = Set{UInt64}()
    root = Node(ex, (0,0), nothing, 0)
    o = heuristic(root.ex)
    # o = exp_size(root.ex, size_cache)
    # root = Node(ex, (0,0), nothing, 0, nothing)
    soltree[root.node_id] = root
    enqueue!(open_list, root, only(o))
    # push!(second_open_list, (root, exp_size(root.ex, size_cache)))
    reached_goal = build_tree!(soltree, heuristic, open_list, close_list, all_symbols, symbols_to_index, max_steps, max_depth, theory, exp_cache)
    # reached_goal = build_tree_with_multiple_queues!(soltree, heuristic, open_list, second_open_list, close_list, max_steps, theory, cache, exp_cache, size_cache, expr_cache, alpha)
    # println("Have successfuly finished bulding simplification tree!")
    # @show length(soltree)
    smallest_node = extract_smallest_terminal_node(soltree, close_list)
    simplified_expression = smallest_node.ex
    # if isa(simplified_expression, ExprWithHash)
    #     @show simplified_expression.ex
    # else
    #     @show simplified_expression
    # end
    @show length(soltree)
    # big_vector, hp, hn, proof_vector = extract_training_data(smallest_node, soltree)
    # tmp = []
    # # @show proof_vector
    # # @show length(soltree)
    # return simplified_expression, [], big_vector, length(open_list) == 0 || reached_goal, hp, hn, root, proof_vector, tmp
end


function TreeSeachConfig()
    soltree = Dict{UInt64, Node}()
    open_list = PriorityQueue{Node, Float32}()
    second_open_list = PriorityQueue{Node, Float32}()
    third_open_list = PriorityQueue{Node, Float32}()
    close_list = Set{UInt64}()
    expansion_history = Dict{UInt64, Vector}()
    encodings_buffer = Dict{UInt64, ProductNode}()
    return TreeSeachConfig(soltree, open_list, second_open_list, third_open_list, close_list, expansion_history, encodings_buffer)
end


function initialize_and_build_tree(ex::Expr, pipeline_config::SearchTreePipelineConfig)
    tree_search_config = TreeSeachConfig()
    root = Node(ex, (0,0), nothing, 0, pipeline_config.hash_expr_cache)
    o = pipeline_config.heuristic(root.ex, pipeline_config.heuristic_cache)
    tree_search_config.soltree[root.node_id] = root
    enqueue!(tree_search_config.open_list, root, only(o))
    # push!(second_open_list, (root, exp_size(root.ex, size_cache)))
    reached_goal = pipeline_config.build_tree_function(pipeline_config, tree_search_config)
    println("Have successfuly finished bulding simplification tree!")
    @show length(tree_search_config.soltree)
    smallest_node = extract_smallest_terminal_node(tree_search_config.soltree, tree_search_config.close_list)
    simplified_expression = smallest_node.ex
    @show simplified_expression.ex
    big_vector, hp, hn, proof_vector = extract_training_data(smallest_node, tree_search_config.soltree)
    tmp = []
    @show length(proof_vector)
    @show length(tree_search_config.soltree)
    return simplified_expression, [], big_vector, length(tree_search_config.open_list) == 0 || reached_goal, hp, hn, root, proof_vector, tmp
end


mutable struct TrainingSample{D, S, E, P, HP, HN, IE}
    training_data::D
    saturated::S
    expression::E
    proof::P
    hp::HP
    hn::HN
    initial_expr::IE
end


function TrainingSample(ex::Expr)
    return TrainingSample{Union{ProductNode, Nothing}, Bool, ExprWithHash, Vector, Matrix, Matrix, Expr}(nothing, false, ExprWithHash(ex, LRU(maxsize=1000)), [], [], [], ex)
end


function TrainingSample(list_of_ex::Vector)
    hash_expr_cache = LRU(maxsize=10_000)
    return map(ex->TrainingSample{Union{ProductNode, Nothing}, Bool, ExprWithHash, Vector, Matrix, Matrix, Expr}(nothing, false, ExprWithHash(ex, hash_expr_cache), [], zeros(0,0), zeros(0,0), ex), list_of_ex)
end


function isbetter(a::TrainingSample, b::TrainingSample)
    size_a = exp_size(a.expression) 
    size_b = exp_size(b.expression)
    if size_a > size_b
        return true
    elseif size_a == size_b && length(a.proof) > length(b.proof)
        return true
    else
        return false
    end
end


function train_heuristic!(heuristic, data, training_samples, max_steps, max_depth, all_symbols, sym_enc, theory, exp_cache)
    for (index, i) in enumerate(data)
        println("Index: $index")
        # if length(training_samples) > index && training_samples[index].saturated
        #     continue
        # end
        # @show i
        # if isa(i, Int)
        #     continue
        # end
        simplified_expression, depth_dict, big_vector, saturated, hp, hn, root, proof_vector, m_nodes = initialize_tree_search(heuristic, i, max_steps, max_depth, all_symbols, sym_enc, theory, exp_cache)
        println("Saturated: $saturated")
        new_sample = TrainingSample(big_vector, saturated, simplified_expression, proof_vector, hp, hn, i)
        if length(training_samples) >= index 
            # if isa(training_samples[index].expression, ExprWithHash)
            training_samples[index] = isbetter(training_samples[index], new_sample) ? new_sample : training_samples[index]
            # if exp_size(new_sample.ex, size_cache) < exp_size(training_samples[index].ex, size_cache)
            #     append!(training_samples[index].proof, proof_vector)
            #     training_samples[index].ex = new_sample.ex
            # end
            # else
                # training_samples[index] = isbetter(training_samples[index], new_sample) ? new_sample : training_samples[index]
            # end
        else
            push!(training_samples, new_sample)
        end
        @show length(cache), cache.hits, cache.misses, length(exp_cache), exp_cache.hits, exp_cache.misses
        # if isempty(depth_dict)
        #     println("Error")
        #     continue
        # end
    end
    data_len = length(data)
    # @save "data/training_data/training_samplesk$(data_len)_v5.jld2" training_samples
    return training_samples
end

function get_expression_arguments(ex, exp_args_cache)
    if !isa(ex, Expr)
        return ex
    end
    get!(exp_args_cache, ex) do
        tmp = Any[]
        exp_args = ex.head == :call ? ex.args[2:end] : ex.args
        for a in exp_args
            v = get_expression_arguments(a, exp_args_cache)
            if isa(v, Vector)
                append!(tmp, v)
            else
                push!(tmp, v)
            end
            # @show v
        end
        tmp
    end
end


function is_reducible(ex::ExprWithHash, exp_args_cache)
    is_reducible(ex.ex, exp_args_cache)
end


function is_reducible(ex::Union{Expr, Int, Symbol}, exp_args_cache)
    if !isa(ex, Expr)
        return false
    end
    reducible = @theory a b c begin
        a::Number <= b * c::Number --> a <= b * c
        a::Number <= c::Number * b --> a <= c * b
        a * b::Number <= c::Number --> a * b <= c
        a::Number * b <= c::Number --> a * b <= c
    end
    if any(r->!isnothing(r(ex)), reducible)
        return false
    end
    all_args = get_expression_arguments(ex, exp_args_cache)
    if length(all_args) == length(Set(all_args))
        return false
    end
    return true
end


function train_heuristic!(pipeline_config::SearchTreePipelineConfig)  
    data = pipeline_config.training_data
    number_of_irreducible = 0
    number_of_improved = 0
    for (index, sample) in enumerate(data)
        println("Index: $index")
        ex = sample.initial_expr
        @show ex
        @time simplified_expression, depth_dict, big_vector, saturated, hp, hn, root, proof_vector, m_nodes = initialize_and_build_tree(ex, pipeline_config)
        new_sample = TrainingSample(big_vector, saturated, simplified_expression, proof_vector, hp, hn, sample.initial_expr)
        if isbetter(pipeline_config.training_data[index], new_sample,)
            pipeline_config.training_data[index] = new_sample
        #     pipeline_config.training_data[index].training_data = big_vector
        #     pipeline_config.training_data[index].expression = new_sample.expression
        #     pipeline_config.training_data[index].proof = new_sample.proof
        #     pipeline_config.training_data[index].hp = new_sample.hp
        #     pipeline_config.training_data[index].hn = new_sample.hn
            number_of_improved += 1
        end
        if !is_reducible(pipeline_config.training_data[index].expression, pipeline_config.exp_args_cache)
            number_of_irreducible += 1
        end 
    end
    return number_of_irreducible, number_of_improved
end
