@my_cache LRU(maxsize=10_000) function exp_size(ex::Union{Expr, Symbol, Number})
    if isa(ex, Symbol) || isa(ex, Number)
        return 1f0
    end
    res = sum(exp_size(a) for a in ex.args)
    return ex.head in [:&&, :||] ? res + 1 : res
end


@my_cache LRU(maxsize=10_000) function exp_size(x::NodeID)
    node = nc[x]
    if node == nullnode
        return 0f0
    elseif !(node.iscall) && node.head ∉ [:&&, :||]
        return 1f0
    end
    return exp_size(node.left) + exp_size(node.right) + 1
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

@my_cache LRU(maxsize=10_000) function matched_expr_cached(ex::Union{Expr,Symbol,Number}, matchers::Vector)
    matches = filter(em -> !isnothing(em[2]), collect(enumerate(rt(ex) for rt in matchers)))
    return [i[1] for i in matches]
end
function new_traverse_expr!(ex::Union{Expr,Symbol,Number}, matchers::Vector, tree_ind::Int, trav_indexs::Vector{Int}, tmp::Vector{Tuple{Vector{Int}, Int}})
    if !isa(ex, Expr)
        return
    end
    match_inds = matched_expr_cached(ex, matchers)

    if !isempty(match_inds)
        b = copy(trav_indexs)
        append!(tmp, [(b, i) for i in match_inds])
    end

    for (ind, arg) in enumerate(ex.args)
        push!(trav_indexs, ind)
        new_traverse_expr!(arg, matchers, tree_ind, trav_indexs, tmp)
        pop!(trav_indexs)
    end
end


function old_traverse_expr!(ex::Union{Expr,Symbol,Number}, matchers::Vector, tree_ind::Int, trav_indexs::Vector{Int}, tmp::Vector{Tuple{Vector{Int}, Int}}, caching::LRU)
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

all_expand(ex::Int, theory) = [], []
function all_expand(ex::Expr, theory)
    res = []
    tmp = Tuple{Vector{Int}, Int}[]
    # old_traverse_expr!(ex, theory, 1, Int64[], tmp, cache) 
    new_traverse_expr!(ex, theory, 1, Int64[], tmp) 
    for (pl, r) in tmp
        old_ex = copy(ex)
        o = my_rewriter!(pl, old_ex, theory[r])
        if isnothing(o)
            push!(res, old_ex)
            
        else
            push!(res, o)
        end
    end
    return res, tmp
end


@my_cache LRU(maxsize=100_000) function all_expand(ex::NodeID, theory)
    node = nc[ex]
    !(node.iscall) && return([], [])
    self = [(ind, r(ex)) for (ind, r) in enumerate(theory)]
    self = [isa(x, Int) ? (ind, intern!(x)) : (ind, x) for (ind, x) in self]

    self = filter(x->!isnothing(x[2]), self)
    pos_indexes = map(x->([], x[1]), self)
    self = map(x->x[2], self)
    lefts, lefts_indexes = all_expand(node.left, theory)
    rights, rights_indexes = all_expand(node.right, theory)
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
    if !isempty(lefts_indexes)
        lefts_indexes = map(x->(vcat([:left], x[1]),x[2]), lefts_indexes)
    end
    if !isempty(rights_indexes)
        rights_indexes = map(x->(vcat([:right], x[1]),x[2]), rights_indexes)
    end
    return vcat(self, childs), vcat(pos_indexes, lefts_indexes, rights_indexes)
end


function extract_proof(node, soltree, nodes_in_proof=[], proof=[])
    if node.parent == node.node_id
        return reverse(nodes_in_proof), reverse(proof)
    end
    push!(nodes_in_proof, node)
    push!(proof, node.rule_index)
    return extract_proof(soltree[node.parent], soltree, nodes_in_proof, proof)
end


function distance_from_proof(soltree, nodes_in_proof)
    d = sizehint!(Dict{UInt64,UInt16}(), length(soltree))
    function length_from_path(node)
        get!(d, node.node_id) do 
            node.parent == node.node_id && return 0 # this is to handle root node
            node ∈ nodes_in_proof ? 0 : length_from_path(soltree[node.parent]) + 1
        end
    end

    foreach(length_from_path, values(soltree))
    return(d)
end


function parent_in_proof(soltree, nodes_in_proof, nodes)
    d = sizehint!(Dict{UInt64,UInt64}(), length(nodes))
    for node in nodes_in_proof
        d[node.node_id] = node.node_id
    end

    function _parent(node)
        get!(d, node) do 
            soltree[node].parent == node && return 0 # this is to handle root node
            soltree[node].parent ∈ nodes_in_proof ? soltree[node].parent : _parent(soltree[node].parent)
        end
    end

    foreach(_parent, keys(nodes))
    return(d)
end


transform_to_expr(exp::Vector{NodeID}) = [expr(nc, n) for n in exp]
transform_to_expr(exp::Vector{Expr}) = exp


function extract_training_data(node, soltree, root; n=1, sym_enc=sym_enc)
    nodes_in_proof, proof = extract_proof(node, soltree)
    nodes_in_proof = vcat(root, nodes_in_proof)
    d2p = distance_from_proof(soltree, nodes_in_proof)
    d2p = filter(v -> v[2] ≤ n, d2p) # ids of nodes of interest
    n2p = parent_in_proof(soltree, nodes_in_proof, d2p)
    @assert 0 ∉ values(n2p)
    proof_node_neighbors = sizehint!(Dict{UInt64, Vector{UInt64}}(), length(nodes_in_proof)) 
    for node in nodes_in_proof
        proof_node_neighbors[node.node_id] = sizehint!(UInt64[], length(n2p))
    end
    for (n, p) in n2p
        n == p && continue
        push!(proof_node_neighbors[p], n)
    end
    max_length = cumsum([length(proof_node_neighbors[n.node_id]) for n in nodes_in_proof])
    hp, hn = sizehint!(Int[], sum(max_length[1:end-1])), sizehint!(Int[], sum(max_length[1:end-1]))
    training_expressions = sizehint!(typeof(node.ex)[], length(d2p))
    for (ind,n) in enumerate(nodes_in_proof[1:end-1])
        id = n.node_id
        if ind == 1
            append!(hp, fill(1, length(proof_node_neighbors[id])))
            append!(hn, 2:length(proof_node_neighbors[id]) + 1)
        else
            new_hn = hn[end] + 2:hn[end] + 1 + length(proof_node_neighbors[id])
            @assert length(new_hn) == length(proof_node_neighbors[id]) 
            prev_inequalities = hn[end - max_length[ind - 1] + 1:end]
            append!(hp, fill(hn[end] + 1, length(proof_node_neighbors[id]) + length(prev_inequalities)))
            append!(hn, prev_inequalities)
            append!(hn, new_hn)
        end
        push!(training_expressions, nodes_in_proof[ind + 1].ex)
        append!(training_expressions, [soltree[i].ex for i in proof_node_neighbors[id]])
    end
    @assert length(hp) == length(hn) == sum(max_length[1:end-1])
    training_expressions = transform_to_expr(training_expressions)
    td = no_reduce_multiple_fast_ex2mill(training_expressions, sym_enc)
    return td, hp, hn, proof, training_expressions
end


function show_proof(initial_expr::NodeID, proof, theory)
    # Todo: !
    ex = initial_expr
    for (pos, rule_index) in proof
        r = theory[rule_index]
        for j in pos
            ex = r(ex)
        end
    end
end


function show_proof(init_ex, proof)
    ex = copy(init_ex)
    for i in proof
        position, rule = i
        println("=====================================================================================================")
        println("Start: $(ex)")
        println("Rule: $(theory[rule])")
        new_ex = my_rewriter!(position, ex, theory[rule])
        if new_ex isa Nothing
            println("Result: $(ex)")
        else
            println("Result: $(new_ex)")
            ex = new_ex
        end
    end
end