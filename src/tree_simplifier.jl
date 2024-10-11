mutable struct ExprWithHash
	head::Union{Symbol, Number}
	args::Vector
	hash::UInt
end


function ExprWithHash(ex::Expr)
	args = length(ex.args) == 3 ? ExprWithHash.(ex.args[2:end]) : ExprWithHash.(ex.args)
	head = ex.head
	h = hash(hash(head), hash(args))
	ExprWithHash(head, args, h)
end


function ExprWithHash(ex::Symbol)
	args = []
	head = ex
	h = hash(ex)
	ExprWithHash(head, args, h) 
end


function ExprWithHash(ex::Number)
	args = []
	head = ex
	h = hash(ex)
	ExprWithHash(head, args, h) 
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


Node(ex::Int, rule_index, parent, depth, ee) = Node(ex, rule_index, UInt64[],  parent, depth, hash(ex), ee)
Node(ex, rule_index, parent, depth, ee) = Node(ex, rule_index, UInt64[],  parent, depth, hash(ex), ee)
Node(ex::Expr, rule_index::Tuple, parent::UInt64, depth::Int64, ee::ProductNode) = Node(ex, rule_index, UInt64[],  parent, depth, hash(ex), ee)

# function Node(ex, rule_index, parent, depth, ee)
#     ex_hash = ExprWithHash(ex)
#     Node(ex_hash, rule_index, UInt64[],  parent, depth, ex_hash.hash, ee)
# end

Base.hash(e::ExprWithHash) = e.hash
Base.hash(e::ExprWithHash, h::UInt) = hash(e.hash, h)
function Base.:(==)(e1::ExprWithHash, e2::ExprWithHash) 
    e1.hash != e2.hash && return(false)
    e1.head != e2.head && return(false)
    length(e1.args) != length(e2.args) && return(false)
    all(i == j for (i,j) in zip(e1.args, a2.args))
end

exp_size(node::Node) = exp_size(node.ex)
exp_size(ex::Expr) = sum(exp_size.(ex.args))
exp_size(ex::Symbol) = 1f0
exp_size(ex::Int) = 1f0
exp_size(ex::Float64) = 1f0
exp_size(ex::Float32) = 1f0


function my_rewriter!(position::Vector{Int}, ex::Expr, rule::AbstractRule)
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


function my_rewriter!(position::Vector{Int}, ex::ExprWithHash, rule::AbstractRule)
    if isempty(position)
        return rule(ex.ex) 
    end
    ind = position[1]
    next_ex = length(ex.ex.args) == 2 ? ex.args[ind] : ex.args[ind - 1] 
    ret = my_rewriter!(position[2:end], next_ex, rule)
    if !isnothing(ret)
        ex.args[ind] = ExprWithHash(ret)
    end
    return nothing
end


function old_traverse_expr!(ex::ExprWithHash, matchers::Vector{AbstractRule}, tree_ind::Int, trav_indexs::Vector{Int}, tmp::Vector{Tuple{Vector{Int}, Int}}, caching::LRU{ExprWithHash, Vector})
    if !isa(ex.ex, Expr)
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
        a = filter(em->!isnothing(em[2]), collect(enumerate(rt(ex.ex) for rt in matchers)))
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



function old_traverse_expr!(ex::Union{Expr,Symbol,Number}, matchers::Vector{AbstractRule}, tree_ind::Int, trav_indexs::Vector{Int}, tmp::Vector{Tuple{Vector{Int}, Int}}, caching::LRU{Expr, Vector})
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


function traverse_expr!(ex::Union{Expr,Symbol,Number}, matchers::Vector{AbstractRule}, tree_ind::Int, trav_indexs::Vector{Int}, tmp::Vector{Tuple{Vector{Int}, Int}}, caching::LRU{Expr, Vector})
    if !isa(ex, Expr)
        return []
    end
    get!(caching, ex) do
        a = filter(em->!isnothing(em[2]), collect(enumerate(rt(ex) for rt in matchers)))
        tmp3 = []
        if !isempty(a)
            b = copy(trav_indexs)
            tmp3 = [(b, i[1]) for i in a]
        end
        for (ind, i) in enumerate(ex.args)
            push!(trav_indexs, ind)
            
            tmp1 = traverse_expr!(i, matchers, tree_ind, trav_indexs, tmp, caching)
            
            # @show t mp3
            append!(tmp3, tmp1)
            pop!(trav_indexs)
        end
        return tmp3
    end
end


function execute(ex::ExprWithHash, theory::Vector{AbstractRule}, caching::LRU{ExprWithHash, Vector}) 
    res = []
    tmp = Tuple{Vector{Int}, Int}[]
    old_traverse_expr!(ex, theory, 1, Int64[], tmp, caching) 
    # @show tmp
    # tmp = caching[ex]
    for (pl, r) in tmp
        old_ex = copy(ex.ex)
        # if isempty(pl)
        #     push!(res, ((pl, r), theory[r](old_ex)))
        # else
            # @show old_ex
        my_rewriter!(pl, old_ex, theory[r])
        push!(res, ((pl, r), old_ex))
        # end
    end
    return res
end


function execute(ex::Union{Expr, Number}, theory::Vector{AbstractRule}, caching::LRU{Expr, Vector}) 
    res = []
    tmp = Tuple{Vector{Int}, Int}[]
    old_traverse_expr!(ex, theory, 1, Int64[], tmp, caching) 
    # @show tmp
    # tmp = unique(caching[ex])
    for (pl, r) in tmp
        old_ex = copy(ex)
        # if isempty(pl)
        #     push!(res, ((pl, r), theory[r](old_ex)))
        # else
        # @show old_ex
        # @show pl, r
        my_rewriter!(pl, old_ex, theory[r])
        push!(res, ((pl, r), old_ex))
        # end
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
        else
            soltree[node_id] = old_node
        end
        return (false)
    else
        soltree[node_id] = new_node
        return (true)
    end
end


# function expand_node_new!(parent::Node, soltree::Dict{UInt64, Node}, heuristic::ExprModel, open_list::PriorityQueue, encodings_buffer::Dict{UInt64, ProductNode}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int64}, theory::Vector, variable_names::Dict) 
function expand_node!(parent::Node, soltree, heuristic, open_list, encodings_buffer, all_symbols, symbols_to_index, theory, variable_names, cache, exp_cache, alpha=0.9) 
    ex = parent.ex
    succesors = execute(ex, theory, exp_cache)
    new_nodes = map(x-> Node(x[2], x[1], parent.node_id, parent.depth + 1, nothing), succesors)
   
    filtered_new_nodes = filter(x-> push_to_tree!(soltree, x), new_nodes)
    # @show length(filtered_new_nodes)
    isempty(filtered_new_nodes) && return(false)
    # exprs = map(x->x.ex.ex, filtered_new_nodes)
    exprs = map(x->x.ex, filtered_new_nodes)
    o = map(x->alpha * x.depth + (1 - alpha) * only(heuristic(x.ex, cache)), filtered_new_nodes)
    # o = map(x->only(embed(heuristic, x)), o)
    # o = map(x->only(embed(heuristic, x[1])) + x[2].depth * 500, zip(o,filtered_new_nodes))
    
    # embeded_exprs = MyModule.no_reduce_multiple_fast_ex2mill(exprs, sym_enc)
    # o = heuristic(embeded_exprs)
    # o = map(x->x.depth, filtered_new_nodes)
    # for (v,n,e) in zip(o, filtered_new_nodes, embeded_exprs)
    #     soltree[n.node_id].expression_encoding = e
    #     n.expression_encoding = e
    #     enqueue!(open_list, n, v)
    # end
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


# function expand_node!(parents::Vector, soltree, heuristic, open_list, encodings_buffer, all_symbols, symbols_to_index, theory, variable_names, cache, exp_cache) 
#     new_nodes = map(parents) do parent
#         succesors = execute(parent.ex, theory, exp_cache)
#         new_nodes = map(x-> Node(x[2], x[1], parent.node_id, parent.depth + 1, nothing), succesors)
#     end

#     new_nodes = vcat(new_nodes...)
#     filtered_new_nodes = filter(x-> push_to_tree!(soltree, x), new_nodes)
#     # @show length(filtered_new_nodes)
#     isempty(filtered_new_nodes) && return(false)
#     exprs = map(x->x.ex, filtered_new_nodes)
#     # o = map(x->MyModule.cached_inference!(x, cache, heuristic, new_all_symbols, sym_enc),exprs)
#     # o = map(x->only(embed(heuristic, x)), o)
#     # embeded_exprs = map(x-> ex2mill(x, symbols_to_index, all_symbols, variable_names), exprs)
#     # embeded_exprs = MyModule.multiple_fast_ex2mill(exprs, sym_enc)
#     # ds = reduce(catobs, embeded_exprs)
#     # o = heuristic(ds)
#     # o = fill(3, length(embeded_exprs))
#     # for (v,n,e) in zip(o, filtered_new_nodes, embeded_exprs)
#     #     soltree[n.node_id].expression_encoding = e
#     #     n.expression_encoding = e
#     #     enqueue!(open_list, n, v)
#     # end
#     for (v,n) in zip(o, filtered_new_nodes)
#         enqueue!(open_list, n, v)
#     end
#     # nodes_ids = map(x->x.node_id, filtered_new_nodes)
#     for n in filtered_new_nodes
#         push!(soltree[n.parent].children, n.node_id)
#     end
#     # map(parents) do parent
#     #     append!(parent.children, )
#     # end
#     if in(:1, exprs)
#         return true
#     end
#     return false
# end


function build_tree!(soltree::Dict{UInt64, Node}, heuristic, open_list::PriorityQueue, close_list::Set{UInt64}, encodings_buffer::Dict{UInt64, ProductNode}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int64}, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, alpha)
    step = 0
    reached_goal = false
    epsilon = 0.05
    expand_n = 25
    expanded_depth = []
    while length(open_list) > 0
        if max_steps <= step
            break
        end
        # nodes = Node[]
        # if length(open_list) == 1
        # @show step
        # @show length(open_list)
        nodes, prob = dequeue_pair!(open_list)
        step += 1
        # push!(expanded_depth, nodes.depth)
        # else
        # n = expand_n > length(open_list) ? length(open_list) : expand_n
        # nodes = map(x->dequeue_pair!(open_list)[1], 1:n)
        # step += n
        #     # for _ in 1:n
        # if rand() < epsilon && length(open_list) > 1
        #     keys_list = keys(open_list)
        #     nodes = rand(keys_list)
        #     prob = open_list[nodes]
        #     # delete!(open_list, nodes)
        # else
        #     nodes, prob = dequeue_pair!(open_list)
        # end
        # step += 1
        # if isin(close_list, nodes.node_id)
        #     continue
        # end
        #     #     # expansion_history[node.node_id] = [step, prob]
        #     #     push!(nodes, node)
        #     #     step += 1
        #     # end
        # end
        # if nodes.depth >= max_depth
        #     continue
        # end
        # @show nodes
        reached_goal = expand_node!(nodes, soltree, heuristic, open_list, encodings_buffer, all_symbols, symbols_to_index, theory, variable_names, cache, exp_cache, alpha)

        if reached_goal
            return true 
        end
        # push!(close_list, nodes.node_id)
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


function extract_training_data(node, soltree)
    training_exp=[]
    hp=Vector[Int16[]]
    hn=Vector[Int16[]]
    proof_vector=[]
    extract_training_data!(node, soltree, training_exp, hp, hn, proof_vector)
    tdata = MyModule.no_reduce_multiple_fast_ex2mill(training_exp, sym_enc)
    max_length = maximum(length, hn)
    padded_vectors = [vcat(vec, fill(0, max_length - length(vec))) for vec in hn]
    hn = hcat(padded_vectors...)   
    for i in size(hn, 2) - 1:-1:1
        hn[:, i] .= hn[:, i] + hn[:, i + 1]
    end
    max_length = maximum(length, hp)
    padded_vectors = [vcat(vec, fill(0, max_length - length(vec))) for vec in hp]
    hp = hcat(padded_vectors...)
    return tdata, hp, hn, reverse(proof_vector)
end


function extract_training_data!(node, soltree, training_exp, hp, hn, proof_vector)
    if node.parent == node.node_id
        return
    end
    push!(proof_vector, node.rule_index)
    children_list = soltree[node.parent].children
    push!(hp,zeros(Int16, length(hp[end])))
    push!(hn,zeros(Int16, length(hn[end])))
    for i in children_list
        if node.node_id == i
            push!(hp[end], 1)
            push!(hn[end], 0)
        else
            push!(hp[end], 0)
            push!(hn[end], 1)
        end
        push!(training_exp, soltree[i].ex)
    end
    extract_training_data!(soltree[node.parent], soltree, training_exp, hp, hn, proof_vector)
end


function extract_training_matrices(node, soltree, hp=[], hn=[])
    if node.parent == node.node_id
        return hp, hn
    end 
    return extract_training_matrices(soltree[node.parent], soltree, hp, hn)
end

 
function extract_rules_applied(node::Node, soltree::Dict{UInt64, Node}) 
    proof_vector = Vector()
    depth_dict = Dict{Int, Vector{Any}}()
    # big_vector = Vector()
    tmp_hp = Vector{Vector{Int16}}()
    tmp_hn = Vector{Vector{Int16}}()
    node_proof_vector = Vector() 
    #while !isnothing(node.parent)
    #nodes_to_use = filter(n->n.depth<node.depth, values(soltree)
    while node.parent != node.node_id
        push!(proof_vector, node.rule_index)
        # push!(node_proof_vector, node.ex)
        push!(node_proof_vector, node.node_id)
        depth_dict[node.depth] = [node.expression_encoding]
        # push!(big_vector, node.expression_encoding)
        push!(tmp_hp, Int16[])
        push!(tmp_hn, Int16[])
        padding = length(tmp_hn) > 1 ? zeros(Int16, length(tmp_hn[end - 1])) : Int16[]
        append!(tmp_hp[end], padding)
        append!(tmp_hn[end], padding)
        push!(tmp_hn[end], 0)
        push!(tmp_hp[end], 1)
        for children_id in soltree[node.parent].children
            if children_id == node.node_id
                continue    
            end
            push!(depth_dict[node.depth], soltree[children_id].expression_encoding)
            # push!(big_vector, soltree[children_id].expression_encoding)
            # push!(node_proof_vector, soltree[children_id].ex)
            push!(node_proof_vector, node.node_id)
            push!(tmp_hp[end], 0)
            push!(tmp_hn[end], 1)
        end
        node = soltree[node.parent]
    end
    padding_value = 0
    hn = nothing
    if !isempty(tmp_hn)
        max_length = maximum(length, tmp_hn)

        padded_vectors = [vcat(vec, fill(padding_value, max_length - length(vec))) for vec in tmp_hn]
        hn = hcat(padded_vectors...)
        # for i in 2:size(hn, 2)
        #     hn[:, i] .= hn[:, i] + hn[:, i - 1]
        # end

        for i in size(hn, 2) - 1:-1:1
            hn[:, i] .= hn[:, i] + hn[:, i + 1]
        end
    end

    hp = nothing
    if !isempty(tmp_hp)
        max_length = maximum(length, tmp_hp)

        padded_vectors = [vcat(vec, fill(padding_value, max_length - length(vec))) for vec in tmp_hp]
        hp = hcat(padded_vectors...)
    end

    # ds = nothing
    # if !isempty(big_vector)
    #     ds = reduce(catobs, big_vector) 
    # end
    #hp = findall(x-> x == 1, hp)

    return reverse(proof_vector), depth_dict, hp, hn, node_proof_vector
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


function initialize_tree_search(heuristic, ex::Expr, max_steps, max_depth, all_symbols, theory, variable_names, cache, exp_cache, alpha)
    soltree = Dict{UInt64, Node}()
    open_list = PriorityQueue{Node, Float32}()
    close_list = Set{UInt64}()
    expansion_history = Dict{UInt64, Vector}()
    encodings_buffer = Dict{UInt64, ProductNode}()
    @show ex
    # encoded_ex = ex2mill(ex, symbols_to_index, all_symbols, variable_names)
    # encoded_ex = MyModule.single_fast_ex2mill(ex, sym_enc)
    # a = MyModule.cached_inference!(ex, cache, heuristic, new_all_symbols, sym_enc)
    # o = MyModule.embed(heuristic, a)
    o = heuristic(ex, cache)
    root = Node(ex, (0,0), hash(ex), 0, nothing)
    soltree[root.node_id] = root
    enqueue!(open_list, root, only(o))

    reached_goal = build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, alpha)
    println("Have successfuly finished bulding simplification tree!")
    @show length(soltree)
    smallest_node = extract_smallest_terminal_node(soltree, close_list)
    # smallest_nodes = extract_smallest_terminal_node1(soltree, close_list)
    simplified_expression = smallest_node.ex
    @show simplified_expression
    # proof_vector, depth_dict, hp, hn, node_proof_vector = extract_rules_applied(smallest_node, soltree)
    # tmp = map(x->extract_rules_applied(x, soltree), smallest_nodes)
    big_vector, hp, hn, proof_vector = extract_training_data(smallest_node, soltree)
    tmp = []
    @show proof_vector
    return simplified_expression, [], big_vector, length(open_list) == 0 || reached_goal, hp, hn, root, proof_vector, tmp
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


function isbetter(a::TrainingSample, b::TrainingSample)
    if exp_size(a.expression) > exp_size(b.expression)
        return true
    elseif exp_size(a.expression) == exp_size(b.expression) && length(a.proof) > length(b.proof)
        return true
    else
        return false
    end
end


function train_heuristic!(heuristic, data, training_samples, max_steps, max_depth, all_symbols, theory, variable_names, cache, exp_cache, alpha)  
    for (index, i) in enumerate(data)
        println("Index: $index")
        # if length(training_samples) > index && training_samples[index].saturated
        #     continue
        # end
        simplified_expression, depth_dict, big_vector, saturated, hp, hn, root, proof_vector, m_nodes = initialize_tree_search(heuristic, i, max_steps, max_depth, all_symbols, theory, variable_names, cache, exp_cache, alpha)
        println("Saturated: $saturated")
        new_sample = TrainingSample(big_vector, saturated, simplified_expression, proof_vector, hp, hn, i)
        if length(training_samples) >= index 
            training_samples[index] = isbetter(training_samples[index], new_sample) ? new_sample : training_samples[index]
        else
            push!(training_samples, new_sample)
        end
        if isempty(depth_dict)
            println("Error")
            continue
        end
    end
    data_len = length(data)
    # @save "data/training_data/training_samplesk$(data_len)_v5.jld2" training_samples
    return training_samples
end
