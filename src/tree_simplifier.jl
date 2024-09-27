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


exp_size(node::Node) = exp_size(node.ex)
exp_size(ex::Expr) = sum(exp_size.(ex.args))
exp_size(ex::Symbol) = 1f0
exp_size(ex::Int) = 1f0
exp_size(ex::Float64) = 1f0
exp_size(ex::Float32) = 1f0


function my_rewriter!(position, ex, rule)
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


function traverse_expr!(ex, matchers, tree_ind, trav_indexs, tmp)
    if !isa(ex, Expr)
        return
    end
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
        traverse_expr!(i, matchers, tree_ind, trav_indexs, tmp)

        # After recursion, pop the last index to backtrack to the correct level
        pop!(trav_indexs)
    end
end


function execute(ex, theory)
    res = []
    tmp = []
    traverse_expr!(ex, theory, 1, Int32[], tmp) 
    for (pl, r) in tmp
        old_ex = copy(ex)
        if isempty(pl)
            push!(res, ((pl, r), theory[r](old_ex)))
        else
            my_rewriter!(pl, old_ex, theory[r])
            push!(res, ((pl, r), old_ex))
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
function expand_node!(parent::Node, soltree, heuristic, open_list, encodings_buffer, all_symbols, symbols_to_index, theory, variable_names) 
    ex = parent.ex
    succesors = execute(ex, theory)
    new_nodes = map(x-> Node(x[2], x[1], parent.node_id, parent.depth + 1, nothing), succesors)
   
    filtered_new_nodes = filter(x-> push_to_tree!(soltree, x), new_nodes)
    # @show length(filtered_new_nodes)
    isempty(filtered_new_nodes) && return(false)
    exprs = map(x->x.ex, filtered_new_nodes)
    # embeded_exprs = map(x-> ex2mill(x, symbols_to_index, all_symbols, variable_names), exprs)
    embeded_exprs = MyModule.multiple_fast_ex2mill(exprs, sym_enc)
    ds = reduce(catobs, embeded_exprs)
    o = heuristic(ds)
    # o = fill(3, length(embeded_exprs))
    for (v,n,e) in zip(o, filtered_new_nodes, embeded_exprs)
        soltree[n.node_id].expression_encoding = e
        n.expression_encoding = e
        enqueue!(open_list, n, v)
    end
    nodes_ids = map(x->x.node_id, filtered_new_nodes)
    append!(parent.children, nodes_ids)
    if in(:1, exprs)
        return true
    end
    return false
end


# function expand_node!(parents::Vector, soltree, heuristic, open_list, encodings_buffer, all_symbols, symbols_to_index, theory, variable_names) 
#     new_nodes = map(parents) do parent
#         succesors = execute(parent.ex, theory)
#         new_nodes = map(x-> Node(x[2], x[1], parent.node_id, parent.depth + 1, nothing), succesors)
#     end

#     new_nodes = vcat(new_nodes...)
#     filtered_new_nodes = filter(x-> push_to_tree!(soltree, x), new_nodes)
#     # @show length(filtered_new_nodes)
#     isempty(filtered_new_nodes) && return(false)
#     exprs = map(x->x.ex, filtered_new_nodes)
#     # embeded_exprs = map(x-> ex2mill(x, symbols_to_index, all_symbols, variable_names), exprs)
#     embeded_exprs = MyModule.multiple_fast_ex2mill(exprs, sym_enc)
#     ds = reduce(catobs, embeded_exprs)
#     o = heuristic(ds)
#     # o = fill(3, length(embeded_exprs))
#     for (v,n,e) in zip(o, filtered_new_nodes, embeded_exprs)
#         soltree[n.node_id].expression_encoding = e
#         n.expression_encoding = e
#         enqueue!(open_list, n, v)
#     end
#     nodes_ids = map(x->x.node_id, filtered_new_nodes)
#     map(parents) do parent
#         append!(parent.children, nodes_ids)
#     end
#     if in(:1, exprs)
#         return true
#     end
#     return false
# end


function build_tree!(soltree::Dict{UInt64, Node}, heuristic::ExprModel, open_list::PriorityQueue, close_list::Set{UInt64}, encodings_buffer::Dict{UInt64, ProductNode}, all_symbols::Vector{Symbol}, symbols_to_index::Dict{Symbol, Int64}, max_steps, max_depth, expansion_history, theory, variable_names)
    step = 0
    reached_goal = false
    epsilon = 0.2
    expand_n = 30
    while length(open_list) > 0
        if max_steps <= step
            break
        end
        # nodes = Node[]
        # if length(open_list) == 1
        nodes, prob = dequeue_pair!(open_list)
        step += 1
        # else
        # n = expand_n > length(open_list) ? length(open_list) : expand_n
        # nodes = map(x->dequeue_pair!(open_list)[1], 1:n)
        # step += n
        #     # for _ in 1:n
        #     #     # if rand() < epsilon && length(open_list) > 1
        #     #     #     keys_list = collect(keys(open_list))
        #     #     #     node = rand(keys_list)
        #     #     #     prob = open_list[node]
        #     #     #     delete!(open_list, node)
        #     #     # else
        #     #     node, prob = dequeue_pair!(open_list)
        #     #     # end
        #     #     # expansion_history[node.node_id] = [step, prob]
        #     #     push!(nodes, node)
        #     #     step += 1
        #     # end
        # end
        # if node.depth >= max_depth
        #     continue
        # end
        # @show nodes
        reached_goal = expand_node!(nodes, soltree, heuristic, open_list, encodings_buffer, all_symbols, symbols_to_index, theory, variable_names)

        if reached_goal
            return true 
        end
        # push!(close_list, node.node_id)
    end
    
    return false
end


function extract_smallest_node_proof(node::Node, soltree::Dict{UInt64, Node}, rules_in_proof=[])
    if node.parent == node.node_id
        return reverse(rules_in_proof)
    end
    push!(nodes_in_proof, node.rule_index)
    return extract_rules_applied1(soltree[node.parent], soltree, rules_in_proof)
end


function extract_training_data(node, soltree, in_proof, training_data=[])
    if node.parent == node.node_id
        return reduce(catobs, training_data)
    end
    children_list = soltree[node.parent].children
    children_encoding = map(x->soltree[x].expression_encoding, children_list)
    append!(training_data, children_encoding)
    return extract_training_data(soltree[node.parent], soltree, training_data)
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
    big_vector = Vector()
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
        push!(big_vector, node.expression_encoding)
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
            push!(big_vector, soltree[children_id].expression_encoding)
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

    ds = nothing
    if !isempty(big_vector)
        ds = reduce(catobs, big_vector) 
    end
    #hp = findall(x-> x == 1, hp)

    return reverse(proof_vector), depth_dict, ds, hp, hn, node_proof_vector
end


function extract_smallest_terminal_node1(soltree::Dict{UInt64, Node}, close_list::Set{UInt64})
    all_nodes = values(soltree)
    tmp1 = exp_size.(all_nodes)
    tmp2 = minimum(tmp1)
    tmp3 = findall(x-> x == tmp2, tmp1)
    if length(tmp3) == 1
        return only(collect(all_nodes)[tmp3])
    end
    filtered_nodes = collect(all_nodes)[tmp3]
    all_depth = map(x->x.depth, filtered_nodes)
    tmp4 = minimum(all_depth)
    tmp5 = findfirst(x->x.depth == tmp4, filtered_nodes)
    return only(filtered_nodes[tmp5])
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


function heuristic_forward_pass(heuristic, ex::Expr, max_steps, max_depth, all_symbols, theory, variable_names)
    soltree = Dict{UInt64, Node}()
    open_list = PriorityQueue{Node, Float32}()
    close_list = Set{UInt64}()
    expansion_history = Dict{UInt64, Vector}()
    encodings_buffer = Dict{UInt64, ProductNode}()
    @show ex
    # encoded_ex = ex2mill(ex, symbols_to_index, all_symbols, variable_names)
    encoded_ex = MyModule.single_fast_ex2mill(ex, sym_enc)
    root = Node(ex, (0,0), hash(ex), 0, encoded_ex)
    soltree[root.node_id] = root
    enqueue!(open_list, root, only(heuristic(root.expression_encoding)))

    reached_goal = build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names)
    println("Have successfuly finished bulding simplification tree!")

    smallest_node = extract_smallest_terminal_node(soltree, close_list)
    simplified_expression = smallest_node.ex
    @show simplified_expression
    proof_vector, depth_dict, big_vector, hp, hn, node_proof_vector = extract_rules_applied(smallest_node, soltree)
    @show proof_vector
    return simplified_expression, depth_dict, big_vector, length(open_list) == 0 || reached_goal, hp, hn, root, proof_vector
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


function train_heuristic!(heuristic, data, training_samples, max_steps, max_depth, all_symbols, theory, variable_names)  
    for (index, i) in enumerate(data)
        println("Index: $index")
        # if length(training_samples) > index && training_samples[index].saturated
        #     continue
        # end
        simplified_expression, depth_dict, big_vector, saturated, hp, hn, root, proof_vector = heuristic_forward_pass(heuristic, i, max_steps, max_depth, all_symbols, theory, variable_names)
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


function apply_proof_to_expr(ex, proof_vector, theory)
    println("Root: $ex")
    for rule_index in proof_vector
        rule_applied = theory[rule_index]
        ex = collect(execute(ex, [rule_applied]))[1]
        println("$ex")
    end
end


function plot_reduction_stats(df, epochs, n_eq)
    # a = combine(groupby(df, "Epoch"), "Length Reduced"  => mean => "Average Reduction")

    plot() 
    for eq_id in 1:n_eq
        eq_data = df[df[!,"Id"] .== eq_id, :]
        plot!(eq_data[!, "Epoch"], eq_data[!, "Length Reduced"],label="Equation $eq_id" )
    end
    xlabel!("Epoch")
    ylabel!("Length Reduction")
    title!("Neural Network Simplification Performance")
    # legend(:topright)
    display(plot())
end
