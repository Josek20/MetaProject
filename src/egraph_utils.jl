# all_terms = [:+, :-, :*, :/]
term_encoding = unique(all_symbols) .== permutedims(all_symbols)
term_encoding_map = Dict(all_symbols[x] => term_encoding[x, :] for x in 1:length(all_symbols))

encode_node(node::EClass) = vcat([1, 0], zeros(length(all_symbols)))
encode_node(node::ENodeTerm) = vcat([0, 0], term_encoding_map[node.operation])
encode_node(node::ENodeLiteral) = vcat([0, 1], zeros(length(all_symbols)))

function encode_graph(g::EGraph)
    eclasses_encoding = [encode_node(eclass) for (eclass_id, eclass) in g.classes] 
    enodes_encoding = [encode_node(enode) for (eclass_id, eclass) in g.classes for enode in eclass] 
    egraph_encoding = [eclasses_encoding; enodes_encoding]
    return Matrix(transpose(hcat(egraph_encoding...)))
end

get_number_of_enodes(g::EGraph) = length(unique(keys(g.memo))) 

function extract_adjacency_matrix1(g::EGraph)
    @assert length(keys(g.memo)) == length(unique(keys(g.memo))) 
    num_nodes = get_number_of_enodes(g)
    adj_matrix = zeros(Int, num_nodes, num_nodes)
    for (k, v) in g.memo
        if k isa ENodeTerm
            for j in k.args
                adj_matrix[v, j] = 1
            end
        end
    end
    return adj_matrix
end


function transform_egraph2graph(g::EGraph)
    
    g = GNNGraph(source, target)
    return g
end


function extract_adjacency_matrix(g::EGraph)
    num_nodes = get_number_of_enodes(g)
    # first classes than nodes
    counter = 1
    adj_matrix = zeros(Int, num_nodes + g.numclasses, num_nodes + g.numclasses)
    ind_to_class_id = Dict(zip(keys(g.classes), collect(1:length(g.classes))))
    #println(g.classes)
    for (enum_ind, (eclass_id, eclass)) in enumerate(g.classes)
        for enode in eclass 
            adj_matrix[g.numclasses + counter, enum_ind] += 1
            if typeof(enode) == ENodeTerm
                for adjacent_class in enode.args
                    adj_matrix[g.numclasses + counter, ind_to_class_id[adjacent_class]] += 1
                end
            end
            counter += 1
        end
    end
    return adj_matrix
end


function update_terms(new_terms::Vector{Symbol})
    global all_terms
    all_terms = new_terms
    global term_encoding_map
    term_encoding = unique(all_terms) .== permutedims(all_terms)
    term_encoding_map = Dict(all_terms[x] => term_encoding[x, :] for x in 1:length(all_terms))
end


function get_enode_to_index_mapping(g::EGraph)
    all_enodes = [enode for (eclass_id, eclass) in g.classes for enode in eclass]
    node_to_id_mapping = Dict(zip(vcat(values(g.classes)..., all_enodes), 1:length(all_enodes)+g.numclasses))
    return node_to_id_mapping
end


function transform_egraph_to_gnn_graph(adjacency_matrix::Matrix)
    s = []
    t = []
    n = size(adjacency_matrix, 1)
    for i in 1:n
        for j in 1:n
            if adjacency_matrix[i, j] == 1
                push!(s, i)
                push!(t, j)
            end
            if adjacency_matrix[i, j] == 2
                push!(s, i)
                push!(t, j)
                push!(s, i)
                push!(t, j)
            end
        end
    end
    s = convert.(Int, s)
    t = convert.(Int, t)
    gnn_graph = GNNGraph(s, t)
    return gnn_graph
end

# export get_number_of_enodes