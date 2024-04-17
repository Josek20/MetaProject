module EGraphProcessor
    using Metatheory
    export encode_graph, extract_adjacency_matrix, get_number_of_enodes, update_terms

    all_terms = [:+, :-, :*, :/]
    term_encoding = unique(all_terms) .== permutedims(all_terms)
    term_encoding_map = Dict(all_terms[x] => term_encoding[x, :] for x in 1:length(all_terms))

    encode_node(node::EClass) = vcat([1, 0], zeros(length(all_terms)))
    encode_node(node::ENodeTerm) = vcat([0, 0], term_encoding_map[node.operation])
    encode_node(node::ENodeLiteral) = vcat([0, 1], zeros(length(all_terms)))
    
    function encode_graph(g::EGraph)
        eclasses_encoding = [encode_node(eclass) for (eclass_id, eclass) in g.classes] 
        enodes_encoding = [encode_node(enode) for (eclass_id, eclass) in g.classes for enode in eclass] 
        egraph_encoding = [eclasses_encoding; enodes_encoding]
        return Matrix(transpose(hcat(egraph_encoding...)))
    end


    function get_number_of_enodes(g::EGraph)
        return length(unique(keys(g.memo))) 
    end
    

    function extract_adjacency_matrix(g::EGraph)
        num_nodes = get_number_of_enodes(g)
        # first classes than nodes
        counter = 1
        adj_matrix = zeros(Int, num_nodes + g.numclasses, num_nodes + g.numclasses)
        ind_to_class_id = Dict(zip(keys(g.classes), collect(1:length(g.classes))))
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
        node_to_id_mapping = Dict(zip(vcat(keys(g.classes)..., all_enodes), collect(1:length(all_enodes)+g.numclasses)))
        return node_to_id_mapping
    end
end
