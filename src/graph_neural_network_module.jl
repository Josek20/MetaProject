params = SaturationParams()


mutable struct GraphNN{GCN, R}
    GCN::GCN
    R::R
end

Flux.params(l::GraphNN) = Flux.params([l.GCN, l.R])

function my_softmax1(matrix::Matrix{T}) where T
    matrix[matrix .!= 0] = softmax(matrix[matrix .!= 0])
    return Flux.Tangent(matrix)
end



function my_softmax(matrix::Matrix{T}) where T
    # Get indices of non-zero elements
    non_zero_indices = findall(matrix .!= 0)
    
    # Apply softmax to non-zero elements
    softmax_values = softmax(matrix[non_zero_indices])
    
    # Assign softmax values back to the appropriate positions in the matrix
    matrix[non_zero_indices] .= softmax_values
    
    return matrix
end



function GraphNN(in_encoding_size::Integer, out_encoding_size::Integer, number_of_rules::Integer, init=Flux.glorot_uniform)
    return GraphNN(GCNConv(in_encoding_size => out_encoding_size), init(number_of_rules, out_encoding_size))
end


function (m::GraphNN)(g::GNNGraph, x::Matrix, rule_applicability_matrix::Matrix)
    #updated_graph_encoding = m.activation(x * m.W)
    #println("input size $(size(x))")
    #println("rule app size $(size(rule_applicability_matrix))")
    #println("rule app size $(size(m.GCN.weight))")
    updated_graph_encoding = m.GCN(g, transpose(x))

    prod1 = transpose(m.R * updated_graph_encoding)
    enode_to_rule_probability = my_softmax(prod1 .* rule_applicability_matrix)


    #symplified_expression = extract!(g, astsize)
    return enode_to_rule_probability
end


function rule_application_locations(g::EGraph, theory::Vector{<:AbstractRule})
    sched = params.scheduler(g, theory, params.schedulerparams...)
    report = SaturationReport(g)
    number_of_matched_rules = eqsat_search!(g, theory, sched, report)
    return number_of_matched_rules
end


function get_rule_applicability_matrix(g::EGraph, theory::Vector{<:AbstractRule}, node_to_id_mapping::Dict)
    _ = rule_application_locations(g, theory) 
    rule_adjacency = zeros(g.numclasses + get_number_of_enodes(g), size(theory, 1))
    for applicable_rule in g.buffer
        rule_index = 0 
        for (k, j) in applicable_rule
            if k == 0
                rule_index = abs(j[1])
                node_index = node_to_id_mapping[g.classes[j[2]]]
                rule_adjacency[node_index, rule_index] = 1
            elseif j[2] == 0
                node_index = node_to_id_mapping[g.classes[j[1]]]
                rule_adjacency[node_index, rule_index] = 1
            else
                node_index = node_to_id_mapping[g.classes[j[1]][j[2]]]
                rule_adjacency[node_index, rule_index] = 1
            end
        end
    end
    return rule_adjacency
end


function get_rule_prob_dictionary(enode_to_rule_probability::Matrix, g::EGraph, node_to_ind_mapping::Dict)
    rule_prob_dict = Dict()
    for rule in g.buffer
        rule_ind = abs(rule[0][1])  # Calculate rule_ind outside the loop
        rule_prob_dict[rule] = 0
        for (key, value) in rule
            if key == 0
                node_ind = node_to_ind_mapping[g.classes[value[2]]]
            elseif value[2] == 0
                node_ind = node_to_ind_mapping[g.classes[value[1]]]
            else
                node_ind = node_to_ind_mapping[g.classes[value[1]][value[2]]]
            end
            rule_prob_dict[rule] += enode_to_rule_probability[node_ind, rule_ind]
        end
    end
    return rule_prob_dict
end


function extract_and_apply_max_rule(enode_to_rule_probability::Matrix, g::EGraph, theory::Vector{<:AbstractRule}, node_to_ind_mapping::Dict)
    rule_probability_dictionary = get_rule_prob_dictionary(enode_to_rule_probability, g, node_to_ind_mapping)
    max_rule = argmax(rule_probability_dictionary)
    g.buffer = [max_rule]
    report = SaturationReport(g)
    eqsat_apply!(g, theory, report, params)
    rebuild!(g)
    return rule_probability_dictionary, max_rule
end


function loss1(rule_prob_dict::Dict)
    if length(rule_prob_dict) == 1
        return log(1 + exp(1))
    end
    rule_prob_vector = collect(values(sort(rule_prob_dict, rev=true)))
    return sum(log.(1 .+ exp.(rule_prob_vector[begin + 1: end] - rule_prob_vector[begin])))
end


function loss(rule_prob_dict::Dict)
    if length(rule_prob_dict) == 1
        return log(1 + exp(1))
    end
   
    # Extract key-value pairs and sort by value
    sorted_pairs = sort(collect(rule_prob_dict), by=x->x[2], rev=true)
    
    # Extract sorted values
    sorted_values = [pair[2] for pair in sorted_pairs]
    return sum(log.(1 .+ exp.(sorted_values[2:end] .- sorted_values[begin])))
end
