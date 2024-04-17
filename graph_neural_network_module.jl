module GraphNeuralNetwork
    using Flux
    using Metatheory
    using .Metatheory.EGraphs: SaturationReport, eqsat_search!, eqsat_apply!, extract!
    include("egraph_processor.jl")
    using .EGraphProcessor
    export GraphNN, train_model, evaluate_model, extract_and_apply_max_rule, clear_non_max_rules!, rule_application_locations, get_rule_applicability_matrix, extract_max_rule_id, clear_non_max_rules!, my_softmax, loss, rebuild!

    params = SaturationParams()

    mutable struct GraphNN{W, R, F}
        W::W
        R::R
        activation::F
    end

    Flux.params(l::GraphNN) = Flux.params([l.W, l.R])


    function my_softmax(matrix::Matrix{T}) where T
        matrix[matrix .== 0] .= -Inf
        exp_matrix = exp.(matrix)
        exp_sum = sum(exp_matrix, dims=2)
        return exp_matrix ./ exp_sum
    end

    
    function GraphNN(in_encoding_size::Integer, out_encoding_size::Integer, number_of_rules::Integer, activation_function=sigmoid, init=Flux.glorot_uniform)
        return GraphNN(init(in_encoding_size, out_encoding_size), init(number_of_rules, out_encoding_size), activation_function)
    end


    function (m::GraphNN)(x::Matrix, g::EGraph, theory::Vector{<:AbstractRule})
        updated_graph_encoding = m.activation(x * m.W)

        rule_applicability_matrix = get_rule_applicability_matrix(g, theory)

        enode_to_rule_probability = my_softmax((updated_graph_encoding * transpose(m.R)) .* rule_applicability_matrix)

        extract_and_apply_max_rule(enode_to_rule_probability, g, theory)

        symplified_expression = extract!(g, astsize)
        return updated_graph_encoding, symplified_expression
    end


    function rule_application_locations(g::EGraph, theory::Vector{<:AbstractRule})
        sched = params.scheduler(g, theory, params.schedulerparams...)
        report = SaturationReport(g)
        number_of_matched_rules = eqsat_search!(g, theory, sched, report)
        return number_of_matched_rules
    end


    function get_rule_applicability_matrix(g::EGraph, theory::Vector{<:AbstractRule})
        _ = rule_application_locations(g, theory) 
        rule_adjacency = zeros(g.numclasses + EGraphProcessor.get_number_of_enodes(g), size(theory, 1))
        node_to_id_mapping = EGraphProcessor.get_enode_to_index_mapping(g)
        for applicable_rule in g.buffer
            rule_index = 0 
            for (k, j) in applicable_rule
                if k == 0
                    rule_index = abs(j[1])
                    node_index = node_to_id_mapping[j[2]]
                    rule_adjacency[node_index, rule_index] = 1
                elseif j[2] == 0
                    node_index = node_to_id_mapping[j[1]]
                    rule_adjacency[node_index, rule_index] = 1
                else
                    node_index = node_to_id_mapping[g.classes[j[1]][j[2]]]
                    rule_adjacency[node_index, rule_index] = 1
                end
            end
        end
        return rule_adjacency
    end


    function extract_max_rule_id(enode_to_rule_probability::Matrix)    
        max_rules_to_apply = argmax(enode_to_rule_probability, dims=2)

        tmp = Dict()
        for i in max_rules_to_apply
            if enode_to_rule_probability[i] != 0 || !isnan(enode_to_rule_probability[i])
                if i[2] in keys(tmp)
                    tmp[i[2]] += 1
                else
                    tmp[i[2]] = 1
                end
            end
        end

        return argmax(tmp)
    end
    
    
    function clear_non_max_rules!(g::EGraph, max_rule_id::Integer)
        for applicable_rule in g.buffer
            if applicable_rule[0][1] == max_rule_id
                g.buffer = [applicable_rule]
                break
            end
        end
    end

    
    function extract_and_apply_max_rule(enode_to_rule_probability::Matrix, g::EGraph, theory::Vector{<:AbstractRule})
        max_rule_id = extract_max_rule_id(enode_to_rule_probability) 
        clear_non_max_rules!(g, max_rule_id)
        report = SaturationReport(g)
        eqsat_apply!(g, theory, report, params)
        rebuild!(g)
    end

    
    function loss(initial_expression::Expr, symplified_expression::Expr)
        length_difference = (length(string(initial_expression)) - length(string(symplified_expression)))^2
        return length_difference
    end


    function loss(initial_expression::Expr, symplified_expression::Integer)
        length_difference = (length(string(initial_expression)) - 1)^2
        return length_difference
    end

end
