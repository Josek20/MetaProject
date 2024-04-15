module GraphNeuralNetwork
    using Flux
    using Metatheory
    using .Metatheory.EGraphs: SaturationReport, eqsat_search!
    include("egraph_processor.jl")
    using .EGraphProcessor
    export GraphNN, train_model, evaluate_model

    params = SaturationParams()

    mutable struct GraphNN{W, R, F}
        W::W
        R::R
        activation::F
    end

    function GraphNN(in_encoding_size::Integer, out_encoding_size::Integer, number_of_rules::Integer, activation_function=sigmoid, init=Flux.glorot_uniform)
        return GraphNN(init(in_encoding_size, out_encoding_size), init(number_of_rules, out_encoding_size), activation_function)
    end

    function (m::GraphNN)(x::Matrix)
        new_x = m.activation(x * m.W)
        return new_x
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
                    rule_index = j[1]
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
end
