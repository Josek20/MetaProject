using Flux
using Flux
using Flux: crossentropy, ADAM
using Statistics: mean
using Metatheory
import .Metatheory.EGraphs: SaturationReport, eqsat_search!
using StatsBase


all_literals = [[Symbol("v$i") for i in 0:12]; [:a, :b, :c, -1, 0]]
all_terms = [:+, :-, :*, :/]

literals_encoding_map = Dict(zip(all_literals, collect(1:length(all_literals))))
term_encoding = unique(all_terms) .== permutedims(all_terms) 
term_encoding_map = Dict(all_terms[x] => term_encoding[x,:] for x in 1:length(all_terms)) 


encode_node(node::EClass) = [1, 0, 0, 0, 0, 0]
#encode_node(node::ENodeTerm) = [0, term_encoding_map[node.operation], 0]
encode_node(node::ENodeTerm) = vcat([0, 0], term_encoding_map[node.operation])
encode_node(node::ENodeLiteral) = [0, 1, 0, 0, 0, 0]




ex1 = :(a - a + b - c / v1 * v1 - v11)
ex = :(a - b)
test_egraph = EGraph(ex1)

classes_encoding(egraph::EGraph) = [encode_node(node_classes) for (i, node_classes) in egraph.classes]
node_encoding(egraph::EGraph) = [encode_node(j) for (i, node_classes) in egraph.classes for j in node_classes]
all_nodes = [j for (i, node_classes) in test_egraph.classes for j in node_classes]
node_to_id_mapping = Dict(zip(vcat(keys(test_egraph.classes)..., all_nodes), collect(1:length(all_nodes)+test_egraph.numclasses)))
classes_graph_encoding = classes_encoding(test_egraph)
node_graph_encoding = node_encoding(test_egraph)

graph_encoding = [classes_graph_encoding; node_graph_encoding]

graph_encoding = Matrix(transpose(hcat(graph_encoding...)))


my_rules = @theory a b c begin
    (a + b) + c --> a + (b + c)
    -a == -1 * a
    a - b == a + -b
    a - a -->0
end


function get_nodes_number(egraph::EGraph)
    return length(unique(keys(egraph.memo)))
end


function get_adjacency_matrix1(egraph::EGraph)
    num_nodes = get_nodes_number(egraph)
    adj_matrix = zeros(Int, num_nodes, num_nodes)
    counter = 1
    for (_, nodes) in egraph.classes
        for node in nodes
            for edge in egraph.memo[node]
                adj_matrix[counter, edge] = 1
                counter += 1
            end
        end
    end
    
    return adj_matrix    
end

function get_adjacency_matrix(egraph::EGraph)
    num_nodes = get_nodes_number(egraph)
    # first classes than nodes
    counter = 1
    adj_matrix1 = zeros(Int, num_nodes + egraph.numclasses, num_nodes + egraph.numclasses)
    #adj_matrix2 = zeros(Int, num_nodes + egraph.numclasses, num_nodes + egraph.numclasses)
    ind_to_class_id = Dict(zip(keys(egraph.classes), collect(1:length(egraph.classes))))
    for (enum_ind, (class_id, class_nodes)) in enumerate(egraph.classes)
        for j in class_nodes
            adj_matrix1[egraph.numclasses + counter, enum_ind] += 1
            #adj_matrix2[egraph.numclasses + counter, enum_ind] += 1
            if typeof(j) == ENodeTerm
                for k in j.args
                    adj_matrix1[egraph.numclasses + counter, ind_to_class_id[k]] += 1
                #adj_matrix2[egraph.numclasses + counter, ind_to_class_id[2]] += 1
                end
            end
            counter += 1
        end
    end
    return adj_matrix1
end

adjacency_matrix = get_adjacency_matrix(test_egraph)



# Define the GNN architecture
struct GraphNN{W, R, F, RA}
    W::W
    R::R
    activation::F
    rule_adjacency::RA
end

GraphNN(in::Integer, out::Integer, num_rules::Integer, act_function=sigmoid, init=Flux.glorot_uniform) = GraphNN(init(in, out), init(num_rules, out), act_function, zeros(size(graph_encoding, 1), num_rules))


function (m::GraphNN)(x::Matrix)
    x_new = m.activation(x * m.W)
    return x_new
end

model = GraphNN(size(graph_encoding, 2), 32, size(my_rules, 1))


# size <number_of_nodes> x 32
updated_nodes_encoding = model(adjacency_matrix * graph_encoding)


# find which rule can be apllied to which node, result is stored in egraph.buffer
#=
for (rule_idx, rule) in enumerate(my_rules)
    rules_matched = 0
    classes_ids = EGraphs.cached_ids(test_egraph, rule.left)
    if rule isa BidirRule
        classes_ids = union(classes_ids, EGraphs.cached_ids(test_egraph, rule.right))
    end
    for i in classes_ids
        rules_matched += rule.ematcher!(test_egraph, rule_idx, i)
    end
end
=#
params = SaturationParams()
sched = params.scheduler(test_egraph, my_rules, params.schedulerparams...)
report = SaturationReport(test_egraph)
eqsat_search!(test_egraph, my_rules, sched, report)
# change the adjacency_rule matrix to correspond to the rules 
# that can be applied to a specific node

for applicable_rule in test_egraph.buffer
    rule_index = 0 
    for (k, j) in applicable_rule
        if k == 0
            rule_index = j[1]
            node_index = node_to_id_mapping[j[2]]
            model.rule_adjacency[node_index, rule_index] = 1
        elseif j[2] == 0
            node_index = node_to_id_mapping[j[1]]
            model.rule_adjacency[node_index, rule_index] = 1
        else
            node_index = node_to_id_mapping[test_egraph.classes[j[1]][j[2]]]
            model.rule_adjacency[node_index, rule_index] = 1
        end

    end
end

node_rule_prob = (updated_nodes_encoding * transpose(model.R)) .* model.rule_adjacency
# check how to extract the simplified expression from egraph
