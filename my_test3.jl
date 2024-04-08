using Metatheory
using StatsBase

function class_encoder()

end

all_literals = [[Symbol("v$i") for i in 0:12]; [:a, :b, :c, -1, 0]]
all_terms = [:+, :-, :*, :/]

literals_encoding_map = Dict(zip(all_literals, collect(1:length(all_literals))))
term_encoding_map = Dict(zip(all_terms, collect(1:length(all_terms))))


encode_node(node::EClass) = [1, 0, 0]
encode_node(node::ENodeTerm) = [0, term_encoding_map[node.operation], 0]
encode_node(node::ENodeLiteral) = [0, 0, literals_encoding_map[node.value]]




ex1 = :(a - a + b - c / v1 * v1 - v11)
ex = :(a - a)
test_egraph = EGraph(ex)

classes_encoding(egraph::EGraph) = [encode_node(node_classes) for (i, node_classes) in egraph.classes]
node_encoding(egraph::EGraph) = [encode_node(j) for (i, node_classes) in egraph.classes for j in node_classes]
classes_graph_encoding = classes_encoding(test_egraph)
node_graph_encoding = node_encoding(test_egraph)

graph_encoding = [classes_graph_encoding; node_graph_encoding]



my_rules = @theory a b c begin
    (a + b) + c --> a + (b + c)
    -a == -1 * a
    a - b == a + -b
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
    adj_matrix = zeros(Int, num_nodes + egraph.numclasses, num_nodes + egraph.numclasses)
    ind_to_class_id = Dict(zip(keys(egraph.classes), collect(1:length(egraph.classes))))
    for (enum_ind, (class_id, class_nodes)) in enumerate(egraph.classes)
        for j in class_nodes
            adj_matrix[egraph.numclasses + counter, enum_ind] += 1
            if typeof(j) == ENodeTerm
                for k in j.args
                    adj_matrix[egraph.numclasses + counter, ind_to_class_id[k]] += 1
                end
            end
            counter += 1
        end
    end
    return adj_matrix
end

adjacency_matrix = get_adjacency_matrix(test_egraph)
