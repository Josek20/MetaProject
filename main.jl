include("small_data_loader.jl")
include("egraph_processor.jl")
include("graph_neural_network_module.jl")
using .SmallDataLoader
using .EGraphProcessor
using .GraphNeuralNetwork
using Metatheory


#data = SmallDataLoader.load_data("path")

my_theory = @theory a b c begin
    (a + b) + c --> a + (b + c)
    -a == -1 * a
    a - b == a + -b
    a - a -->0
end

model = GraphNeuralNetwork.GraphNN(6, 32, size(my_theory, 1))

ex = :(a - a)
g = EGraph(ex) 

graph_encoding = EGraphProcessor.encode_graph(g)
adj_matrix = EGraphProcessor.extract_adjacency_matrix(g)

updated_graph_encoding = model(adj_matrix * graph_encoding)

rule_applicability_matrix = GraphNeuralNetwork.get_rule_applicability_matrix(g, my_theory)

enode_to_rule_probability = (updated_graph_encoding * transpose(model.R)) .* rule_applicability_matrix

# Missing step 1) choose highets prob rules
max_rules_to_apply = argmax(enode_to_rule_probability, dims=2)

# Missing step 2) apply chosen rules
for i in max_rules_to_apply
    if enode_to_rule_probability[i] != 0
        bindings = pop!(g.buffer)
        rule_idx, id = bindings[0]
        direction = sign(rule_idx)
        rule_idx = abs(rule_idx)
        rule = theory[rule_idx]


        halt_reason = apply_rule!(bindings, g, rule, id, direction) 
    end
end


symplified_expression = extract!(g, astsize)

