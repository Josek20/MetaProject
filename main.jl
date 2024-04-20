include("small_data_loader.jl")
include("egraph_processor.jl")
include("graph_neural_network_module.jl")
using .SmallDataLoader
using .EGraphProcessor
using .GraphNeuralNetwork
using Flux
using Metatheory
using Statistics
using Metatheory.EGraphs: apply_rule!, extract!


data = SmallDataLoader.load_data("data/test.json")
data = SmallDataLoader.preprosses_data_to_expressions(data)
#data = [:(a - a), :(v0 + 112 <= v0 + 112)]
my_theory = @theory a b c begin
    (a + b) + c --> a + (b + c)
    -a == -1 * a
    a - b == a + -b
    a - a --> 0
    a <= a --> 1
    (a / b::Number) * b::Number --> a * 1
end


algebra_rules = @theory a b c begin
  a * (b * c) == (a * b) * c
  a + (b + c) == (a + b) + c

  a + b == b + a
  a * (b + c) == (a * b) + (a * c)
  (a + b) * c == (a * c) + (b * c)

  -a == -1 * a
  a - b == a + -b
  1 * a == a

  0 * a --> 0
  a + 0 --> a

  a::Number * b == b * a::Number
  a::Number * b::Number => a * b
  a::Number + b::Number => a + b
end

EGraphProcessor.update_terms([:+, :-, :*, :/, :<=, :min])

model = GraphNeuralNetwork.GraphNN(length(EGraphProcessor.all_terms) + 2, 32, size(algebra_rules, 1))
optimizer = ADAM()
max_iterations = 10

pc = Flux.params(model)

loss(x::Vector) = sum(log.(1 .+ exp.(x[begin + 1: end] .- x[begin])))

for ex in data[1:2]
    println("$ex")
    g = EGraph(ex) 
    for iteration in 1:max_iterations
        println("Iteration $iteration started")
        graph_encoding = EGraphProcessor.encode_graph(g)
        adj_matrix = EGraphProcessor.extract_adjacency_matrix(g)

        node_to_id_mapping = EGraphProcessor.get_enode_to_index_mapping(g)

        rule_applicability_matrix = GraphNeuralNetwork.get_rule_applicability_matrix(g, algebra_rules, node_to_id_mapping)
        
        enode_to_rule_probability = model(adj_matrix * graph_encoding, rule_applicability_matrix)
        
        rule_prob_dict = GraphNeuralNetwork.extract_and_apply_max_rule(enode_to_rule_probability, g, algebra_rules, node_to_id_mapping)
        
        symplified_expression = extract!(g, astsize)
        println("==>$symplified_expression")
        
        prob_vector = sort(collect(values(rule_prob_dict)), rev=true)
        grad = gradient(pc) do 
            if length(prob_vector) == 1
                return log(1 + exp(-prob_vector[1]))
            end
            #my_loss = GraphNeuralNetwork.loss(rule_prob_dict)
            my_loss = loss(prob_vector)
            return my_loss
        end
        Flux.update!(optimizer, pc, grad)
    end
end
