include("small_data_loader.jl")
include("egraph_processor.jl")
include("graph_neural_network_module.jl")
using .SmallDataLoader
using .EGraphProcessor
using .GraphNeuralNetwork
using Flux
using Metatheory
using Statistics
using Metatheory.EGraphs: apply_rule!


data = SmallDataLoader.load_data("data/test.json")
data = SmallDataLoader.preprosses_data_to_expressions(data)
#data = [:(a - a), :(v0 + 112 <= v0 + 112)]
my_theory = @theory a b c begin
    (a + b) + c --> a + (b + c)
    -a == -1 * a
    a - b == a + -b
    a - a --> 0
    a <= a --> 1
end

EGraphProcessor.update_terms([:+, :-, :*, :/, :<=, :min])

model = GraphNeuralNetwork.GraphNN(length(EGraphProcessor.all_terms) + 2, 32, size(my_theory, 1))
optimizer = ADAM()
max_iterations = 10

pc = Flux.params(model)

for ex in data[1:2]
    println("$ex")
    g = EGraph(ex) 
    for iteration in 1:max_iterations
        println("Iteration $iteration started")
        graph_encoding = EGraphProcessor.encode_graph(g)
        adj_matrix = EGraphProcessor.extract_adjacency_matrix(g)

        updated_graph_encoding, symplified_expression = model(adj_matrix * graph_encoding, g, my_theory)
        println("==>$symplified_expression")
        grad = gradient(pc) do 
            my_loss = GraphNeuralNetwork.loss(ex, symplified_expression)
            return my_loss
        end
        Flux.update!(optimizer, pc, grad)
    end
end
