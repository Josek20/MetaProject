include("small_data_loader.jl")
include("egraph_processor.jl")
include("graph_neural_network_module.jl")
using .SmallDataLoader
using .EGraphProcessor
using .GraphNeuralNetwork
using Metatheory
using Statistics
using Metatheory.EGraphs: apply_rule!


#data = SmallDataLoader.load_data("/home/rudy/JuliaProjects/data/test.json")
#data = SmallDataLoader.preprosses_data_to_expressions(data)
data = [:(a - a), :(v0 + 112 <= v0 + 112)]
my_theory = @theory a b c begin
    (a + b) + c --> a + (b + c)
    -a == -1 * a
    a - b == a + -b
    a - a --> 0
end

EGraphProcessor.update_terms([:+, :-, :*, :/, :<=])

model = GraphNeuralNetwork.GraphNN(length(EGraphProcessor.all_terms) + 2, 32, size(my_theory, 1))


for ex in data[1:2]
    if occursin("min", string(ex)) || occursin("max", string(ex)) || occursin("select", string(ex))
        continue
    end
    println("$ex")
    g = EGraph(ex) 

    graph_encoding = EGraphProcessor.encode_graph(g)
    adj_matrix = EGraphProcessor.extract_adjacency_matrix(g)

    symplified_expression = model(adj_matrix * graph_encoding, g, my_theory)

    #loss()
    println("==>$symplified_expression")
end
