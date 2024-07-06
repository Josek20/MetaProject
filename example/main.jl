include("../src/MyModule.jl")
using .MyModule
using Flux
using Metatheory
using Statistics
using Metatheory.EGraphs: extract!


#data = SmallDataLoader.load_data("data/test.json")
#data = SmallDataLoader.preprosses_data_to_expressions(data)
data = [:(a - a), :(v0 + 112 <= v0 + 112)]
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
    (a / b) * c == (a * c) / b


    -a == -1 * a
    a - b == a + -b
    1 * a == a
    a + a == 2*a

    0 * a --> 0
    a + 0 --> a
    a - a --> 0
    a <= a --> 1
    (a / b) * b --> a

    min(a, min(a, b)) --> min(a, b)
    min(min(a, b), b) --> min(a, b)
    max(a, max(a, b)) --> max(a, b)
    max(max(a, b), b) --> max(a, b)

    max(a, min(a, b)) --> a
    min(a, max(a, b)) --> a
    max(min(a, b), b) --> b
    min(max(a, b), b) --> b
end

MyModule.update_terms([:+, :-, :*, :/, :<=, :min, :max, :select, :&&, :||, :>=, :<, :>])

model = MyModule.GraphNN(length(MyModule.all_terms) + 2, 32, size(algebra_rules, 1))

optimizer = ADAM()
max_iterations = 50

pc = Flux.params(model)

loss(x::Vector) = length(x) != 1 ? sum(log.(1 .+ exp.(x[begin + 1: end] .- x[begin]))) : log(1 + exp(-x[1]))


for ex in data[1:2]
    println("$ex")
    g = EGraph(ex) 
    applied_rules_buffer = []

    ### Training code
    for iteration in 1:max_iterations
        #println("Iteration $iteration started")
        graph_encoding = MyModule.encode_graph(g)
        adj_matrix = MyModule.extract_adjacency_matrix(g)
        


        node_to_id_mapping = MyModule.get_enode_to_index_mapping(g)
        gnn_graph = MyModule.transform_egraph_to_gnn_graph(adj_matrix)


        rule_applicability_matrix = MyModule.get_rule_applicability_matrix(g, algebra_rules, node_to_id_mapping)
        
        enode_to_rule_probability = model(gnn_graph, adj_matrix * graph_encoding, rule_applicability_matrix)
        
        #println("probability size $(size(enode_to_rule_probability))")
        rule_prob_dict, max_rule = MyModule.extract_and_apply_max_rule(enode_to_rule_probability, g, algebra_rules, node_to_id_mapping)

        push!(applied_rules_buffer, max_rule)
        #println(rule_prob_dict) 
        #symplified_expression = extract!(g, astsize)
        #println("==>$symplified_expression")
        
        prob_vector = sort(collect(values(rule_prob_dict)), rev=true)
        grad = gradient(pc) do 
            my_loss = loss(prob_vector)
            return my_loss
        end
        Flux.update!(optimizer, pc, grad)
    end
    symplified_expression = extract!(g, astsize)
    println("==>$symplified_expression")
    ### Extracting plan
    #
end
