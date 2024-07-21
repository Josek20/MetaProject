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
    a::Number + b::Number => a + b
    a::Number - b::Number => a - b
    a::Number * b::Number => a * b
    a::Number / b::Number => a / b


    a * (b * c) --> (a * b) * c
    (a * b) * c --> a * (b * c)
    a + (b + c) --> (a + b) + c
    (a + b) + c --> a + (b + c)
    a + (b + c) --> (a + c) + b
    (a + b) + c --> (a + c) + b
    (a - b) - c --> (a - c) - b
    (a - b::Number) - c::Number --> a - (c + b)
    (a + b::Number) - c::Number --> a + (b - c)
    (a - b) + b --> a
    (-a + b) + a --> b


    a + b --> b + a
    a * (b + c) --> (a * b) + (a * c)
    (a + b) * c --> (a * c) + (b * c)
    (a / b) * c --> (a * c) / b
    (a / b) * b --> a
    (a * b) / b --> a



    #-a --> -1 * a
    #a - b --> a + -b
    1 * a --> a
    a + a --> 2*a

    0 * a --> 0
    a + 0 --> a
    a - a --> 0

    a <= a --> 1
    a + b <= c + b --> a <= c
    a * b <= c * b --> a <= c
    a / b <= c / b --> a <= c
    a - b <= c - b --> a <= c
    a::Number <= b::Number => a<=b ? 1 : 0
    a <= b - c --> a + c <= b
    #a <= b - c --> a - b <= -c
    a <= b + c --> a - c <= b
    a <= b + c --> a - b <= c

    min(a, min(a, b)) --> min(a, b)
    min(min(a, b), b) --> min(a, b)
    max(a, max(a, b)) --> max(a, b)
    max(max(a, b), b) --> max(a, b)

    max(a, min(a, b)) --> a
    min(a, max(a, b)) --> a
    max(min(a, b), b) --> b
    min(max(a, b), b) --> b

    min(a + b::Number, a + c::Number) => b < c ? :($a + $b) : :($a + $c)
    max(a + b::Number, a + c::Number) => b < c ? :($a + $c) : :($a + $b)
    min(a - b::Number, a - c::Number) => b < c ? :($a - $c) : :($a - $b)
    max(a - b::Number, a - c::Number) => b < c ? :($a - $b) : :($a - $c)
    min(a * b::Number, a * c::Number) => b < c ? :($a * $b) : :($a * $c)
    max(a + b::Number, a + c::Number) => b < c ? :($a + $c) : :($a + $b)
    #min(a + b, a + c) --> min(b, c)
end

MyModule.update_terms([:+, :-, :*, :/, :<=, :min, :max, :select, :&&, :||, :>=, :<, :>])

model = MyModule.GraphNN(length(MyModule.all_terms) + 2, 32, size(algebra_rules, 1))

optimizer = ADAM()
max_iterations = 10

pc = Flux.params(model)

loss(x::Vector) = length(x) != 1 ? sum(log.(1 .+ exp.(x[begin + 1: end] .- x[begin]))) : log(1 + exp(-x[1]))

function loss(model, gnn_graph, x, rule_applicability_matrix, y_ind)
  new_x = sum(model(gnn_graph, x, rule_applicability_matrix))
  #result = mean(exp.(1 .+ log.(new_x[y_ind] .- new_x[1:end .!= y_ind])))
  # return result
end

function train_gnn(model, train_data, training_samples)
    for (ind, ex) in enumerate(train_data)
        println("Expression: $ex")
        g = EGraph(ex) 
        applied_rules_buffer = []

        if !training_samples[ind].saturated
            continue
        end
        pr = training_samples[ind].proof
        println("Proof vector: $pr")
        ## Training code
        for iteration in 1:length(training_samples[ind].proof)
            #println("Iteration $iteration started")
            graph_encoding = MyModule.encode_graph(g)
            adj_matrix = MyModule.extract_adjacency_matrix(g)



            node_to_id_mapping = MyModule.get_enode_to_index_mapping(g)
            gnn_graph = MyModule.transform_egraph_to_gnn_graph(adj_matrix)


            rule_applicability_matrix = MyModule.get_rule_applicability_matrix(g, algebra_rules, node_to_id_mapping)

            enode_to_rule_probability = model(gnn_graph, adj_matrix * graph_encoding, rule_applicability_matrix)

            #println("probability size $(size(enode_to_rule_probability))")
            rule_prob_dict, max_rule = MyModule.extract_and_apply_max_rule(enode_to_rule_probability, g, algebra_rules, node_to_id_mapping)

            #push!(applied_rules_buffer, max_rule)
            #println(rule_prob_dict) 
            #symplified_expression = extract!(g, astsize)
            #println("==>$symplified_expression")

            #prob_vector = sort(collect(values(rule_prob_dict)), rev=true)
            grad = gradient(pc) do
              loss(model, gnn_graph, adj_matrix * graph_encoding, rule_applicability_matrix, training_samples[ind].proof[iteration])
            end
            # grad = gradient(pc) do 
            #     my_loss = loss(prob_vector)
            #     return my_loss
            # end
            Flux.update!(optimizer, pc, grad)
            # x = adj_matrix * graph_encoding 
            # updated_graph_encoding = model.GCN(gnn_graph, transpose(x))
            #
            # prod1 = transpose(model.R * updated_graph_encoding)
            # change this to be just softmax or make so that initially it will pass only non zero values
            #enode_to_rule_probability = my_softmax(prod1 .* rule_applicability_matrix)
        end
        #symplified_expression = extract!(g, astsize)
        #println("==>$symplified_expression")
        ### Extracting plan
        #
    end
    return model
end

function test_gnn(mode, test_data, max_depth = 10)
    symplified_length = []
    for (ind, ex) in enumerate(test_data)
        println("Expression: $ex")
        g = EGraph(ex) 
        applied_rules_buffer = []

        ## Training code
        for iteration in 1:max_depth
            #println("Iteration $iteration started")
            graph_encoding = MyModule.encode_graph(g)
            adj_matrix = MyModule.extract_adjacency_matrix(g)



            node_to_id_mapping = MyModule.get_enode_to_index_mapping(g)
            gnn_graph = MyModule.transform_egraph_to_gnn_graph(adj_matrix)


            rule_applicability_matrix = MyModule.get_rule_applicability_matrix(g, algebra_rules, node_to_id_mapping)

            enode_to_rule_probability = model(gnn_graph, adj_matrix * graph_encoding, rule_applicability_matrix)

            rule_prob_dict, max_rule = MyModule.extract_and_apply_max_rule(enode_to_rule_probability, g, algebra_rules, node_to_id_mapping)

        end
        symplified_expression = extract!(g, astsize)
        ls = exp_size(symplified_expression)
        li = exp_size(ex)
        push!(symplified_length, abs(li - ls))
        println("==>$symplified_expression")
        ### Extracting plan
        #
    end
    result = mean(symplified_length)
    println("AES: $result")
end


model = train_gnn(model, train_data, training_samples)
test_gnn(model, test_data)
