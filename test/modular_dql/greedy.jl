using CSV
using DataFrames
using Statistics
using MyModule
using MyModule: all_expand, exp_size, Node, NodeID, intern!, expr, DeduplicatingNode, AbstractModel

function get_data()
    train_data_path = "./data/neural_rewrter/val.json"
    train_data = load_data(train_data_path)[1:1_000]
    train_data = filter(x->!occursin("select", x[1]), train_data)
    train_data = preprosses_data_to_expressions(train_data)
    sorted_data = sort(train_data, by=x->MyModule.exp_size(x))
    data = sorted_data
    return Vector{Expr}(data)
end

data = get_data()
max_steps = 100
max_depth = 100
MyModule.reset_all_function_caches()
res = map(data) do i
    tmp = intern!(i)
    soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(tmp, exp_size; max_expansions=max_steps, max_depth=max_depth)
    smalles_node = MyModule.extract_smallest_node(soltree)
    tmp = smalles_node.ex
    pr = Tuple{Expr, Int}[]
    # for i in 1:smalles_node.depth
    #     push!(pr, (expr(MyModule.nc, smallest_node.ex), smalles_node.rule_index[2]))
    #     smalles_node = soltree[smalles_node.parent]
    # end
    (; s₀ = exp_size(root.ex), sₙ = exp_size(tmp), se = expr(MyModule.nc, tmp), pr = reverse(pr))
end |> DataFrame
@show mean(res[!, :s₀] - res[!, :sₙ])
# CSV.write("stats/results_of_greedy_baseline_steps$(max_steps).csv", res)