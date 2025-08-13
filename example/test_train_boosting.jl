using CSV
using DataFrames
using MyModule
using MyModule: exp_size
using MyModule.DataStructures
using MyModule.Flux
using MyModule.Mill
using Serialization
using Optimisers
using Statistics
using Statistics
using Plots

experiment_name = "test_heuristic_boosted_training_1ksamples"
train_data_path = "./data/neural_rewrter/train.json"
train_data = load_data(train_data_path)[1:1_000]
train_data = filter(x->!occursin("select", x[1]), train_data)
train_data = preprosses_data_to_expressions(train_data)
sorted_data = sort(train_data, by=x->MyModule.exp_size(x))
data = sorted_data

# epochs = 2
# inner_epochs = 10

hidden_size = 64

function ffnn(idim, hidden_size, layers)
    layers == 1 && return Dense(idim, hidden_size, Flux.gelu)
    layers == 2 && return Flux.Chain(Dense(idim, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu))
end

head_model = ProductModel(
    (;head = ffnn(length(new_all_symbols), hidden_size, 1),
      args = ffnn(hidden_size, hidden_size, 1),  
        ),
    ffnn(2*hidden_size, hidden_size, 1)
    )

args_model = ProductModel(
    (;args = ffnn(hidden_size, hidden_size, 1),  
      position = Dense(2,hidden_size),  
        ),
    ffnn(2*hidden_size, hidden_size, 1)
    )

model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Flux.Chain(Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, 1)),
    );



function self_boosted_train(model, data, epochs=1, initial_steps=100, initial_depth=100)
    optimizer=ADAM()
    opt_state = Flux.setup(optimizer, model)
    empty!(MyModule.nc)
    empty!(MyModule.memoize_cache(MyModule.exp_size))
    empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
    empty!(MyModule.memoize_cache(MyModule.all_expand))
    samples = [(;ds=nothing,hp=[],hn=[],goal_size=typemax(Int),initial_expr=i) for i in data]
    all_stats = []
    avg_solutionv = []
    avg_nodes = []
    avg_solutionsl = []
    for ep in 1:epochs
        # Building Tree extracting training samples
        solutionv = []               
        nodes = []                   
        solutionsl = []
        println("Ep: $(ep), Searching the tree for new training samples")
        t = @elapsed samples = map(samples) do i
            goal_size, ex = i.goal_size, i.initial_expr
            goal_size == 1 && return(i)
            soltree, smallest_node, root = MyModule.initialize_tree_search(MyModule.intern!(ex), model, max_expansions=initial_steps * ep, max_depth=initial_depth)
            push!(nodes, length(soltree))                    
            push!(solutionsl, smallest_node.depth)         
            push!(solutionv, exp_size(root.ex) - exp_size(smallest_node.ex))
           
            if MyModule.exp_size(smallest_node.ex) < goal_size
                println("$(ex) -> $(MyModule.expr(MyModule.nc, smallest_node.ex))")
                training_data, hp, hn, _, _ = MyModule.extract_training_data(smallest_node, soltree, root)
                return(;ds=MyModule.deduplicate(training_data), hp=hp, hn=hn, goal_size=MyModule.exp_size(smallest_node.ex), initial_expr=i.initial_expr)
            else
                return(i)
            end
        end
        push!(avg_nodes, mean(nodes))
        push!(avg_solutionsl, mean(solutionsl))
        push!(avg_solutionv, mean(solutionv))
        push!(all_stats,  [s.goal_size for s in samples])
        println(mean.(all_stats))
        println("Ep: $(ep), Finished Searching the tree in $(t)")
        println("Ep: $(ep), Training model on a training samples")
        p1 = plot(nodes, xlabel="Expression ID", ylabel="Number of unique explored nodes", legend=false)
        p2 = plot(solutionsl, xlabel="Expression ID", ylabel="Best solution legnth", legend=false)
        p3 = plot(solutionv, xlabel="Expression ID", ylabel="Best solution simplification", legend=false)
        plot(p1, p2, p3, layout=(3,1))
        savefig("trained_planning_stats_avg1.png")
        # Model training
        # for in_ep in 1:epochs
        #     sum_loss = 0
        #     t = @elapsed for (i, (ds, I₊, I₋)) in enumerate(samples)
        #         sa, grad = Flux.Zygote.withgradient(Base.Fix2(MyModule.lossfun, (ds, I₊, I₋)), model)
        #         sum_loss += sa
        #         Optimisers.update!(opt_state, model, only(grad))
        #     end
        #     violations = [MyModule.hardloss(model, sample) for sample in samples]
        #     @show (t, sum(violations), quantile(violations, 0:0.1:1))
        # end
        empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
    end
    # p1 = plot(avg_nodes, xlabel="Expression ID", ylabel="Number of unique explored nodes", legend=false)
    # p2 = plot(avg_solutionsl, xlabel="Expression ID", ylabel="Best solution legnth", legend=false)
    # p3 = plot(avg_solutionv, xlabel="Expression ID", ylabel="Best solution simplification", legend=false)
    # plot(p1, p2, p3, layout=(3,1))
    # savefig("trained_planning_stats_avg1.png")
    # serialize("models/trained_heuristic_$(experiment_name)_ep$(epochs)_hidden$(hidden_size).bin", model)
end


function validate_train(model, data)
    empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
    empty!(MyModule.memoize_cache(MyModule.exp_size))
    df = map(data) do ex
        soltree, smallest_node, root = MyModule.initialize_tree_search(MyModule.intern!(ex), model, max_expansions=1000, max_depth=100)
        (; s₀ = MyModule.exp_size(root.ex), sₙ = MyModule.exp_size(smallest_node.ex), se = smallest_node.ex, pr = [])
    end |> DataFrame
    CSV.write("stats/results_of_$(experiment_name)_ep$(epochs)_hidden$(hidden_size).csv", df)
end

self_boosted_train(model, data)
# validate_train(model, data)

# model = deserialize("models/trained_heuristic_test_heuristic_boosted_1h_ep10_hidden64.bin")
function get_solution_ids(soltree, smallest_node, solution_ids=[])
    if smallest_node.depth == 0
        push!(solution_ids, smallest_node.node_id)
        return
    end
    get_solution_ids(soltree, soltree[smallest_node.parent], solution_ids)
    push!(solution_ids, smallest_node.node_id)
    # return solution_ids
end
function visualize(model, data)
    ex = data[end]
    soltree, smallest_node, root = MyModule.initialize_tree_search(MyModule.intern!(ex), model, max_expansions=100, max_depth=45)
    sort_by_depth = sort(collect(values(soltree)), by=x->x.depth)
    solution_ids = []
    get_solution_ids(soltree, smallest_node, solution_ids)
    filtered_sorted_by_depth = filter(x->x.node_id in solution_ids,sort_by_depth)
    children = Vector[]
    text = [string(expr(MyModule.nc, sort_by_despth[1].ex))]
    link_style = [""]
    style = [""]
    ind_t = 1
    tmp = 1
    current_range = 1:1

    # for i in 1:sort_by_depth[end].depth
    for i in 1:length(filtered_sorted_by_depth)
        # push!(children, [])
        if length(children) == 0
            current_range = 2:length(filtered_sorted_by_depth[i].children)
            push!(children, collect(current_range))
        else
            current_range = current_range.stop + 1:length(filtered_sorted_by_depth[i].children) + current_range.stop
            children[tmp] = collect(current_range)
        end
        for (ind, node_ids) in enumerate(filtered_sorted_by_depth[i].children)
            push!(children, [])
            ind_t += 1
            if node_ids in solution_ids
                tmp = ind_t
                push!(text, string(expr(MyModule.nc, soltree[node_ids].ex)) * "\nPred: $(only(model(soltree[node_ids].ex)))")
                push!(link_style, "stroke:blue")
                push!(style, "fill:green")
            else
                push!(text, "")
                push!(link_style, "")
                push!(style, "")
            end
        end
    end
    t = D3Tree(children, text=text, style=style, link_style=link_style, init_expand=2,  svg_node_size=(2020, 2020))
    inbrowser(t, "Mircosoft Edge")
end
