using CSV
using DataFrames
using MyModule
using MyModule.DataStructures
using MyModule.Flux
using MyModule.Mill
using Serialization
using Optimisers
using Statistics

experiment_name = "test_heuristic_boosted_training_10ksamples"
train_data_path = "./data/neural_rewrter/train.json"
train_data = load_data(train_data_path)[1:10_000]
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



function self_boosted_train(model, data, epochs=10, initial_steps=100, initial_depth=100)
    optimizer=ADAM()
    opt_state = Flux.setup(optimizer, model)
    empty!(MyModule.nc)
    empty!(MyModule.memoize_cache(MyModule.exp_size))
    empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
    empty!(MyModule.memoize_cache(MyModule.all_expand))
    samples = [(;ds=nothing,hp=[],hn=[],goal_size=typemax(Int),initial_expr=i) for i in data]
    all_stats = []
    for ep in 1:epochs
        # Building Tree extracting training samples
        
        println("Ep: $(ep), Searching the tree for new training samples")
        t = @elapsed samples = map(samples) do i
            goal_size, ex = i.goal_size, i.initial_expr
            goal_size == 1 && return(i)
            soltree, smallest_node, root = MyModule.initialize_tree_search(MyModule.intern!(ex), model, max_expansions=initial_steps * ep, max_depth=initial_depth)
            if MyModule.exp_size(smallest_node.ex) < goal_size
                println("$(ex) -> $(MyModule.expr(MyModule.nc, smallest_node.ex))")
                training_data, hp, hn, _, _ = MyModule.extract_training_data(smallest_node, soltree, root)
                return(;ds=MyModule.deduplicate(training_data), hp=hp, hn=hn, goal_size=MyModule.exp_size(smallest_node.ex), initial_expr=i.initial_expr)
            else
                return(i)
            end
        end
        push!(all_stats,  [s.goal_size for s in samples])
        println(mean.(all_stats))
        println("Ep: $(ep), Finished Searching the tree in $(t)")
        println("Ep: $(ep), Training model on a training samples")
        # Model training
        for in_ep in 1:epochs
            sum_loss = 0
            t = @elapsed for (i, (ds, I₊, I₋)) in enumerate(samples)
                sa, grad = Flux.Zygote.withgradient(Base.Fix2(MyModule.lossfun, (ds, I₊, I₋)), model)
                sum_loss += sa
                Optimisers.update!(opt_state, model, only(grad))
            end
            violations = [MyModule.hardloss(model, sample) for sample in samples]
            @show (t, sum(violations), quantile(violations, 0:0.1:1))
        end
        empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
    end

    serialize("models/trained_heuristic_$(experiment_name)_ep$(epochs)_hidden$(hidden_size).bin", model)
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

# self_boosted_train(model, data)
# validate_train(model, data)
