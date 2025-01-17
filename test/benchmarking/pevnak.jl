using MyModule
using MyModule.Flux
using MyModule.Mill
using MyModule.LRUCache
using Serialization
using DataFrames
using Optimisers
using Statistics
using CSV

function prepare_dataset(n=typemax(Int))
    samples = deserialize("data/training_data/size_heuristic_training_samples1.bin")
    samples = vcat(samples...)
    # samples = sort(samples, by=x->MyModule.exp_size(x.initial_expr, LRU(maxsize=10000)))
    samples = sort(samples, by=x->length(x.proof))
    last(samples, min(length(samples), n))
end

Heaviside(x) = x > 0 ? 1 : 0
hinge0(x::T) where {T} = x > 0 ? x : zero(T)
hinge1(x) = hinge0(x + 1)
 
function get_batched_samples(samples, batch_size=10)
    batched_samples = []
    for i in 1:batch_size:length(samples)-1
        pds = reduce(catobs, [j[1] for j in samples[i:batch_size - 1 + i]])
        pds_size = cumsum([size(bd.data.head)[2] for (bd, _, _) in samples[i:batch_size - 1 + i]])
        pushfirst!(pds_size, 0)
        modified_hps = map(x->x[1] .+ x[2], zip([hp for (_, hp, _) in samples[i:batch_size - 1 + i]], pds_size))
        modified_hns = map(x->x[1] .+ x[2], zip([hn for (_, _, hn) in samples[i:batch_size - 1 + i]], pds_size))

        hps = vcat(modified_hps...)
        hns = vcat(modified_hns...)
        
        push!(batched_samples, (pds, hps, hns))
    end
    return batched_samples
end

function lossfun(m, (ds, I₊, I₋))
    o = vec(MyModule.heuristic(m, ds))
    mean(softplus.(o[I₊] - o[I₋]))
end

function hardloss(m, (ds, I₊, I₋))
    o = vec(MyModule.heuristic(m, ds))
    sum(Heaviside.(o[I₊] - o[I₋]))
end


function train(model, samples, training_samples)
    optimizer=ADAM()
    opt_state = Flux.setup(optimizer, model)
    loss_stats = []
    matched_stats = []
    cache = LRU(maxsize=1000_000)
    exp_cache = LRU(maxsize=1000_000)
    size_cache = LRU(maxsize=1000_000)
    expr_cache = LRU(maxsize=1000_000) 
    # batched_samples = get_batched_samples(samples, batch_size)
    for outer_epoch in 1:10
        for ep in 1:10
            sum_loss = 0.0
            hard_loss = 0
            t = @elapsed for (i, (ds, I₊, I₋)) in enumerate(samples)
                sa, grad = Flux.Zygote.withgradient(Base.Fix2(lossfun, (ds, I₊, I₋)), model)
                sum_loss += sa
                Optimisers.update!(opt_state, model, only(grad))
            end
            violations = [hardloss(model, sample) for sample in samples]
            @show (t, sum(violations), quantile(violations, 0:0.1:1))
        end
        new_samples = []
        better_samples_counter = 0
        empty!(cache)
        t = @elapsed for old_sample in training_samples
            simplified_expression, _, big_vector, saturated, hp, hn, _, proof_vector, _ = MyModule.initialize_tree_search(model, old_sample.initial_expr, 1000, 10, new_all_symbols, theory, cache, exp_cache, size_cache, expr_cache, 0.5)
            new_sample = MyModule.TrainingSample(big_vector, saturated, simplified_expression, proof_vector, hp, hn, old_sample.initial_expr)
            if MyModule.isbetter(old_sample, new_sample, size_cache)
                 push!(new_samples, (big_vector, hp, hn))
                 better_samples_counter += 1
            else
                push!(new_samples, (old_sample.training_data, old_sample.hp, old_sample.hn))
            end
        end
        @show (t, better_samples_counter)
        samples = new_samples
    # resolve all expressions
    # improve the current solutions
    # form the training samples

    end
end


hidden_size = 128

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

heuristic = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Flux.Chain(Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, 1)),
    );
    
training_samples = prepare_dataset(1);
epochs = 1000

samples = [MyModule.get_training_data_from_proof(sample.proof, sample.initial_expr) for sample in training_samples]
samples = map(x->(MyModule.deduplicate(x[1]), x[2], x[3]), samples)
train(heuristic, samples, training_samples)

data = [i.initial_expr for i in training_samples]
experiment_name = "testing_new_training_heuristic_v1_sorted_$(length(data))_$(epochs)_hidden_size_$(hidden_size)"
serialize("models/trained_heuristic_$(experiment_name).bin", heuristic)
#heuristic = deserialize("models/trained_heuristic_testing_new_training_api_sorted_filtered_955_10_hidden_size_64.bin")
df = map(data) do ex
    exp_cache = LRU{MyModule.ExprWithHash, Vector}(maxsize=100_000)
    cache = LRU{MyModule.ExprWithHash, Vector}(maxsize=1_000_000)
    size_cache = LRU{MyModule.ExprWithHash, Int}(maxsize=100_000)
    expr_cache = LRU{UInt, MyModule.ExprWithHash}(maxsize=100_000)
    # ex = MyModule.interned
    # root = MyModule.Node(ex, (0,0), nothing, 0, expr_cache)

    # soltree = Dict{UInt64, MyModule.Node}()
    # open_list = PriorityQueue{MyModule.Node, Float32}()
    # close_list = Set{UInt64}()
    # println("Initial expression: $ex")
    # soltree[root.node_id] = root
    # o = heuristic(root.ex, cache)
    # enqueue!(open_list, root, only(o))

    # MyModule.build_tree!(soltree, heuristic, open_list, close_list, new_all_symbols, sym_enc, 1000, 10, theory, cache, exp_cache, size_cache, expr_cache, 0)
    # smallest_node = MyModule.extract_smallest_terminal_node(soltree, close_list, size_cache)
    # tdata, hp, hn, proof = MyModule.extract_training_data(smallest_node, soltree)
    simplified_expression, _, big_vector, saturated, hp, hn, root, proof_vector, _ = MyModule.initialize_tree_search(heuristic, ex, 1000, 10, new_all_symbols, theory, cache, exp_cache, size_cache, expr_cache, 0.5)
    (; s₀ = MyModule.exp_size(root.ex, size_cache), sₙ = MyModule.exp_size(simplified_expression.ex, size_cache), se = simplified_expression.ex, pr = proof_vector)
end |> DataFrame
CSV.write("profile_results_trained_heuristic_$(experiment_name).csv", df)