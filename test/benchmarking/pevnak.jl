using MyModule
using MyModule.Flux
using MyModule.Mill
using MyModule.LRUCache
using MyModule.Metatheory
using Serialization
using DataFrames
using Optimisers
using Statistics
using CSV



function training_data(n=typemax(Int))
    train_data_path = "data/neural_rewrter/train.json"
    train_data = load_data(train_data_path)[10_001:70_000]
    train_data = filter(x->!occursin("select", x[1]), train_data)
    train_data = preprosses_data_to_expressions(train_data)
    train_data = sort(train_data, by=x->MyModule.exp_size(x))
    last(train_data, min(length(train_data), n))
end


function prepare_dataset(n=typemax(Int))
    samples = deserialize("../../data/training_data/size_heuristic_training_samples1.bin")
    samples = vcat(samples...)
    samples = sort(samples, by=x->MyModule.exp_size(x.initial_expr))
    # samples = sort(samples, by=x->length(x.proof))
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
    # batched_samples = get_batched_samples(samples, batch_size)
    for outer_epoch in 1:100
        for ep in 1:100
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
        # empty!(cache)
        exp_sizes = 
        samples = map(training_samples) do old_sample
            # search_tree, solution = build_search_tree

            # best_node = find_the_solution


            # minibach = create_training_sample(size_of_neigborhood(1,2,3,∞))

            simplified_expression, _, big_vector, saturated, hp, hn, _, proof_vector, _ = MyModule.interned_initialize_tree_search(model, old_sample.initial_expr, 1000, 10, new_all_symbols, sym_enc, theory)
            @show simplified_expression
            new_sample = MyModule.TrainingSample(big_vector, saturated, simplified_expression, proof_vector, hp, hn, old_sample.initial_expr)
            if MyModule.isbetter(old_sample, new_sample)
                 return(big_vector, hp, hn)
                 better_samples_counter += 1
            else
                return(old_sample)
            end
        end
        MyModule.reset_all_function_caches()
        @show (t, better_samples_counter)
        samples = new_samples
    # resolve all expressions
    # improve the current solutions
    # form the training samples
    end
end


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

heuristic = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Flux.Chain(Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, 1)),
    );
    
training_samples = prepare_dataset();
epochs = 1000

samples = [MyModule.get_training_data_from_proof_with_soltree(sample.proof, sample.initial_expr) for sample in training_samples]
samples = map(x->(MyModule.deduplicate(x[1]), x[2], x[3]), samples)
train(heuristic, samples, training_samples)

# begin
#     MyModule.reset_all_function_caches()
#     soltree = Dict{UInt64, MyModule.Node}()
#     open_list = PriorityQueue{MyModule.Node, Float32}()
#     close_list = Set{UInt64}()
#     ex = MyModule.intern!(ex)
#     root = MyModule.Node(ex, (0,0), nothing, 0, nothing)
#     o = heuristic(root.ex)
#     soltree[root.node_id] = root
#     enqueue!(open_list, root, only(o))
#     reached_goal = MyModule.interned_build_tree!(soltree, heuristic, open_list, close_list, all_symbols, symbols_to_index, 1000, 10, theory)
#     smallest_node = MyModule.extract_smallest_terminal_node(soltree, close_list)
#     big_vector, hp, hn, proof_vector, _ = extract_training_data1(smallest_node, soltree, root, 2)
# end
    
@assert 1 == 0

data = [i.initial_expr for i in training_samples]
experiment_name = "test_v1_sorted_$(length(data))_$(epochs)_hidden_size_$(hidden_size)"
serialize("models/trained_heuristic_$(experiment_name).bin", heuristic)
#heuristic = deserialize("models/trained_heuristic_testing_new_training_api_sorted_filtered_955_10_hidden_size_64.bin")
MyModule.reset_all_function_caches()
df = map(data) do ex
    simplified_expression, _, big_vector, saturated, hp, hn, root, proof_vector, _ = MyModule.interned_initialize_tree_search(heuristic, ex, 1000, 10, new_all_symbols, sym_enc, theory)
    (; s₀ = MyModule.exp_size(root.ex), sₙ = MyModule.exp_size(simplified_expression), se = simplified_expression, pr = proof_vector)
end |> DataFrame
CSV.write("profile_results_trained_heuristic_$(experiment_name).csv", df)