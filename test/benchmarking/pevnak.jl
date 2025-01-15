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
    samples = deserialize("../../data/training_data/size_heuristic_training_samples1.bin")
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


function train(heuristic, training_samples, surrogate::Function, agg::Function, batch_size=10)
    optimizer=ADAM()
    opt_state = Flux.setup(optimizer, heuristic)
    loss_stats = []
    matched_stats = []
    samples = [MyModule.get_training_data_from_proof(sample.proof, sample.initial_expr) for sample in training_samples]
    batched_samples = get_batched_samples(samples, batch_size)
    for ep in 1:epochs
        sum_loss = 0.0
        hard_loss = 0
        for (i, (ds, I₊, I₋)) in enumerate(batched_samples)
            sa, grad = Flux.Zygote.withgradient(heuristic) do h
                o = vec(h(ds))
                agg(surrogate.(o[I₊] - o[I₋]))
            end
            sum_loss += sa
            Optimisers.update!(opt_state, heuristic, grad[1])
        end
        violations = [MyModule.loss(heuristic, sample..., Heaviside, sum) for sample in batched_samples]
        matched_count, improved_count, new_proof_count, not_matched_count = MyModule.heuristic_sanity_check(heuristic, training_samples, [])
        @show matched_count
        @show improved_count
        @show new_proof_count
        @show not_matched_count
        push!(matched_stats, matched_count)
        println(ep, ": ", sum_loss/length(training_samples), " matched_count: ", matched_count, " violations: ", sum(violations), " ", quantile(violations, 0:0.1:1))
    end
end


hidden_size = 128

head_model = ProductModel(
    (;head = ArrayModel(Flux.Chain(Dense(length(new_all_symbols), hidden_size, Flux.gelu), Dense(hidden_size,hidden_size))),
      args = Flux.Chain(Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size,hidden_size)),  
        ),
    Flux.Chain(Dense(2*hidden_size, hidden_size, Flux.gelu), Dense(hidden_size,hidden_size))
    )

args_model = ProductModel(
    (;args = Flux.Chain(Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size,hidden_size)),  
      position = ArrayModel(Dense(2,hidden_size)),  
        ),
    Flux.Chain(Dense(2*hidden_size, hidden_size, Flux.gelu), Dense(hidden_size,hidden_size))
    )

heuristic = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Flux.Chain(Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, 1)),
    );

# heuristic = ExprModel(
#     Flux.Chain(Dense(length(new_all_symbols), hidden_size, Flux.gelu)),
#     Mill.SegmentedSum(hidden_size),
#     Flux.Chain(Dense(2*hidden_size + 2, hidden_size, Flux.gelu)),
#     Flux.Chain(Dense(hidden_size, 1)),
#     );
    
training_samples = prepare_dataset(10);
epochs = 1000

surrogate = softplus
agg = sum

samples = [MyModule.get_training_data_from_proof(sample.proof, sample.initial_expr) for sample in training_samples]
ds = samples[end][1]
m = heuristic


train(heuristic, training_samples, surrogate, agg)
# data = [i.initial_expr for i in training_samples]
# experiment_name = "testing_new_training_api_$(length(data))
# df = map(data) do ex
#     # @benchmark begin
#     exp_cache = LRU{MyModule.ExprWithHash, Vector}(maxsize=100_000)
#     cache = LRU{MyModule.ExprWithHash, Vector}(maxsize=1_000_000)
#     size_cache = LRU{MyModule.ExprWithHash, Int}(maxsize=100_000)
#     expr_cache = LRU{UInt, MyModule.ExprWithHash}(maxsize=100_000)
#     root = MyModule.Node(ex, (0,0), nothing, 0, expr_cache)

#     soltree = Dict{UInt64, MyModule.Node}()
#     open_list = PriorityQueue{MyModule.Node, Float32}()
#     close_list = Set{UInt64}()
#     expansion_history = Dict{UInt64, Vector}()
#     encodings_buffer = Dict{UInt64, ProductNode}()
#     println("Initial expression: $ex")
#     soltree[root.node_id] = root
#     o = heuristic(root.ex, cache)
#     enqueue!(open_list, root, only(o))

#     MyModule.build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, new_all_symbols, sym_enc, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, 0)
#     smallest_node = MyModule.extract_smallest_terminal_node(soltree, close_list, size_cache)
#     tdata, hp, hn, proof = MyModule.extract_training_data(smallest_node, soltree)
#     (; s₀ = MyModule.exp_size(root.ex, size_cache), sₙ = MyModule.exp_size(smallest_node.ex, size_cache), se = MyModule.reconstruct(smallest_node.ex, new_all_symbols, LRU(maxsize=1000)), pr = proof)
# end |> DataFrame
# CSV.write("profile_results_trained_heuristic_$(experiment_name).csv", df)