using MyModule
using MyModule.Flux
using MyModule.Mill
using MyModule.LRUCache
using Serialization
using DataFrames
using Optimisers
using Statistics


function prepare_dataset(n=typemax(Int))
    samples = deserialize("data/training_data/size_heuristic_training_samples1.bin")
    samples = vcat(samples...)
    samples = sort(samples, by=x->MyModule.exp_size(x.initial_expr, LRU(maxsize=10000)))
    last(samples, min(length(samples), n))
end

Heaviside(x) = x > 0 ? 1 : 0
hinge0(x::T) where {T} = x > 0 ? x : zero(T)
hinge1(x) = hinge0(x + 1)
 
function train(heuristic, training_samples, surrogate::Function, agg::Function)
    optimizer=ADAM()
    opt_state = Flux.setup(optimizer, heuristic)
    loss_stats = []
    matched_stats = []
    samples = [MyModule.get_training_data_from_proof(sample.proof, sample.initial_expr) for sample in training_samples]
    for ep in 1:epochs
        sum_loss = 0.0
        hard_loss = 0
        for (i, (ds, I₊, I₋)) in enumerate(samples)
            sa, grad = Flux.Zygote.withgradient(heuristic) do h
                o = vec(h(ds))
                agg(surrogate.(o[I₊] - o[I₋]))
            end
            sum_loss += sa
            Optimisers.update!(opt_state, heuristic, grad[1])
        end
        violations = [MyModule.loss(heuristic, sample..., Heaviside, sum) for sample in samples]
        matched_count = MyModule.heuristic_sanity_check(heuristic, training_samples, [])
        push!(matched_stats, matched_count)
        println(ep, ": ", sum_loss/length(training_samples), " matched_count: ", matched_count, " violations: ", sum(violations), " ", quantile(violations, 0:0.1:1))
    end
end


hidden_size = 128
heuristic = ExprModel(
    Flux.Chain(Dense(length(new_all_symbols), hidden_size, Flux.gelu), Dense(hidden_size,hidden_size)),
    Mill.SegmentedSum(hidden_size),
    Flux.Chain(Dense(2*hidden_size + 2, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size)),
    Flux.Chain(Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, 1))
    );


training_samples = prepare_dataset();
epochs = 1000

surrogate = hinge0
agg = sum

train(heuristic, training_samples, surrogate, agg)
