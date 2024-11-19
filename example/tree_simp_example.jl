using Distributed
using Revise

number_of_workers = nworkers()

if number_of_workers > 1
    addprocs(number_of_workers)
    # using Pkg
    # Pkg.instantiate()
    # Pkg.add("Plots")
    @everywhere begin
        using MyModule
    #     using MyModule
    #     using MyModule
    #     using MyModule.Metatheory
    #     using MyModule.Flux
        using MyModule.Mill
    #     using MyModule.DataFrames
        using MyModule.LRUCache
        using MyModule.SimpleChains
    #     using StatsBase
    #     using CSV
    #     using BSON
    #     using JLD2
    end
end

using MyModule
using MyModule.Metatheory
using MyModule.Flux
using MyModule.Mill
using MyModule.DataFrames
using MyModule.LRUCache
using MyModule.SimpleChains
using Serialization
using StatsBase
using CSV
using BSON
using JLD2


train_data_path = "./data/neural_rewrter/train.json"
test_data_path = "./data/neural_rewrter/test.json"


train_data = isfile(train_data_path) ? load_data(train_data_path)[1:10_000] : load_data(test_data_path)[1:1000]
test_data = load_data(test_data_path)[1:1000]
train_data = preprosses_data_to_expressions(train_data)
test_data = preprosses_data_to_expressions(test_data)


@everywhere hidden_size = 128
@everywhere simple_heuristic = ExprModelSimpleChains(ExprModel(
    SimpleChain(static(length(new_all_symbols)), TurboDense{true}(SimpleChains.relu, hidden_size),TurboDense{true}(identity, hidden_size)),
    Mill.SegmentedSum(hidden_size),
    SimpleChain(static(2 * hidden_size + 2), TurboDense{true}(SimpleChains.relu, hidden_size),TurboDense{true}(identity, hidden_size)),
    SimpleChain(static(hidden_size), TurboDense{true}(SimpleChains.relu, hidden_size),TurboDense{true}(identity, 1)),
))
heuristic = ExprModel(
    Flux.Chain(Dense(length(new_all_symbols), hidden_size, Flux.leakyrelu), Dense(hidden_size,hidden_size)),
    Mill.SegmentedSum(hidden_size),
    Flux.Chain(Dense(2*hidden_size + 2, hidden_size,Flux.leakyrelu), Dense(hidden_size, hidden_size)),
    Flux.Chain(Dense(hidden_size, hidden_size,Flux.leakyrelu), Dense(hidden_size, 1))
    )
pc = Flux.params([heuristic.head_model, heuristic.aggregation, heuristic.joint_model, heuristic.heuristic])

epochs = 1
optimizer = Adam()
global training_samples = [Vector{TrainingSample}() for _ in 1:number_of_workers]
# training_samples = Vector{TrainingSample}()
max_steps = 1000
max_depth = 60
n = 100

myex = :( (v0 + v1) + 119 <= min((v0 + v1) + 120, v2) && ((((v0 + v1) - v2) + 127) / (8 / 8) + v2) - 1 <= min(((((v0 + v1) - v2) + 134) / 16) * 16 + v2, (v0 + v1) + 119))
# myex = :((v0 + v1) + 119 <= min((v0 + v1) + 120, v2))
# myex = :((v0*23 + v2) - (v2 + v0*23) <= 100) 
# myex = :(121 - max(v0 * 59, 109) <= 1024)
df = DataFrame([[], [], [], [], []], ["Epoch", "Id", "Simplified Expr", "Proof", "Length Reduced"])
df1 = DataFrame([[] for _ in 0:epochs], ["Epoch$i" for i in 0:epochs])
df2 = DataFrame([[] for _ in 0:epochs], ["Epoch$i" for i in 0:epochs])
heuristic = MyModule.simple_to_flux(simple_heuristic, heuristic)

# @load "training_samplesk1000_v3.jld2" training_samples
# x,y,r = MyModule.caviar_data_parser("data/caviar/288_dataset.json")
# x,y,r = MyModule.caviar_data_parser("data/caviar/dataset-batch-2.json")
# train_heuristic!(heuristic, train_data[1:], training_samples, max_steps, max_depth, all_symbols, theory, variable_names)


# @assert 0 == 1
stats = []
loss_stats = []
proof_stats = []
stp = div(n, number_of_workers)
# sorted_train_data = create_batches_varying_length(train_data, epochs)
# batched_train_data = [sorted_train_data[1][i:i + stp - 1] for i in 1:stp:n]
batched_train_data = [train_data[i:i + stp - 1] for i in 1:stp:n]
dt = 1
# exp_cache = LRU(maxsize=100_000)
# cache = LRU(maxsize=1_000_000)
# size_cache = LRU(maxsize=100_000)
# expr_cache = LRU(maxsize=100_000)
# global some_alpha = 1
some_alpha = 0.05
# @everywhere max_steps = $max_steps
# @everywhere max_depth = $max_depth
# @everywhere batched_train_data = $batched_train_data
@everywhere exp_cache = LRU(maxsize=100_000)
@everywhere cache = LRU(maxsize=1_000_000)
@everywhere size_cache = LRU(maxsize=100_000)
@everywhere expr_cache = LRU(maxsize=100_000)
# @everywhere training_samples = $training_samples
# @everywhere simple_heuristic = $simple_heuristic
# @everywhere some_alpha = $some_alpha
# results = RemoteChannel(() -> Channel{TrainingSample}(number_of_workers))
# results = SharedVector{Vector{TrainingSample}}(number_of_workers)
for ep in 1:epochs 
    empty!(cache)
    
    # batched_train_data = [sorted_train_data[ep][i:i + stp - 1] for i in 1:stp:n]
    
    # train_heuristic!(heuristic, train_data, training_samples, max_steps, max_depth)
    println("Epoch $ep")
    println("Training===========================================")
    global training_samples = pmap(dt -> train_heuristic!(simple_heuristic, batched_train_data[dt], training_samples[dt], max_steps, max_depth, all_symbols, theory, variable_names, cache, exp_cache, size_cache, expr_cache, 1 - (ep - 1) * 0.1), collect(1:number_of_workers))
    # train_heuristic!(simple_heuristic, train_data, training_samples, max_steps, max_depth, new_all_symbols, theory, variable_names, cache, exp_cache, size_cache, expr_cache, some_alpha)
    # cat_samples = vcat(training_samples...)
    # batched_train_data = [i.expression.ex for i in cat_samples]
    # batched_train_data = [batched_train_data[i:i + stp - 1] for i in 1:stp:n]
    # push!(stats, [i.expression.ex for i in cat_samples])
    # push!(proof_stats, [i.proof for i in cat_samples])
    # serialize("data/training_data/size_heuristic_training_samples.bin", training_samples)
    # continue
    # @assert 0 == 1

    # simplified_expression, depth_dict, big_vector, saturated, hp, hn, root, proof_vector, m_nodes = MyModule.initialize_tree_search(heuristic, myex, max_steps, max_depth, all_symbols, theory, variable_names, cache, exp_cache)
    # some_alpha += 0.1
    ltmp = []
    # @show training_samples
    # @assert 0 == 1
    # MyModule.test_expr_embedding(heuristic, training_samples[1:n], theory, symbols_to_index, all_symbols, variable_names)
    # training_samples = fetch(results)
    cat_samples = vcat(training_samples...)
    cat_samples = sort(cat_samples, by=x->MyModule.exp_size(MyModule.ExprWithHash(x.initial_expr, expr_cache), size_cache))
    # cat_samples = filter(x->length(x.proof)>1, cat_samples)
    for _ in 1:10
        for sample in cat_samples
        # for i in 1:10000
        #     sample = StatsBase.sample(cat_samples)
        # for (ind,i) in enumerate(training_samples)
            # bd, hp, hn = MyModule.get_training_data_from_proof(i.proof, i.initial_expr)
        #     training_samples[ind].training_data = 
        # end
        # for (ind,sample) in enumerate(training_samples[1:100])
        #     bd, hp, hn = MyModule.get_training_data_from_proof(sample.proof, sample.initial_expr)
        #     # @show size(hp)
        #     for _ in 1:10
                # training_data = get_training_data_from_proof(sample.proof)
                # for j in m_nodes
                # if isnothing(sample.training_data) 
                #     continue
                # end
            sa, grad = Flux.Zygote.withgradient(pc) do
                # heuristic_loss(heuristic, sample.training_data, sample.hp, sample.hn)
                # MyModule.loss(heuristic, big_vector, hp, hn)
                # MyModule.loss(heuristic, bd, hp, hn)
                MyModule.loss(heuristic, sample.training_data, sample.hp, sample.hn)
                # MyModule.loss(heuristic, j[3], j[4], j[5])
            end
            # @show grad
            @show sa
            if any(g->any(isinf, g) || any(isnan, g), grad)
                println("Gradient is Inf/NaN")
                BSON.@save "models/inf_grad_heuristic1.bson" heuristic
                JLD2.@save "data/training_data/training_samples_inf_grad1.jld2" training_samples
                @assert 0 == 1
            end
        # if isna(grad)
            # push!(ltmp, sa)
            Flux.update!(optimizer, pc, grad)
        end
    end
    # end
    for sample in cat_samples
        sa = MyModule.loss(heuristic, sample.training_data, sample.hp, sample.hn)
        push!(ltmp, sa)
    end
    #
    # println("Testing===========================================")
    # # length_reduction, proofs, simp_expressions = test_heuristic(heuristic, train_data[1:n], max_steps, max_depth, variable_names, theory)
    # # push!(stats, simp_expressions)
    push!(stats, [i.expression.ex for i in cat_samples])
    push!(proof_stats, [i.proof for i in cat_samples])
    push!(loss_stats, ltmp)
    global simple_heuristic = MyModule.flux_to_simple(simple_heuristic, heuristic)
end
# BSON.@save "models/tree_search_heuristic.bson" heuristic
# CSV.write("stats/data100.csv", df)
# @assert 0 == 1
serialize("models/new_trained__gatted_heuristic.bin", heuristic)
serialize("data/training_data/size_heuristic_training_samples.bin", training_samples)
for i in 1:length(stats[1])
    tmp = Any[train_data[i]]
    for j in 1:epochs
        push!(tmp, stats[j][i])
    end
    push!(df1, tmp)
end
for i in 1:length(proof_stats[1])
    tmp = [[]]
    for j in 1:epochs
        push!(tmp, proof_stats[j][i])
    end
    push!(df2, tmp)
end
r = "_manual_version"
CSV.write("stats/new_theory_epoch_exp_progress$r.csv", df1)
CSV.write("stats/new_theory_epoch_proof_progress$r.csv", df2)
using Plots
# plot(1:epochs, transpose(hcat(loss_stats...)))
# savefig("stats/loss_new_theory$r.png")
plot()
for i in 1:size(df1)[1]
    tmp = map(x->MyModule.exp_size(x, size_cache), Vector(df1[i, :]))
    plot!(0:size(df1)[2] - 1, tmp)
end
plot!()
savefig("stats/new_theory_expr_size_progress$r.png")
plot()
for i in 1:size(df2)[1]
    tmp = length.(Vector(df2[i, :]))
    plot!(0:size(df2)[2] - 1, tmp)
end
plot!()
savefig("stats/new_theory_proof_progress$r.png")
# avg_proof_size = [mean([length(i.args) - 1 for i in Meta.parse.(df2[!,j])]) for j in 1:6]