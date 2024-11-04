using Distributed
using Revise

number_of_workers = nworkers()

if number_of_workers > 1
    using Pkg
    Pkg.instantiate()
    Pkg.add("Plots")
    @everywhere using MyModule
end

using MyModule
using MyModule.Metatheory
using MyModule.Flux
using MyModule.Mill
using MyModule.DataFrames
using MyModule.LRUCache
# using MyModule.SimpleChains
using StatsBase
using CSV
using BSON
using JLD2


train_data_path = "./data/neural_rewrter/train.json"
test_data_path = "./data/neural_rewrter/test.json"


train_data = isfile(train_data_path) ? load_data(train_data_path)[1:10000] : load_data(test_data_path)[1:1000]
test_data = load_data(test_data_path)[1:1000]
train_data = preprosses_data_to_expressions(train_data)
test_data = preprosses_data_to_expressions(test_data)

hidden_size = 128
# heuristic = MyModule.ExprModelSimpleChains(ExprModel(
#     SimpleChain(static(length(new_all_symbols)), TurboDense{true}(SimpleChains.relu, hidden_size),TurboDense{true}(identity, hidden_size)),
#     Mill.SegmentedSum(hidden_size),
#     SimpleChain(static(2 * hidden_size + 2), TurboDense{true}(SimpleChains.relu, hidden_size),TurboDense{true}(identity, hidden_size)),
#     SimpleChain(static(hidden_size), TurboDense{true}(SimpleChains.relu, hidden_size),TurboDense{true}(identity, 1)),
# ))
heuristic = ExprModel(
    Flux.Chain(Dense(length(new_all_symbols), hidden_size, Flux.leakyrelu), Dense(hidden_size,hidden_size)),
    Mill.SegmentedSum(hidden_size),
    Flux.Chain(Dense(2*hidden_size + 2, hidden_size,Flux.leakyrelu), Dense(hidden_size, hidden_size)),
    Flux.Chain(Dense(hidden_size, hidden_size,Flux.leakyrelu), Dense(hidden_size, 1))
    )
pc = Flux.params([heuristic.head_model, heuristic.aggregation, heuristic.joint_model, heuristic.heuristic])

epochs = 10
optimizer = ADAM()
# global training_samples = [Vector{TrainingSample}() for _ in 1:number_of_workers]
training_samples = Vector{TrainingSample}()
max_steps = 1000
max_depth = 60
n = 1000

myex = :( (v0 + v1) + 119 <= min((v0 + v1) + 120, v2) && ((((v0 + v1) - v2) + 127) / (8 / 8) + v2) - 1 <= min(((((v0 + v1) - v2) + 134) / 16) * 16 + v2, (v0 + v1) + 119))
myex = :((v0 + v1) + 119 <= min((v0 + v1) + 120, v2))
# myex = :((v0*23 + v2) - (v2 + v0*23) <= 100) 
# myex = :(121 - max(v0 * 59, 109) <= 1024)
df = DataFrame([[], [], [], [], []], ["Epoch", "Id", "Simplified Expr", "Proof", "Length Reduced"])
df1 = DataFrame([[] for _ in 0:epochs], ["Epoch$i" for i in 0:epochs])
df2 = DataFrame([[] for _ in 0:epochs], ["Epoch$i" for i in 0:epochs])

# @load "training_samplesk1000_v3.jld2" training_samples
# x,y,r = MyModule.caviar_data_parser("data/caviar/288_dataset.json")
# x,y,r = MyModule.caviar_data_parser("data/caviar/dataset-batch-2.json")
# train_heuristic!(heuristic, train_data[1:], training_samples, max_steps, max_depth, all_symbols, theory, variable_names)
@assert 0 == 1
if isfile("../models/tre1e_search_heuristic.bson")
    # BSON.@load "../models/tree_search_heuristic.bson" heuristic
    tmp = []
elseif isfile("../data/training_data/tr2aining_samplesk1000_v4.jld2")
    # @load "../data/training_data/training_samplesk1000_v3.jld2" training_samples
    tmp = []
else
    # @load "data/training_data/training_samplesk1000_v3.jld2" training_samples

    # test_training_samples(training_samples, train_data, theory)
    stats = []
    loss_stats = []
    proof_stats = []
    stp = div(n, number_of_workers)
    batched_train_data = [train_data[i:i + stp - 1] for i in 1:stp:n]
    dt = 1
    exp_cache = LRU(maxsize=100000)
    cache = LRU(maxsize=1000000)
    size_cache = LRU(maxsize=100000)
    expr_cache = LRU(maxsize=100000)
    # global some_alpha = 1
    some_alpha = 0
    
    for ep in 1:epochs 
        empty!(cache)
        
        
        # train_heuristic!(heuristic, train_data, training_samples, max_steps, max_depth)
        println("Epoch $ep")
        println("Training===========================================")
        # global training_samples = pmap(dt -> train_heuristic!(heuristic, batched_train_data[dt], training_samples[dt], max_steps, max_depth, all_symbols, theory, variable_names, cache, exp_cache, some_alpha), collect(1:number_of_workers))
        train_heuristic!(heuristic, test_data[5:5], training_samples, max_steps, max_depth, new_all_symbols, theory, variable_names, cache, exp_cache, size_cache, expr_cache, some_alpha)
        @assert 0 == 1
        # simplified_expression, depth_dict, big_vector, saturated, hp, hn, root, proof_vector, m_nodes = MyModule.initialize_tree_search(heuristic, myex, max_steps, max_depth, all_symbols, theory, variable_names, cache, exp_cache)
        # some_alpha -= 0.1
        ltmp = []
        # @show training_samples
        # @assert 0 == 1
        # MyModule.test_expr_embedding(heuristic, training_samples[1:n], theory, symbols_to_index, all_symbols, variable_names)

        cat_samples = vcat(training_samples...)
        # for sample in training_samples
        for i in 1:100
            sample = StatsBase.sample(cat_samples)
            # for j in m_nodes
            # if isnothing(sample.training_data) 
            #     continue
            # end
            sa, grad = Flux.Zygote.withgradient(pc) do
                # heuristic_loss(heuristic, sample.training_data, sample.hp, sample.hn)
                # MyModule.loss(heuristic, big_vector, hp, hn)
                MyModule.loss(heuristic, sample.training_data, sample.hp, sample.hn)
                # MyModule.loss(heuristic, j[3], j[4], j[5])
            end
            if any(g->any(isinf, g) || any(isnan, g), grad)
                println("Gradient is Inf/NaN")
                BSON.@save "models/inf_grad_heuristic1.bson" heuristic
                JLD2.@save "data/training_data/training_samples_inf_grad1.jld2" training_samples
                @assert 0 == 1
            end
        # if isna(grad)
            @show sa
            # push!(ltmp, sa)
            Flux.update!(optimizer, pc, grad)
            # end
        end
        for sample in cat_samples
            sa = MyModule.loss(heuristic, sample.training_data, sample.hp, sample.hn)
            push!(ltmp, sa)
        end
        #
        # println("Testing===========================================")
        # # length_reduction, proofs, simp_expressions = test_heuristic(heuristic, train_data[1:n], max_steps, max_depth, variable_names, theory)
        # # push!(stats, simp_expressions)
        push!(stats, [i.expression for i in cat_samples])
        push!(proof_stats, [i.proof for i in cat_samples])
        push!(loss_stats, ltmp)
    end
    # BSON.@save "models/tree_search_heuristic.bson" heuristic
    # CSV.write("stats/data100.csv", df)
end
# @assert 0 == 1
for i in 1:length(stats[1])
    tmp = Any[myex]
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
r = "_manual_version1"
CSV.write("stats/new_theory_epoch_exp_progress$r.csv", df1)
CSV.write("stats/new_theory_epoch_proof_progress$r.csv", df2)
using Plots
plot(1:epochs, transpose(hcat(loss_stats...)))
savefig("stats/loss_new_theory$r.png")
plot()
for i in 1:size(df1)[1]
    tmp = MyModule.exp_size.(Vector(df1[i, :]))
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
