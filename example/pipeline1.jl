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
using MyModule.Metatheory.TermInterface
using MyModule.Flux
using MyModule.Mill
using MyModule.DataFrames
using MyModule.LRUCache
using MyModule.SimpleChains
using Serialization
using StatsBase
using CSV


train_data_path = "./data/neural_rewrter/train.json"
test_data_path = "./data/neural_rewrter/test.json"


train_data = isfile(train_data_path) ? load_data(train_data_path)[10_001:70_000] : load_data(test_data_path)[1:1000]
test_data = load_data(test_data_path)[1:1000]
train_data = preprosses_data_to_expressions(train_data)
test_data = preprosses_data_to_expressions(test_data)
size_cache = LRU(maxsize=10000)
sorted_train_data = sort(train_data, by=x->MyModule.exp_size(x,size_cache))

hidden_size = 128
simple_heuristic = ExprModelSimpleChains(ExprModel(
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
    
epochs = 4
optimizer = Adam()
# global training_samples = [Vector{TrainingSample}() for _ in 1:number_of_workers]
max_steps = 1000
max_depth = 60
n = 100
if "trainied_heuristic1000samples1341234.bin" in readdir("models/") 
    heuristic = deserialize("models/trainied_heuristic1000samples.bin")
    simple_heuristic = MyModule.flux_to_simple(simple_heuristic, heuristic)
else
    heuristic = MyModule.simple_to_flux(simple_heuristic, heuristic)
end
pc = Flux.params([heuristic.head_model, heuristic.aggregation, heuristic.joint_model, heuristic.heuristic])

@assert 0 == 1
stats = []
loss_stats = []
proof_stats = []
# stp = div(n, number_of_workers)
# batched_train_data = [train_data[i:i + stp - 1] for i in 1:stp:n]

if "size_heuristic_training_samples123.bin" in readdir("data/training_data") 
    training_samples = deserialize("data/training_data/size_heuristic_training_samples1.bin")
    training_samples = vcat(training_samples...)[1:10]
else
    training_samples = TrainingSample(last(train_data,10))
end
dt = 1
some_alpha = 0.05
if number_of_workers > 1
    stp = div(length(training_samples), number_of_workers)
    batched_training_samples = [training_samples[i:i + stp - 1] for i in 1:stp:length(training_samples)]
    list_of_configs = [MyModule.SearchTreePipelineConfig(batched_training_samples[i], simple_heuristic, MyModule.build_tree!) for i in 1:number_of_workers]
else
    stp_config = MyModule.SearchTreePipelineConfig(training_samples, MyModule.exp_size, MyModule.build_tree!)
end
@assert 1 == 0
improve_samples_stats = []
reducible_samples_stats = []
# push!(improve_samples_stats, zeros(length(training_samples)))
for ep in 1:epochs 
    println("Epoch $ep")
    println("Training===========================================")
    if number_of_workers > 1
        for config in list_of_configs
            empty!(config.inference_cache)
        end
        samples_stats = pmap(dt -> train_heuristic!(list_of_configs[dt]), collect(1:number_of_workers))
        push!(improve_samples_stats, sum(st[2] for st in samples_stats))
        training_data = vcat([st_conf.training_data for st_conf in list_of_configs]...)
        sorted_samples = sort(training_data, by=x->node_count(x.initial_expr))
    else
        empty!(stp_config.inference_cache)
        number_of_reducible, number_of_improved = train_heuristic!(stp_config)
        push!(improve_samples_stats, number_of_improved)
        @show number_of_reducible, number_of_improved
        # training_data = stp_config.training_data
        sorted_samples = sort(stp_config.training_data, by=x->node_count(x.initial_expr))
    end
    ltmp = []
    for _ in 1:10
        for sample in sorted_samples
            sa, grad = Flux.Zygote.withgradient(pc) do
                MyModule.loss(heuristic, sample.training_data, sample.hp, sample.hn)
            end
            @show sa
            if any(g->any(isinf, g) || any(isnan, g), grad)
                println("Gradient is Inf/NaN")
                BSON.@save "models/inf_grad_heuristic1.bson" heuristic
                JLD2.@save "data/training_data/training_samples_inf_grad1.jld2" training_samples
                @assert 0 == 1
            end
            Flux.update!(optimizer, pc, grad)
        end
    end
    # for sample in cat_samples
    #     sa = MyModule.loss(heuristic, sample.training_data, sample.hp, sample.hn)
    #     push!(ltmp, sa)
    # end
    # push!(stats, [i.expression.ex for i in cat_samples])
    # push!(proof_stats, [i.proof for i in cat_samples])
    # push!(loss_stats, ltmp)
    simple_heuristic = MyModule.flux_to_simple(simple_heuristic, heuristic)
    if number_of_workers > 1
        for config in list_of_configs
            config.heuristic = simple_heuristic
        end
    end
end
plot(1:epochs, improve_samples_stats)
savefig("improve_samples_progression.png")
@assert 1 == 0
serialize("models/updated_trained_heuristic_softplus.bin", heuristic)
serialize("data/training_data/updated_heuristic_training_samples.bin", training_samples)
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