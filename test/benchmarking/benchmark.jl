using MyModule
using MyModule.Flux
using MyModule.Mill
using MyModule.DataStructures
using MyModule.LRUCache
using MyModule.Metatheory
using MyModule.Metatheory.TermInterface
using MyModule.CSV
using BenchmarkTools
using Serialization
using Profile
using SimpleChains
using DataFrames

 

hidden_size = 128
heuristic = ExprModel(
    Flux.Chain(Dense(length(new_all_symbols), hidden_size, Flux.relu), Dense(hidden_size,hidden_size)),
    Mill.SegmentedSum(hidden_size),
    Flux.Chain(Dense(2*hidden_size + 2, hidden_size,Flux.relu), Dense(hidden_size, hidden_size)),
    Flux.Chain(Dense(hidden_size, hidden_size,Flux.relu), Dense(hidden_size, 1))
    )

heuristic1 = MyModule.ExprModelSimpleChains(ExprModel(
    SimpleChain(static(length(new_all_symbols)), TurboDense{true}(SimpleChains.relu, hidden_size),TurboDense{true}(identity, hidden_size)),
    Mill.SegmentedSum(hidden_size),
    SimpleChain(static(2 * hidden_size + 2), TurboDense{true}(SimpleChains.relu, hidden_size),TurboDense{true}(identity, hidden_size)),
    SimpleChain(static(hidden_size), TurboDense{true}(SimpleChains.relu, hidden_size),TurboDense{true}(identity, 1)),
))


heuristic = MyModule.simple_to_flux(heuristic1, heuristic)


# heuristic = deserialize("models/trainied_heuristic_inf.bin")
name = "not_overfit"
# @assert 1 == 0
training_samples = deserialize("data/training_data/size_heuristic_training_samples1.bin")
training_samples = vcat(training_samples...)
training_samples = sort(training_samples, by=x->MyModule.exp_size(x.initial_expr, LRU(maxsize=10000)))
training_samples = last(training_samples, 1000)
optimizer = Adam()
stp_config = MyModule.SearchTreePipelineConfig(training_samples, heuristic, MyModule.build_tree!)
# heuristic, ireducible_stats, loss_stats = MyModule.train_heuristic_on_data_overfit(heuristic, training_samples, stp_config)

heuristic, ireducible_stats, loss_stats, matched_stats = MyModule.train_heuristic_on_data_epochs(heuristic, training_samples, stp_config)
plot(matched_stats, legend=false)
savefig("matched_stats_$(name).png")
# loss_df = DataFrame(loss_stats[1:end-1],:auto)
# CSV.write("track_loss_over_training_for_each_sample_softplus_$(name).csv", loss_df)
serialize("models/trainied_heuristic1000sorted_samples_softplus_$(name).bin", heuristic)
@assert 0 == 1
# heuristic = deserialize("models/trainied_heuristic1000sorted_samples3.bin")
heuristic1 = MyModule.flux_to_simple(heuristic1, heuristic)

# myex = 
exp_data = deserialize("data/training_data/benchmarking.bin")

test_data_path = "./data/neural_rewrter/train.json"
test_data = load_data(test_data_path)[10000:20000]
test_data = preprosses_data_to_expressions(test_data)

data  = vcat(exp_data[1], test_data[5], test_data[25:100])
max_steps = 1000
max_depth = 50
# @assert 1 == 0
function small_testing()
    exs = [:(v0 - v1), :(v2 + v3), :(v4 + v5 + v6)]
    cache = LRU{Expr,Vector}(maxsize=1000)
    MyModule.batched_cached_inference!(exs, cache, heuristic1, new_all_symbols, sym_enc)
end

function compare_two_methods(data, model, batched=64)
    cache = LRU(maxsize=50000)
    for ex in 1:batched:batched*10
        @show ex
        @time begin
            # m1 = map(x->MyModule.cached_inference!(x, cache, model, new_all_symbols, sym_enc), data[ex:ex+batched])
            o1 = map(x->only(model(x,cache)), data[ex:ex+batched])
            # o1 = map(x->MyModule.embed(model, x), m1)
        end
        @time begin
            m2 = MyModule.no_reduce_multiple_fast_ex2mill(data[ex:ex+batched], sym_enc)
            o2 = model(m2)
        end
        # @show only(o1)
        # @show only(o2)
        # map(zip(o1,o2)) do f
        # for i in zip(o1,o2)
        #     @assert abs(only(i[1]) - only(i[2])) <= 0.000001
        # end
    end
end
# use vector instead of matrix in embed
# try using cache for your expr precompute hashs as you construct your nodes

function compare_two_methods2(data, model)
    cache = LRU(maxsize=100000)
    @time begin
        tmp = map(ex->MyModule.cached_inference!(ex, cache, model, new_all_symbols, sym_enc), data)
        map(x->MyModule.embed(model, x), tmp)
    end
    # @time begin
    #     m2 = MyModule.multiple_fast_ex2mill(data, sym_enc)
    #     ds = reduce(catobs, m2)
    #     o2 = model(ds)
    # end
    @time begin
        m3 = MyModule.no_reduce_multiple_fast_ex2mill(data, sym_enc)
        o3 = model(m3)
    end
end
# compare_two_methods(exp_data, heuristic)
# compare_two_methods2(exp_data, heuristic)

function pevnaks_profile_method(data, heuristic)
    df = map(enumerate(data[1:20])) do (ex_id, ex)
        cache = LRU(maxsize=100_000)
        exp_cache = LRU{Expr, Vector}(maxsize=20000)
        training_samples = Vector{TrainingSample}()
        stats = @timed train_heuristic!(heuristic1, [ex], training_samples, 1000, 60, new_all_symbols, theory, variable_names, cache, exp_cache, 1)  
        (;time = stats.time, ic_hits = cache.hits, ic_misses = cache.misses, exp_hits = exp_cache.hits, exp_misses = exp_cache.misses, gctime = stats.gctime)
    end |> DataFrame
    CSV.write("profile_results.csv", df)
end


function pevnaks_profile_method(exp_data, heuristic; max_steps = 1000, max_depth = 30)
    df = map(enumerate(exp_data[1:20])) do (ex_id, ex)
        exp_cache = LRU{Expr, Vector}(maxsize=100_000)
        cache = LRU(maxsize=1_000_000)
        size_cache = LRU(maxsize=100_000)
        expr_cache = LRU(maxsize=100_000)
        root = MyModule.Node(ex, (0,0), nothing, 0, nothing)

        soltree = Dict{UInt64, MyModule.Node}()
        # open_list = PriorityQueue{MyModule.Node, Float32}()
        open_list = Tuple[]
        close_list = Set{UInt64}()
        expansion_history = Dict{UInt64, Vector}()
        encodings_buffer = Dict{UInt64, ProductNode}()
        println("Initial expression: $ex")
        soltree[root.node_id] = root
        push!(open_list, (root, 0))
        # o = heuristic(ex, cache)
        # enqueue!(open_list, root, only(o))
        MyModule.build_tree_with_reward_function1!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, 1)
        smallest_node = MyModule.extract_smallest_terminal_node(soltree, close_list, size_cache)
        (; s₀ = MyModule.exp_size(ex), sₙ = MyModule.exp_size(smallest_node.ex))
    end |> DataFrame
    CSV.write("profile_results.csv", df)
end


function benchmark_method_precomputed(data, heuristic, max_steps=1000, max_depth=50)
    # bmark1 = ProfileCanvas.@profview begin
    bmark1 = @benchmark begin
        ex = data[1]
        exp_cache = LRU(maxsize=100_000)
        cache = LRU(maxsize=1_000_000)
        size_cache = LRU(maxsize=100_000)
        expr_cache = LRU(maxsize=100_000)
        root = MyModule.Node(ex, (0,0), nothing, 0, expr_cache)

        soltree = Dict{UInt64, MyModule.Node}()
        open_list = PriorityQueue{MyModule.Node, Float32}()
        close_list = Set{UInt64}()
        expansion_history = Dict{UInt64, Vector}()
        encodings_buffer = Dict{UInt64, ProductNode}()
        println("Initial expression: $ex")
        soltree[root.node_id] = root
        # o = MyModule.exp_size(ex, size_cache)
        o = node_count(ex)
        enqueue!(open_list, root, only(o))
        MyModule.build_tree!(soltree, heuristic1, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, 1)
        @show length(soltree), exp_cache.hits, exp_cache.misses, cache.hits, cache.misses
    end
    bmark1 = ProfileCanvas.@profview begin
    # bmark2 = @benchmark begin
        ex = sorted_train_data[end]
        # ex = ex = :((v0 - 10) - 7 <= select(384 < v2, (min(((max(((v2 + 127) / 128) * 128, 4) + 22) / 524) * 524, max(((v2 + 127) / 128) * 128, 4) + 23) + (min(((v2 + 127) / 128) * 128, 4) + v3)) - 17, (min(((v2 + 127) / 128) * 128, 4) + v3) - 25) + (((v1 - (((v2 + 127) / 128) * 128 + v3)) + 30) / 8) * 8)
        exp_cache = LRU(maxsize=100_000)
        cache = LRU(maxsize=1_000_000)
        size_cache = LRU(maxsize=100_000)
        expr_cache = LRU(maxsize=100_000)
        root = MyModule.Node(ex, (0,0), nothing, 0, expr_cache)

        soltree = Dict{UInt64, MyModule.Node}()
        open_list = PriorityQueue{MyModule.Node, Float32}()
        close_list = Set{UInt64}()
        expansion_history = Dict{UInt64, Vector}()
        encodings_buffer = Dict{UInt64, ProductNode}()
        println("Initial expression: $ex")
        soltree[root.node_id] = root
        o = heuristic1(root.ex, cache)
        enqueue!(open_list, root, only(o))
        MyModule.build_tree!(soltree, heuristic1, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, 1)
        smallest_node = MyModule.extract_smallest_terminal_node(soltree, close_list, size_cache)
        @show length(soltree), exp_cache.hits, exp_cache.misses, cache.hits, cache.misses, length(cache)
    end
    @show bmark1, bmark2
end


function making_faster_reward(data, heuristic)
    bmark1 = @benchmark begin
        ex = data[1]
        # exp_cache = LRU{Expr, Vector}(maxsize=100_000)
        exp_cache = LRU(maxsize=100_000)
        cache = LRU(maxsize=1_000_000)
        size_cache = LRU(maxsize=100_000)
        expr_cache = LRU(maxsize=100_000)
        root = MyModule.Node(ex, (0,0), nothing, 0, expr_cache)

        soltree = Dict{UInt64, MyModule.Node}()
        # open_list = PriorityQueue{MyModule.Node, Float32}()
        open_list = Tuple[]
        close_list = Set{UInt64}()
        expansion_history = Dict{UInt64, Vector}()
        encodings_buffer = Dict{UInt64, ProductNode}()
        println("Initial expression: $ex")
        soltree[root.node_id] = root
        push!(open_list, (root, 0))
        # o = heuristic(ex, cache)
        # enqueue!(open_list, root, only(o))
        MyModule.build_tree_with_reward_function1!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, 1)
        smallest_node = MyModule.extract_smallest_terminal_node(soltree, close_list, size_cache)
    end
    bmark2 = @benchmark begin
        ex = $data[1]
        exp_cache = LRU(maxsize=100_000)
        cache = LRU(maxsize=1_000_000)
        size_cache = LRU(maxsize=100_000)
        expr_cache = LRU(maxsize=100_000)
        root = MyModule.Node(ex, (0,0), nothing, 0, nothing)

        soltree = Dict{UInt64, MyModule.Node}()
        open_list = PriorityQueue{MyModule.Node, Float32}()
        close_list = Set{UInt64}()
        expansion_history = Dict{UInt64, Vector}()
        encodings_buffer = Dict{UInt64, ProductNode}()
        println("Initial expression: $ex")
        soltree[root.node_id] = root
        o = heuristic(root.ex, cache)
        enqueue!(open_list, root, only(o))
        MyModule.build_tree_with_reward_function!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, 1, -50)
        smallest_node = MyModule.extract_smallest_terminal_node(soltree, close_list, size_cache)
    end
    @show bmark1, bmark2
end


function test_different_searches(heuristic, data; max_steps=1000,max_depth=50,experiment_name="overite")
    # data = data[1:50]
    # df = map(data) do ex
    #     exp_cache = LRU{MyModule.ExprWithHash, Vector}(maxsize=100_000)
    #     cache = LRU{MyModule.ExprWithHash, Vector}(maxsize=1_000_000)
    #     size_cache = LRU{MyModule.ExprWithHash, Int}(maxsize=100_000)
    #     expr_cache = LRU{UInt, MyModule.ExprWithHash}(maxsize=100_000)
    #     root = MyModule.Node(ex, (0,0), nothing, 0, expr_cache)

    #     soltree = Dict{UInt64, MyModule.Node}()
    #     open_list = Tuple{MyModule.Node, Float32}[]
    #     close_list = Set{UInt64}()
    #     expansion_history = Dict{UInt64, Vector}()
    #     encodings_buffer = Dict{UInt64, ProductNode}()
    #     println("Initial expression: $ex")
    #     soltree[root.node_id] = root
    #     push!(open_list, (root, 0))
    #     MyModule.build_tree_with_reward_function1!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, 1, 1)
    #     smallest_node = MyModule.extract_smallest_terminal_node(soltree, close_list, size_cache)
    #     tdata, hp, hn, proof = MyModule.extract_training_data(smallest_node, soltree)
    #     (; s₀ = MyModule.exp_size(root.ex, size_cache), sₙ = MyModule.exp_size(smallest_node.ex, size_cache), se = MyModule.reconstruct(smallest_node.ex, new_all_symbols, LRU(maxsize=1000)), pr = proof)
    # end |> DataFrame
    # CSV.write("profile_results_reward_function.csv", df)
    df = map(data) do ex
    # @benchmark begin
        exp_cache = LRU{MyModule.ExprWithHash, Vector}(maxsize=100_000)
        cache = LRU{MyModule.ExprWithHash, Vector}(maxsize=1_000_000)
        size_cache = LRU{MyModule.ExprWithHash, Int}(maxsize=100_000)
        expr_cache = LRU{UInt, MyModule.ExprWithHash}(maxsize=100_000)
        root = MyModule.Node(ex, (0,0), nothing, 0, expr_cache)

        soltree = Dict{UInt64, MyModule.Node}()
        open_list = PriorityQueue{MyModule.Node, Float32}()
        close_list = Set{UInt64}()
        expansion_history = Dict{UInt64, Vector}()
        encodings_buffer = Dict{UInt64, ProductNode}()
        println("Initial expression: $ex")
        soltree[root.node_id] = root
        o = heuristic(root.ex, cache)
        enqueue!(open_list, root, only(o))

        MyModule.build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, new_all_symbols, sym_enc, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, 0)
        smallest_node = MyModule.extract_smallest_terminal_node(soltree, close_list, size_cache)
        tdata, hp, hn, proof = MyModule.extract_training_data(smallest_node, soltree)
        (; s₀ = MyModule.exp_size(root.ex, size_cache), sₙ = MyModule.exp_size(smallest_node.ex, size_cache), se = MyModule.reconstruct(smallest_node.ex, new_all_symbols, LRU(maxsize=1000)), pr = proof)
    end |> DataFrame
    CSV.write("profile_results_trained_heuristic_$(experiment_name).csv", df)
    # df = map(data) do ex
    # # @benchmark begin
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
    #     MyModule.build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, 0.5)
    #     smallest_node = MyModule.extract_smallest_terminal_node(soltree, close_list, size_cache)
    #     (; s₀ = MyModule.exp_size(root.ex, size_cache), sₙ = MyModule.exp_size(smallest_node.ex, size_cache))
    # end |> DataFrame
    # CSV.write("profile_results_gatting_heuristic.csv", df)
    # df = map(data) do ex
    # @benchmark begin
    #     exp_cache = LRU(maxsize=100_000)
    #     cache = LRU(maxsize=1_000_000)
    #     size_cache = LRU(maxsize=100_000)
    #     expr_cache = LRU{UInt, MyModule.ExprWithHash}(maxsize=100_000)
    #     root = MyModule.Node(ex, (0,0), nothing, 0, expr_cache)

    #     soltree = Dict{UInt64, MyModule.Node}()
    #     open_list = PriorityQueue{MyModule.Node, Float32}()
    #     second_open_list = PriorityQueue{MyModule.Node, Float32}()
    #     close_list = Set{UInt64}()
    #     expansion_history = Dict{UInt64, Vector}()
    #     encodings_buffer = Dict{UInt64, ProductNode}()
    #     println("Initial expression: $ex")
    #     soltree[root.node_id] = root
    #     o = MyModule.exp_size(root.ex, size_cache)
    #     # o = node_count(root.ex.ex)
    #     enqueue!(open_list, root, only(o))
    #     MyModule.build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, 1)
    #     # reached_goal = MyModule.build_tree_with_multiple_queues!(soltree, heuristic, open_list, second_open_list, close_list, max_steps, theory, cache, exp_cache, size_cache, expr_cache, 1)
    #     smallest_node = MyModule.extract_smallest_terminal_node(soltree, close_list, size_cache)
    #     tdata, hp, hn, proof = MyModule.extract_training_data(smallest_node, soltree)
    #     (; s₀ = MyModule.exp_size(root.ex, size_cache), sₙ = MyModule.exp_size(smallest_node.ex, size_cache), se = MyModule.reconstruct(smallest_node.ex, new_all_symbols, LRU(maxsize=1000)), pr = proof)
    # end |> DataFrame
    # CSV.write("profile_results_exp_size_as_heuristic.csv", df)
    df = map(data) do ex
    # @benchmark begin
        exp_cache = LRU{MyModule.ExprWithHash, Vector}(maxsize=100_000)
        cache = LRU{MyModule.ExprWithHash, Vector}(maxsize=1_000_000)
        size_cache = LRU{MyModule.ExprWithHash, Int}(maxsize=100_000)
        expr_cache = LRU{UInt, MyModule.ExprWithHash}(maxsize=100_000)
        root = MyModule.Node(ex, (0,0), nothing, 0, expr_cache)

        soltree = Dict{UInt64, MyModule.Node}()
        open_list = PriorityQueue{MyModule.Node, Float32}()
        second_open_list = PriorityQueue{MyModule.Node, Float32}()
        close_list = Set{UInt64}()
        expansion_history = Dict{UInt64, Vector}()
        encodings_buffer = Dict{UInt64, ProductNode}()
        println("Initial expression: $ex")
        soltree[root.node_id] = root
        o = heuristic(root.ex, cache)
        enqueue!(open_list, root, only(o))
        enqueue!(second_open_list, root, MyModule.exp_size(root.ex, size_cache))
        # MyModule.build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, 1)
        reached_goal = MyModule.build_tree_with_multiple_queues!(soltree, heuristic, open_list, second_open_list, close_list, max_steps, theory, cache, exp_cache, size_cache, expr_cache, 0.5)
        smallest_node = MyModule.extract_smallest_terminal_node(soltree, close_list, size_cache)
        tdata, hp, hn, proof = MyModule.extract_training_data(smallest_node, soltree)
        (; s₀ = MyModule.exp_size(root.ex, size_cache), sₙ = MyModule.exp_size(smallest_node.ex, size_cache), se = MyModule.reconstruct(smallest_node.ex, new_all_symbols, LRU(maxsize=1000)), pr = proof)
    end |> DataFrame
    CSV.write("profile_results_multiple_queues_as_heuristic_$(experiment_name).csv", df)
end


function plot_hist_comparison(name)
    df1 = CSV.read("profile_results_exp_size_as_heuristic.csv", DataFrame)
    # df2 = CSV.read("profile_results_gatting_heuristic.csv", DataFrame)
    # df3 = CSV.read("profile_results_random_heuristic.csv", DataFrame)
    # df3 = CSV.read("profile_results_trained_heuristic_$(name).csv", DataFrame)
    # df4 = CSV.read("profile_results_reward_function.csv", DataFrame)
    # df5 = CSV.read("profile_results_multiple_queues_as_heuristic_$(name).csv", DataFrame)
    
    df1 = CSV.read("profile_results_trained_heuristic_not_overfit_128_hidden_size_1000.csv", DataFrame)
    df3 = CSV.read("profile_results_trained_heuristic_overfit_128_hidden_size_1000.csv", DataFrame)
    df4 = CSV.read("profile_results_trained_heuristic_old_heuristic_check_1000.csv", DataFrame)
    # df6 = CSV.read("profile_results_trained_heuristic_beam_search.csv", DataFrame)
    av1 = mean(df1[!, 1] .- df1[!, 2])
    av2 = 0
    av3 = mean(df3[!, 1] .- df3[!, 2])
    av4 = mean(df4[!, 1] .- df4[!, 2])
    av5 = 0
    # av6 = mean(df6[!, 1] .- df6[!, 2])
    bar(["Size Heuristic", "Gated Heuristic", "Trained Heuristic", "Reward Function", "Multiple Queues"], [av1,av2,av3,av4, av5], color=[:red, :green, :blue, :orange, :yellow], legend=false, ylabel="Average expression length reduction")
    # bar(["Size Heuristic", "Multiple Queue Heuristic", "Trained Heuristic", "Reward Function", "Beam Search"], [av1,av5,av3,av4, av6], color=[:red, :green, :blue, :orange, :yellow], legend=false, ylabel="Average expression length reduction")
end


function plot_ireducible()
    df1 = CSV.read("profile_results_exp_size_as_heuristic.csv", DataFrame)
    df2 = CSV.read("profile_results_trained_heuristic.csv", DataFrame)
    exp_args_cache = LRU(maxsize=10000)
    simp_exp_1 = Meta.parse.(df1[!, 3])
    simp_exp_2 = Meta.parse.(df2[!, 3])
    reduced_exp_1 = 0
    reduced_exp_2 = 0
    line1 = []
    line2 = []
    for (i,j) in zip(simp_exp_1, simp_exp_2)
        i_r = !MyModule.is_reducible(i, exp_args_cache)
        j_r = !MyModule.is_reducible(j, exp_args_cache)
        push!(line1, i_r)
        push!(line2, j_r)
        reduced_exp_1 += i_r
        reduced_exp_2 += j_r
    end
    plot(1:length(simp_exp_1), line1)
    # plot(1:length(simp_exp_1), line2)
    println("Reduced exp count: df1 ->$(reduced_exp_1)/$(length(simp_exp_1)); df2 ->$(reduced_exp_2)/$(length(simp_exp_2))")
end


function plot_grouped_difference(name)
    df1 = CSV.read("profile_results_exp_size_as_heuristic.csv", DataFrame)
    df2 = CSV.read("profile_results_trained_heuristic_not_overfit2.csv", DataFrame)
    df3 = CSV.read("profile_results_trained_heuristic_overfit2.csv", DataFrame)
    df1 = CSV.read("profile_results_trained_heuristic_old_heuristic_check_10000.csv", DataFrame)
    big_df = DataFrame()
    big_df[!, :s0] = df2[!, 1]
    big_df[!, :s2] = df1[!, 1] .- df1[!,2]
    big_df[!, :s1] = df2[!, 1] .- df2[!,2]
    big_df[!, :s3] = df3[!, 1] .- df3[!,2]
    combined_df = combine(groupby(big_df, :s0),              
           [:s1, :s2, :s3] .=> mean .=> [:size_heuristic, :not_overfited_heuristic, :overfitted_heuristic])
    plot_data = stack(combined_df, Not(:s0))
    rename!(plot_data, :variable => :method, :value => :performance)
    bar(plot_data.:s0, plot_data.performance, group=plot_data.method,
        xlabel="Expressions size", ylabel="Average Expression Reduction", legend=:topleft, title="1k samples")
end


function plot_exp_size_distribution(name)
    df1 = CSV.read("profile_results_trained_heuristic_not_overfit2.csv", DataFrame)
    df2 = CSV.read("profile_results_trained_heuristic_overfit_1000.csv", DataFrame)
    freq1 = countmap(df1[!, 1])
    freq2 = countmap(df2[!, 1])
    k = []
    v = []
    for (i,j) in freq1
        push!(k, i)
        push!(v, j)
    end
    bar(k, v, xlabel="Expression Size", ylabel="Number of Expressions", title="Expression Size Distribution", label="10k samples D1", color=:blue, legend=:topright)
    k = []
    v = []
    for (i,j) in freq2
        push!(k, i)
        push!(v, j)
    end
    bar!(k, v, xlabel="Expression Size", ylabel="Number of Expressions", title="Expression Size Distribution", label="1k samples D2", color=:red,legend=:topright)
    # values = keys(freq)
    # counts = collect(values .=> x -> freq[x] for x in values)
end


function test_have_learned_the_proof()
    
end

test_different_searches(heuristic1, data, experiment_name="overfit")
plot_hist_comparison()
savefig("my_search_not_overfit.png")

# test_different_searches(heuristic1, data)
# plot_hist_comparison()
# savefig("my_search_not_overfit.png")
# savefig("my_search_test_data_trained_heuristic.png")
# test_different_searches(heuristic1, [i.initial_expr for i in training_samples])
# plot_hist_comparison()
# savefig("my_search_train_data_trained_heuristic.png")
# savefig("my_search_test_data_trained_heuristic.png")
