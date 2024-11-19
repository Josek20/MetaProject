using MyModule
using MyModule.Flux
using MyModule.Mill
using MyModule.DataStructures
using MyModule.LRUCache
using MyModule.Metatheory
using MyModule.CSV
using BenchmarkTools
using Serialization
using Profile
using PProf
using DataFrames
using ProfileCanvas
using SimpleChains
using Plots



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


training_samples = deserialize("data/training_data/size_heuristic_training_samples1.bin")
training_samples = vcat(training_samples...)
training_samples = sort(training_samples, by=x->MyModule.exp_size(x.initial_expr, LRU(maxsize=1000)))
training_samples = vcat(training_samples[1:50], training_samples[400:449], training_samples[800:end])
pc = Flux.params([heuristic.head_model, heuristic.aggregation, heuristic.joint_model, heuristic.heuristic])    
for (ind,sample) in enumerate(training_samples)
    bd, hp, hn = MyModule.get_training_data_from_proof(sample.proof, sample.initial_expr)
    @show ind
    for _ in 1:10
        sa, grad = Flux.Zygote.withgradient(pc) do
            # heuristic_loss(heuristic, sample.training_data, sample.hp, sample.hn)
            # MyModule.loss(heuristic, big_vector, hp, hn)
            MyModule.loss(heuristic, bd, hp, hn)
            # MyModule.loss(heuristic, j[3], j[4], j[5])
        end
        # @show grad
        @show sa
        if any(g->any(isinf, g) || any(isnan, g), grad)
            println("Gradient is Inf/NaN")
            # BSON.@save "models/inf_grad_heuristic1.bson" heuristic
            # JLD2.@save "data/training_data/training_samples_inf_grad1.jld2" training_samples
            @assert 0 == 1
        end
        Flux.update!(optimizer, pc, grad)
    end
end

# heuristic = MyModule.train_heuristic_on_data(heuristic, training_samples)
heuristic1 = MyModule.flux_to_simple(heuristic1, heuristic)




exp_data = deserialize("data/training_data/benchmarking.bin")

test_data_path = "./data/neural_rewrter/test.json"
test_data = load_data(test_data_path)[1:1000]
test_data = preprosses_data_to_expressions(test_data)

data  = vcat(exp_data[1], test_data[5], test_data[25:100])
max_steps = 1000
max_depth = 50

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

function profile_method(data, heuristic)
    ex = :((v0 + v1) + 119 <= min((v0 + v1) + 120, v2))
    # ex = train_data[1]
    # ex = myex
    # ex = :((v0 + v1) * 119 + (v3 + v7) <= (v0 + v1) * 119 + ((v2 * 30 + ((v3 * 4 + v4) + v5)) + v7))
    # ex =  :((v0 + v1) * 119 + (v3 + v7) <= (v0 + v1) * 119 + (((v3 * (4 + v2 / (30 / v3)) + v5) + v4) + v7))
    # encoded_ex = MyModule.ex2mill(ex, symbols_to_index, all_symbols, variable_names)
    # encoded_ex = MyModule.single_fast_ex2mill(ex, MyModule.sym_enc)
    ex = :((v0 + v1) + 119 <= min(120 + (v0 + v1), v2) && min(((((v0 + v1) - v2) + 127) / 8) * 8 + v2, (v0 + v1) + 120) - 1 <= ((((v0 + v1) - v2) + 134) / 16) * 16 + v2)

    # root = MyModule.Node(ex, (0,0), hash(ex), 0, nothing)
    
    # soltree = Dict{UInt64, MyModule.Node}()
    # open_list = PriorityQueue{MyModule.Node, Float32}()
    # close_list = Set{UInt64}()
    # expansion_history = Dict{UInt64, Vector}()
    # encodings_buffer = Dict{UInt64, ProductNode}()
    # @show ex
    # soltree[root.node_id] = root
    # a = MyModule.cached_inference!(ex,cache,heuristic, new_all_symbols, sym_enc)
    # hp = MyModule.embed(heuristic, a)
    # enqueue!(open_list, root, only(hp))
    # enqueue!(open_list, root, only(heuristic(root.expression_encoding)))
    # ProfileCanvas.@profview MyModule.build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache)
    cache = LRU(maxsize=100000)
    exp_cache = LRU{Expr, Vector}(maxsize=20000)
    training_samples = Vector{TrainingSample}()

    ProfileCanvas.@profview train_heuristic!(heuristic, [ex], training_samples, 1000, 60, new_all_symbols, theory, variable_names, cache, exp_cache, 1)  
    @benchmark begin 
        cache = LRU(maxsize=100_000)
        exp_cache = LRU{Expr, Vector}(maxsize=20000)
        training_samples = Vector{TrainingSample}()
        train_heuristic!(heuristic, [ex], training_samples, 1000, 60, new_all_symbols, theory, variable_names, cache, exp_cache, 1)  
    end
    # @benchmark train_heuristic!(heuristic, [ex], training_samples, 1000, 60, new_all_symbols, theory, variable_names, cache, exp_cache)  
    # @time MyModule.build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, 1000, 60, expansion_history, theory, variable_names, cache, exp_cache)
    # @show length(soltree)
    # @show cache.hits
    # @show cache.misses
    # @show exp_cache.hits
    # @show exp_cache.misses
    # @profile peakflops()
    # pprof()
end


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
    bmark1 = ProfileCanvas.@profview begin
    # bmark1 = @benchmark begin
        ex = data[1]
        exp_cache = LRU{Expr, Vector}(maxsize=100_000)
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
        o = heuristic1(ex, cache)
        enqueue!(open_list, root, only(o))
        MyModule.build_tree!(soltree, heuristic1, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, 1)
        @show length(soltree), exp_cache.hits, exp_cache.misses, cache.hits, cache.misses
    end
    # bmark1 = ProfileCanvas.@profview begin
    bmark2 = @benchmark begin
        ex = data[4]
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
# (ind, pl, r) = (11, [3, 2, 2, 2], 4)



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


function test_different_searches(heuristic, data, max_steps=1000,max_depth=50)
    data = data[1:50]
    df = map(data) do ex
        exp_cache = LRU{MyModule.ExprWithHash, Vector}(maxsize=100_000)
        cache = LRU{MyModule.ExprWithHash, Vector}(maxsize=1_000_000)
        size_cache = LRU{MyModule.ExprWithHash, Int}(maxsize=100_000)
        expr_cache = LRU{UInt, MyModule.ExprWithHash}(maxsize=100_000)
        root = MyModule.Node(ex, (0,0), nothing, 0, expr_cache)

        soltree = Dict{UInt64, MyModule.Node}()
        open_list = Tuple{MyModule.Node, Float32}[]
        close_list = Set{UInt64}()
        expansion_history = Dict{UInt64, Vector}()
        encodings_buffer = Dict{UInt64, ProductNode}()
        println("Initial expression: $ex")
        soltree[root.node_id] = root
        push!(open_list, (root, 0))
        MyModule.build_tree_with_reward_function1!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, 1, 1)
        smallest_node = MyModule.extract_smallest_terminal_node(soltree, close_list, size_cache)
        (; s₀ = MyModule.exp_size(root.ex, size_cache), sₙ = MyModule.exp_size(smallest_node.ex, size_cache), se = smallest_node.ex.ex)
    end |> DataFrame
    # mean(df[!,1] .- df[!,2])
    CSV.write("profile_results_reward_function.csv", df)
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
        (; s₀ = MyModule.exp_size(root.ex, size_cache), sₙ = MyModule.exp_size(smallest_node.ex, size_cache))
    end |> DataFrame
    CSV.write("profile_results_random_heuristic.csv", df)
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
        MyModule.build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, 0.5)
        smallest_node = MyModule.extract_smallest_terminal_node(soltree, close_list, size_cache)
        (; s₀ = MyModule.exp_size(root.ex, size_cache), sₙ = MyModule.exp_size(smallest_node.ex, size_cache))
    end |> DataFrame
    CSV.write("profile_results_gatting_heuristic.csv", df)
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
        MyModule.build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, 1)
        # reached_goal = MyModule.build_tree_with_multiple_queues!(soltree, heuristic, open_list, second_open_list, close_list, max_steps, theory, cache, exp_cache, size_cache, expr_cache, 1)
        smallest_node = MyModule.extract_smallest_terminal_node(soltree, close_list, size_cache)
        (; s₀ = MyModule.exp_size(root.ex, size_cache), sₙ = MyModule.exp_size(smallest_node.ex, size_cache), se = MyModule.reconstruct(smallest_node.ex, new_all_symbols, LRU(maxsize=1000)))
    end |> DataFrame
    CSV.write("profile_results_depth_as_heuristic.csv", df)
    # df = map(data) do ex
    # # @benchmark begin
    #     exp_cache = LRU{MyModule.ExprWithHash, Vector}(maxsize=100_000)
    #     cache = LRU{MyModule.ExprWithHash, Vector}(maxsize=1_000_000)
    #     size_cache = LRU{MyModule.ExprWithHash, Int}(maxsize=100_000)
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
    #     o = heuristic(root.ex, cache)
    #     enqueue!(open_list, root, only(o))
    #     # MyModule.build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache, size_cache, expr_cache, 1)
    #     reached_goal = MyModule.build_tree_with_multiple_queues!(soltree, heuristic, open_list, second_open_list, close_list, max_steps, theory, cache, exp_cache, size_cache, expr_cache, 0.5)
    #     smallest_node = MyModule.extract_smallest_terminal_node(soltree, close_list, size_cache)
    #     (; s₀ = MyModule.exp_size(root.ex, size_cache), sₙ = MyModule.exp_size(smallest_node.ex, size_cache), se = MyModule.reconstruct(smallest_node.ex, new_all_symbols, LRU(maxsize=1000)))
    # end |> DataFrame
    # CSV.write("profile_results_multiple_queues_as_heuristic.csv", df)
end


function plot_hist_comparison()
    df1 = CSV.read("profile_results_depth_as_heuristic.csv", DataFrame)
    df2 = CSV.read("profile_results_gatting_heuristic.csv", DataFrame)
    df3 = CSV.read("profile_results_random_heuristic.csv", DataFrame)
    df4 = CSV.read("profile_results_reward_function.csv", DataFrame)
    # df5 = CSV.read("profile_results_multiple_queues_as_heuristic.csv", DataFrame)
    av1 = mean(df1[!, 1] .- df1[!, 2])
    av2 = mean(df2[!, 1] .- df2[!, 2])
    av3 = mean(df3[!, 1] .- df3[!, 2])
    av4 = mean(df4[!, 1] .- df4[!, 2])
    # av5 = mean(df5[!, 1] .- df5[!, 2])
    # bar(["Size Heuristic", "Gated Heuristic", "Trained Heuristic", "Reward Function", "Multiple Queues"], [av1,av2,av3,av4, av5], color=[:red, :green, :blue, :orange, :yellow], legend=false, ylabel="Average expression length reduction")
    bar(["Size Heuristic", "Gated Heuristic", "Trained Heuristic", "Reward Function"], [av1,av2,av3,av4], color=[:red, :green, :blue, :orange], legend=false, ylabel="Average expression length reduction")
end

test_different_searches(heuristic1, data)
plot_hist_comparison()
savefig("my_search_test_data_trained_heuristic.png")