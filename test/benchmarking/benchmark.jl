using MyModule
using MyModule.Flux
using MyModule.Mill
using MyModule.DataStructures
using MyModule.LRUCache
using BenchmarkTools
using Serialization
using Profile
using PProf
using ProfileCanvas
using SimpleChains





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

exp_data = Vector()
open("./data/training_data/benchmarking.bin", "r") do file
    data = deserialize(file)
    append!(exp_data, data)
end

test_data_path = "./data/neural_rewrter/test.json"
test_data = load_data(test_data_path)
test_data = preprosses_data_to_expressions(test_data)



function compare_two_methods(data, model, batched=64)
    cache = LRU(maxsize=50000)
    for ex in 1:batched:batched*10
        @show ex
        anodes = map(x->MyModule.Node(ex, (0,0), ex, 0, nothing), data[ex:ex+batched])
        # anodes = map(x->MyModule.Node(ex, (0,0), hash(ex), 0, nothing), data[ex:ex+batched])
        @time begin
            # m1 = map(x->MyModule.cached_inference!(x, cache, model, new_all_symbols, sym_enc), data[ex:ex+batched])
            # o1 = map(x->only(model(x,cache)), data[ex:ex+batched])
            o1 = map(x->only(model(x.ex,cache)), anodes)
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
compare_two_methods(exp_data, heuristic)
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
    cache = LRU(maxsize=100000)
    exp_cache = LRU{Expr, Vector}(maxsize=20000)
    # a = MyModule.cached_inference!(ex,cache,heuristic, new_all_symbols, sym_enc)
    # hp = MyModule.embed(heuristic, a)
    # enqueue!(open_list, root, only(hp))
    # enqueue!(open_list, root, only(heuristic(root.expression_encoding)))
    # ProfileCanvas.@profview MyModule.build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names, cache, exp_cache)
    training_samples = Vector{TrainingSample}()
    # ProfileCanvas.@profview train_heuristic!(heuristic, [ex], training_samples, 1000, 60, new_all_symbols, theory, variable_names, cache, exp_cache)  
    @benchmark train_heuristic!(heuristic, [ex], training_samples, 1000, 60, new_all_symbols, theory, variable_names, cache, exp_cache)  
    # @time MyModule.build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, 1000, 60, expansion_history, theory, variable_names, cache, exp_cache)
    # @show length(soltree)
    # @show cache.hits
    # @show cache.misses
    # @show exp_cache.hits
    # @show exp_cache.misses
    # @profile peakflops()
    # pprof()
end

# profile_method(exp_data, heuristic) 
