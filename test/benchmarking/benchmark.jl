using MyModule
using MyModule.Flux
using MyModule.Mill
using MyModule.DataStructures
using Serialization
using Profile
using PProf
using ProfileCanvas


exp_data = Vector()
open("./data/training_data/benchmarking.bin", "r") do file
    data = deserialize(file)
    append!(exp_data, data)
end

test_data_path = "./data/neural_rewrter/test.json"
test_data = load_data(test_data_path)
test_data = preprosses_data_to_expressions(test_data)



hidden_size = 128
heuristic = ExprModel(
    Flux.Chain(Dense(length(symbols_to_index) + 1 + 13, hidden_size, relu), Dense(hidden_size,hidden_size)),
    Mill.SegmentedSum(hidden_size),
    Flux.Chain(Dense(2*hidden_size + 2, hidden_size,relu), Dense(hidden_size, hidden_size)),
    Flux.Chain(Dense(hidden_size, hidden_size,relu), Dense(hidden_size, 1))
    )


function compare_two_methods(data, model)
    cache = Dict()
    for ex in data[1:100]
        @show ex
        @time begin
            m1 = MyModule.cached_inference!(ex, cache, model, new_all_symbols, sym_enc)
            o1 = MyModule.embed(model, m1)
        end
        @time begin
            m2 = MyModule.single_fast_ex2mill(ex, sym_enc)
            o2 = model(m2)
        end
        @show only(o1)
        @show only(o2)
        @assert abs(only(o1) - only(o2)) <= 0.000001
    end
end


function compare_two_methods2(data, model)
    cache = Dict()
    @time begin
        tmp = map(ex->MyModule.cached_inference!(ex, cache, model, new_all_symbols, sym_enc), data)
        map(x->MyModule.embed(model, x), tmp)
    end
    @time begin
        m2 = MyModule.multiple_fast_ex2mill(data, sym_enc)
        ds = reduce(catobs, m2)
        o2 = model(ds)
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
    root = MyModule.Node(ex, (0,0), hash(ex), 0, nothing)
    
    soltree = Dict{UInt64, MyModule.Node}()
    open_list = PriorityQueue{MyModule.Node, Float32}()
    close_list = Set{UInt64}()
    expansion_history = Dict{UInt64, Vector}()
    encodings_buffer = Dict{UInt64, ProductNode}()
    @show ex
    soltree[root.node_id] = root
    cache = Dict()
    a = MyModule.cached_inference(ex,cache,heuristic, new_all_symbols, sym_enc)
    hp = MyModule.embed(heuristic, a)
    enqueue!(open_list, root, only(hp))
    # enqueue!(open_list, root, only(heuristic(root.expression_encoding)))
    # ProfileCanvas.@profview MyModule.build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, max_steps, max_depth, expansion_history, theory, variable_names)

    @time MyModule.build_tree!(soltree, heuristic, open_list, close_list, encodings_buffer, all_symbols, symbols_to_index, 1000, 100, expansion_history, theory, variable_names, cache)
    @profile peakflops()
    pprof()
    MyModule.cache_hits = 0
    MyModule.cache_misses = 0
end

# profile_method(exp_data, heuristic) 