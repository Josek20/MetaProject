using MyModule
using MyModule.InternedExpr
using MyModule.LRUCache
using MyModule.DataStructures
using BenchmarkTools



function prepare_dataset(n=typemax(Int))
    samples = deserialize("data/training_data/size_heuristic_training_samples1.bin")
    samples = vcat(samples...)
    # samples = sort(samples, by=x->MyModule.exp_size(x.initial_expr, LRU(maxsize=10000)))
    samples = sort(samples, by=x->length(x.proof))
    last(samples, min(length(samples), n))
end

training_samples = prepare_dataset(10)
ex = training_samples[end].initial_expr

@benchmark begin
    # ex = training_samples[end].initial_expr
    ex = $ex
    exp_cache = LRU(maxsize=100_000)
    cache = LRU(maxsize=1_000_000)
    size_cache = LRU(maxsize=100_000)
    expr_cache = LRU(maxsize=100_000)
    soltree = Dict{UInt64, MyModule.Node}()
    open_list = PriorityQueue{MyModule.Node, Float32}()
    close_list = Set{UInt64}()
    ex = MyModule.intern!(ex)
    root = MyModule.Node(ex, (0,0), nothing, 0, nothing)
    o = MyModule.exp_size(root.ex, size_cache)
    soltree[root.node_id] = root
    enqueue!(open_list, root, only(o))
    reached_goal = MyModule.interned_build_tree!(soltree, heuristic, open_list, close_list, all_symbols, symbols_to_index, 1000, 10, theory, cache, exp_cache, size_cache, expr_cache, 0.5)
end
@benchmark begin
    # ex = training_samples[end].initial_expr
    ex = $ex
    exp_cache = LRU(maxsize=100_000)
    cache = LRU(maxsize=1_000_000)
    size_cache = LRU(maxsize=100_000)
    expr_cache = LRU{UInt, MyModule.ExprWithHash}(maxsize=100_000)
    root = MyModule.Node(ex, (0,0), nothing, 0, expr_cache)

    soltree = Dict{UInt64, MyModule.Node}()
    open_list = PriorityQueue{MyModule.Node, Float32}()
    second_open_list = PriorityQueue{MyModule.Node, Float32}()
    close_list = Set{UInt64}()
    expansion_history = Dict{UInt64, Vector}()
    encodings_buffer = Dict{UInt64, ProductNode}()
    soltree[root.node_id] = root
    o = MyModule.exp_size(root.ex, size_cache)
    enqueue!(open_list, root, only(o))
    MyModule.build_tree!(soltree, heuristic, open_list, close_list,  new_all_symbols, sym_enc, 1000, 10, theory, cache, exp_cache, size_cache, expr_cache, 1)
end
