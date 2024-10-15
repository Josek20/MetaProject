using MyModule
using MyModule.Flux
using MyModule.Mill
using MyModule.DataStructures
using MyModule.LRUCache
using MyModule.Metatheory
using BenchmarkTools
using Serialization
using Profile
using PProf
using DataFrames
using ProfileCanvas
using SimpleChains


theory = @theory a b c d x y begin
    a::Number + b::Number => a + b
    a::Number - b::Number => a - b
    a::Number * b::Number => a * b
    # a::Number / b::Number => b != 0 ? a / b : :($a / $b)
    # add.rc
    a + b --> b + a
    a + (b + c) --> (a + b) + c
    (a + b) + c --> a + (b + c)
    (a + b) - c --> a + (b - c)
    a + (b - c) --> (a + b) - c
    (a - b) - c --> a - (b + c) 
    a - (b + c) --> (a - b) - c
    (a - b) + c --> a - (b - c)
    a - (b - c) --> (a - b) + c
    # (a + b) - c --> a + (b - c)
    # a + (b - c) --> (a + b) - c 
    a + 0 --> a
    a * (b + c) --> a*b + a*c
    a*b + a*c --> a * (b + c)
    (a / b) + c --> (a + (c * b)) / b
    (a + (c * b)) / b --> (a / b) + c
    x / 2 + x % 2 --> (x + 1) / 2
    x * a + y * b --> ((a / b) * x + y) * b
    
    # and.rc
    a && b --> b && a
    a && (b && c) --> (a && b) && c
    1 && a --> a
    a && 1 --> a
    a && a --> a
    a && !a --> 0
    !a && a --> 0

    (a == c::Number) && (a == x::Number) => c != x ? 0 : :($a == $c)
    !a::Number && b::Number => a != b ? 0 : :($a)
    (a < y) && (a < b) --> a < min(y, b)
    a < min(y, b) --> (a < y) && (a < b)
    (a <= y) && (a <= b) --> a <= min(y, b)
    a <= min(y, b) --> (a <= y) && (a <= b)
    (a > y) && (a > b) --> x > max(y, b)
    a > max(y, b) --> (a > y) && (a > b)
    
    (a >= y) && (a >= b) --> a >= max(y, b)
    a >= max(y, b) --> (a >= y) && (a >= b)
    
    (a::Number > b) && (c::Number < b) => a < c ? 0 : :($a > $b) && :($c < $b)
    (a::Number >= b) && (c::Number <= b) => a < c ? 0 : :($a >= $b) && :($c <= $b)
    (a::Number >= b) && (c::Number < b) => a <= c ? 0 : :($a >= $b) && :($c < $b)
    
    a && (b || c) --> (a && b) || (a && c)
    a || (b && c) --> (a || b) && (a || c)
    b || (b && c) --> b

    # div.rs
    0 / a --> 0
    a / a --> 1
    (-1 * a) / b --> a / (-1 * b)
    a / (-1 * b) --> (-1 * a) / b
    -1 * (a / b) --> (-1 * a) / b
    (-1 * a) / b --> -1 * (a / b)

    (a * b) / c --> a / (c / b)
    a / (c / b) --> (a * b) / c
    (a / b) * c --> a / (b / c)
    a / (b / c) --> (a / b) * c
    # (a * b) / c --> a * (b / c) ?
    (a + b) / c --> (a / c) + (b / c)
    ((a * b) + c) / d --> ((a * b) / d) + (c / d)

    # eq.rs
    x == y --> y == x
    x == y => y != 0 ? :(($x - $y) == 0) : :($x == $y)
    x + y == a --> x == a - y
    x == x --> 1
    x*y == 0 --> (x == 0) || (y == 0)
    max(x,y) == y --> x <= y
    min(x,y) == y --> y <= x
    y <= x --> min(x,y) == y

    # ineq.rs
    x != y --> !(x == y)

    # lt.rs
    x > y --> y < x
    # x < y --> (-1 * y) < (-1 * x)
    a < a --> 0
    a <= a --> 1
    a + b < c --> a < c - b

    a - b < a --> 0 < b
    0 < a::Number => 0 < a ? 1 : 0
    a::Number < 0 => a < 0 ? 1 : 0
    min(a , b) < a --> b < a
    min(a, b) < min(a , c) --> b < min(a, c)
    max(a, b) < max(a , c) --> max(a ,b) < c

    # max.rs
    max(a, b) --> (-1 * min(-1 * a, -1 * b))
    
    # min.rs
    min(a, b) --> min(b, a)
    min(min(a, b), c) --> min(a, min(b, c))
    min(a,a) --> a
    min(max(a, b), a) --> a
    min(max(a, b), max(a, c)) --> max(min(b, c), a)
    min(max(min(a,b), c), b) --> min(max(a,c), b)
    min(a + b, c) --> min(b, c - a) + a
    min(a, b) + c --> min(a + c, b + c)
    
    min(a + c, b + c) --> min(a, b) + c
    min(c + a, b + c) --> min(a, b) + c
    min(a + c, b + c) --> min(a, b) + c
    min(c + a, c + b) --> min(a, b) + c

    min(a, a + b::Number) => b > 0 ? :($a) : :($a + $b)
    min(a ,b) * c::Number => c > 0 ? :(min($a * $c, $b * $c)) : :(max($a * $c, $b * $c)) 
    min(a * c::Number, b * c::Number) => c > 0 ? :(min($a ,$b) * $c) : :(max($a, $b) * $c)
    min(a, b) / c::Number => c > 0 ? :(min($a / $c, $b / $c)) : :(max($a/$c,$b/$c))
    min(a / c::Number, b / c::Number) => c>0 ? :(min($a, $b) / $c) : :(max($a,$b) / $c)
    max(a , b) / c::Number => c < 0 ? :(max($a / $c , $b / $c)) : :(min($a / $c, $b / $c))
    max(a / c::Number, b / c::Number) => c < 0 ? :(max($a, $b) / $c) : :(min($a, $b) / $c)
    min(max(a,b::Number), c::Number) => c <= b ? :($c) : :(min(max($a,$b),$c))
    # min((a / b::Number) * b::Number , a) => b > 0 ? :(($a / $b) * $b) : :($)
    min(a % b::Number, c::Number) => c >= b - 1 ? :($a % $b) : :(min($a % $b, $c))
    min(a % b::Number, c::Number) => c <= 1 - b ? :($c) : :(min($a % $b, $c))

    min(max(a, b::Number), c::Number) => b <= c ? :(max(min($a, $c), $b)) : :(min(max($a, $b), $c))
    max(min(a, c::Number), b::Number) => b <= c ? :(min(max($a, $b), $c)) : :(max(min($a, $c), $b))
    min(a , b::Number) <= c::Number --> a <= c || b <= c
    max(a , b::Number) <= c::Number --> a <= c && b <= c
    c::Number <= max(a , b::Number) --> c <= a || c <= b
    c::Number <= min(a , b::Number) --> c <= a && c <= b
    min(a * b::Number, c::Number) => c != 0 && b % c == 0 && b > 0 ? :(min($a, $b / $c) * $c) : :(min($a * $b, $c))
    min(a * b::Number, d * c::Number) => c != 0 && b % c == 0 && b > 0 ? :(min($a, $d * ($c/$b))*$b) : :(min($a * $b, $d * $c))
    min(a * b::Number, c::Number) => c != 0 && b % c == 0 && b < 0 ? :(max($a, $b / $c) * $c) : :(min($a * $b, $c))
    min(a * b::Number, d * c::Number) => c != 0 && b % c == 0 && b < 0 ? :(max($a, $d * ($c/$b))*$b) : :(min($a * $b, $d * $c))
    max(a * c::Number, b * c::Number) => c < 0 ? :(min($a, $b) * $c) : :(max($a, $b) * $c)

    # modulo.rs
    a % 0 --> 0
    a % a --> 0
    a % 1 --> 0

    # a % b::Number --> b > 0 ? :(($a + $b) % $b) : :($a % $b)
    (a * -1) % b --> -1 * (a % b)
    -1 * (a % b) --> (a * -1) % b
    (a - b) % 2 --> (a + b) % 2

    ((a * b::Number) + d) % c::Number => c != 0 && b % c == 0 ? :($b % $c) : :((($a * $b) + $d) % $c)
    (b::Number * a) % c::Number => c != 0 && b % c == 0 ? 0 : :(($b * $a) % $c)
    
    # mul.rs
    a * b --> b * a
    # a * b::Number --> b * a
    a * (b * c) --> (a * b) * c
    a * 0 --> 0
    0 * a --> 0
    a * 1 --> a
    1 * a --> a
    # (a / b) * b --> (a - (a % b))
    max(a,b) * min(a, b) --> a * b
    min(a,b) * max(a, b) --> a * b
    (a * b) / b --> a
    (b * a) / b --> a

    # not.rs
    x <= y --> !(y < x)
    !(y < x) --> x <= y
    x >= y --> !(x < y)
    !(x == y) --> x != y
    !(!x) --> x

    # or.rs
    x || y --> !((!x) && (!y))
    y || x --> x || y

    # sub.rs
    # a - b --> a + (-1 * b)
    
    # my rules
    a || 1 --> 1
    1 || a --> 1
    a::Number <= b::Number => a<=b ? 1 : 0
    a <= b - c --> a + c <= b
    a + c <= b --> a <= b - c
    #a <= b - c --> a - b <= -c
    a <= b + c --> a - c <= b
    a - c <= b --> a <= b + c
    a <= b + c --> a - b <= c
    a - b <= c --> a <= b + c
    a - a --> 0
    min(a::Number, b::Number) => a >= b ? b : a 
    max(a::Number, b::Number) => a >= b ? a : b
    # a + b::Number <= min(a + c::Number, d) =>
    a <= min(b,c) --> a<=b && a<=c
    min(b,c) <= a --> b<=a || c<=a
end





hidden_size = 128
heuristic = ExprModel(
    Flux.Chain(Dense(length(new_all_symbols), hidden_size, Flux.relu), Dense(hidden_size,hidden_size)),
    Mill.SegmentedSum(hidden_size),
    Flux.Chain(Dense(2*hidden_size + 2, hidden_size,Flux.relu), Dense(hidden_size, hidden_size)),
    Flux.Chain(Dense(hidden_size, hidden_size,Flux.relu), Dense(hidden_size, 1))
    )

# made_simple_chain(model::BagModel) = BagModel(made_simple_chain(model.im), model.a, made_simple_chain(model.bm))
# made_simple_chain(model::ProductModel) = ProductModel(map(made_simple_chain, model.ms), made_simple_chain(model.m))
# made_simple_chain(model::ArrayModel) = ArrayModel(made_simple_chain(model.m))
# made_simple_chain(m::Flux.Chain) = SimpleChain....
# made_simple_chain(m::Dense) = SimpleDense

heuristic1 = MyModule.ExprModelSimpleChains(ExprModel(
    SimpleChain(static(length(new_all_symbols)), TurboDense{true}(SimpleChains.relu, hidden_size),TurboDense{true}(identity, hidden_size)),
    Mill.SegmentedSum(hidden_size),
    SimpleChain(static(2 * hidden_size + 2), TurboDense{true}(SimpleChains.relu, hidden_size),TurboDense{true}(identity, hidden_size)),
    SimpleChain(static(hidden_size), TurboDense{true}(SimpleChains.relu, hidden_size),TurboDense{true}(identity, 1)),
))

exp_data = Vector()
open("benchmarking.bin", "r") do file
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
# BenchmarkTools.Trial: 2 samples with 1 evaluation.
#  Range (min … max):  3.609 s …    4.244 s  ┊ GC (min … max): 4.81% … 6.23%
#  Time  (median):     3.927 s               ┊ GC (median):    5.58% 
#  Time  (mean ± σ):   3.927 s ± 448.733 ms  ┊ GC (mean ± σ):  5.58% ± 1.01%

#   █                                                        █       
#   █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁     
#   3.61 s         Histogram: frequency by time         4.24 s <     

#  Memory estimate: 1.13 GiB, allocs estimate: 26841698.

function profile_method(data, heuristic)
    df = map(enumerate(data[1:20])) do (ex_id, ex)
        cache = LRU(maxsize=100_000)
        exp_cache = LRU{Expr, Vector}(maxsize=20000)
        training_samples = Vector{TrainingSample}()
        stats = @timed train_heuristic!(heuristic1, [ex], training_samples, 1000, 60, new_all_symbols, theory, variable_names, cache, exp_cache, 1)  
        (;time = stats.time, ic_hits = cache.hits, ic_misses = cache.misses, exp_hits = exp_cache.hits, exp_misses = exp_cache.misses, gctime = stats.gctime)
    end |> DataFrame
    CSV.write("profile_results.csv", df)
end


# profile_method(exp_data, heuristic) 
