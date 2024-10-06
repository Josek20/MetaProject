using Distributed

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
using Revise
using StatsBase
using CSV
using BSON
using JLD2


train_data_path = "./data/neural_rewrter/train.json"
test_data_path = "./data/neural_rewrter/test.json"


train_data = isfile(train_data_path) ? load_data(train_data_path)[1:1000] : load_data(test_data_path)[1:1000]
test_data = load_data(test_data_path)[1:1000]
train_data = preprosses_data_to_expressions(train_data)
test_data = preprosses_data_to_expressions(test_data)


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
    (a / b) * c --> a / (c / b)
    # (a * b) / c --> a * (b / c) ?
    (a + b) / c --> (a / c) + (b / c)
    ((a * b) + c) / d --> ((a * b) / d) + (c / d)

    # eq.rs
    x == y --> y == x
    x == y --> (x - y) == 0
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
    (a / b) * b --> (a - (a % b))
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
    a - b --> a + (-1 * b)
    
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
    Flux.Chain(Dense(length(symbols_to_index) + 1 + 13, hidden_size, relu), Dense(hidden_size,hidden_size)),
    Mill.SegmentedSum(hidden_size),
    Flux.Chain(Dense(2*hidden_size + 2, hidden_size,relu), Dense(hidden_size, hidden_size)),
    Flux.Chain(Dense(hidden_size, hidden_size,relu), Dense(hidden_size, 1))
    )

pc = Flux.params([heuristic.head_model, heuristic.aggregation, heuristic.joint_model, heuristic.heuristic])

epochs = 10
optimizer = ADAM()
# training_samples = [Vector{TrainingSample}() for _ in 1:number_of_workers]
training_samples = Vector{TrainingSample}()
max_steps = 1000
max_depth = 100
n = 1000

myex = :( (v0 + v1) + 119 <= min((v0 + v1) + 120, v2) && ((((v0 + v1) - v2) + 127) / (8 / 8) + v2) - 1 <= min(((((v0 + v1) - v2) + 134) / 16) * 16 + v2, (v0 + v1) + 119))
# myex = :((v0 + v1) + 119 <= min((v0 + v1) + 120, v2))
# myex = :((v0*23 + v2) - (v2 + v0*23) <= 100) 
# myex = :(121 - max(v0 * 59, 109) <= 1024)
df = DataFrame([[], [], [], [], []], ["Epoch", "Id", "Simplified Expr", "Proof", "Length Reduced"])
df1 = DataFrame([[] for _ in 0:epochs], ["Epoch$i" for i in 0:epochs])
df2 = DataFrame([[] for _ in 0:epochs], ["Epoch$i" for i in 0:epochs])

# @load "training_samplesk1000_v3.jld2" training_samples
# x,y,r = MyModule.caviar_data_parser("data/caviar/288_dataset.json")
# x,y,r = MyModule.caviar_data_parser("data/caviar/dataset-batch-2.json")
# train_heuristic!(heuristic, train_data[1:], training_samples, max_steps, max_depth, all_symbols, theory, variable_names)
# @assert 0 == 1
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
    exp_cache = LRU{Expr, Vector}(maxsize=10000)
    cache = LRU(maxsize=10000)
    for ep in 1:epochs 
        empty!(cache)
        # train_heuristic!(heuristic, train_data, training_samples, max_steps, max_depth)
        println("Epcoh $ep")
        println("Training===========================================")
        # training_samples = pmap(dt -> train_heuristic!(heuristic, batched_train_data[dt], training_samples[dt], max_steps, max_depth, all_symbols, theory, variable_names), collect(1:number_of_workers))
        train_heuristic!(heuristic, train_data[1:5], training_samples, max_steps, max_depth, all_symbols, theory, variable_names, cache, exp_cache)
        # simplified_expression, depth_dict, big_vector, saturated, hp, hn, root, proof_vector, m_nodes = MyModule.initialize_tree_search(heuristic, myex, max_steps, max_depth, all_symbols, theory, variable_names, cache, exp_cache)
        
        # ltmp = []
        # @show training_samples
        # @assert 0 == 1
        # MyModule.test_expr_embedding(heuristic, training_samples[1:n], theory, symbols_to_index, all_symbols, variable_names)

        # cat_samples = vcat(training_samples...)
        # for sample in training_samples
        for i in 1:50
            sample = StatsBase.sample(training_samples)
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
                BSON.@save "models/inf_grad_heuristic.bson" heuristic
                JLD2.@save "data/training_data/training_samples_inf_grad.jld2" training_samples
                @assert 0 == 1
            end
        # if isna(grad)
            @show sa
            # push!(ltmp, sa)
            Flux.update!(optimizer, pc, grad)
            # end
        end
        # for sample in cat_samples
        #     sa = MyModule.loss(heuristic, sample.training_data, sample.hp, sample.hn)
        #     push!(ltmp, sa)
        # end
        #
        # println("Testing===========================================")
        # # length_reduction, proofs, simp_expressions = test_heuristic(heuristic, train_data[1:n], max_steps, max_depth, variable_names, theory)
        # # push!(stats, simp_expressions)
        # push!(stats, [i.expression for i in cat_samples])
        # push!(proof_stats, [i.proof for i in cat_samples])
        # push!(loss_stats, ltmp)
    end
    # BSON.@save "models/tree_search_heuristic.bson" heuristic
    # CSV.write("stats/data100.csv", df)
end
@assert 0 == 1
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
r = "_manual_version"
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
