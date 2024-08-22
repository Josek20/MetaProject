using Metatheory
using Flux
using MyModule
using Mill
using DataFrames
using BSON
using CSV
using JLD2

train_data_path = "data/neural_rewrter/train.json"
test_data_path = "data/neural_rewrter/test.json"

train_data = isfile(train_data_path) ? load_data(train_data_path)[10000:10500] : load_data(test_data_path)[1:1000]
test_data = load_data(test_data_path)[1:1000]
train_data = preprosses_data_to_expressions(train_data)
test_data = preprosses_data_to_expressions(test_data)

# theory = @theory a b c begin
#     a::Number + b::Number => a + b
#     a::Number - b::Number => a - b
#     a::Number * b::Number => a * b
#     a::Number / b::Number => a / b
#     #(a / b::Number) * c::Number => :($a * div(c, b))
#     #(a * b::Number) / c::Number => :($a * div(b, c))
#
#
#     a * (b * c) --> (a * b) * c
#     (a * b) * c --> a * (b * c)
#     a + (b + c) --> (a + b) + c
#     (a + b) + c --> a + (b + c)
#     a + (b + c) --> (a + c) + b
#     (a + b) + c --> (a + c) + b
#     (a - b) + b --> a
#     (-a + b) + a --> b
#
#
#     a + b --> b + a
#     a * (b + c) --> (a * b) + (a * c)
#     (a + b) * c --> (a * c) + (b * c)
#     (a / b) * c --> (a * c) / b
#     (a / b) * b --> a
#     (a * b) / b --> a
#
#
#
#     #-a --> -1 * a
#     #a - b --> a + -b
#     1 * a --> a
#     a + a --> 2*a
#
#     0 * a --> 0
#     a + 0 --> a
#     a - a --> 0
#
#     a <= a --> 1
#     a + b <= c + b --> a <= c
#     a + b <= b + c --> a <= c
#     a * b <= c * b --> a <= c
#     a / b <= c / b --> a <= c
#     a - b <= c - b --> a <= c
#     a::Number <= b::Number => a<=b ? 1 : 0
#     a <= b - c --> a + c <= b
#     #a <= b - c --> a - b <= -c
#     a <= b + c --> a - c <= b
#     a <= b + c --> a - b <= c
#
#     a + c <= b --> a <= b - c
#     a - c <= b --> a <= b + c
#
#     min(a, min(a, b)) --> min(a, b)
#     min(min(a, b), b) --> min(a, b)
#     max(a, max(a, b)) --> max(a, b)
#     max(max(a, b), b) --> max(a, b)
#
#     max(a, min(a, b)) --> a
#     min(a, max(a, b)) --> a
#     max(min(a, b), b) --> b
#     min(max(a, b), b) --> b
#
#     min(a + b::Number, a + c::Number) => b < c ? :($a + $b) : :($a + $c)
#     max(a + b::Number, a + c::Number) => b < c ? :($a + $c) : :($a + $b)
#     min(a - b::Number, a - c::Number) => b < c ? :($a - $c) : :($a - $b)
#     max(a - b::Number, a - c::Number) => b < c ? :($a - $b) : :($a - $c)
#     min(a * b::Number, a * c::Number) => b < c ? :($a * $b) : :($a * $c)
#     max(a + b::Number, a + c::Number) => b < c ? :($a + $c) : :($a + $b)
#     max(a, b::Number) <= c::Number => b > c ? 0 : :($max($a, $b) <= $c)
#     #min(a, b::Number) <= c::Number => b < c ? 0 : 1
#     #min(a + b, a + c) --> min(b, c)
# end
theory = @theory a b c d x y begin
    # add.rc
    a + b --> b + a
    # + (~a, ~b) --> + (~b,~a)
    a + (b + c) --> (a + b) + c
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
    a && a --> a
    a && !a --> 0
    (a == c::Number) && (b == x::Number) => c != x ? 0 : :($a)
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
    (a * b) / c --> a * (b / c)
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
    x < y --> (-1 * y) < (-1 * x)
    a < a --> 0
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
    min(a , b::Number) < c::Number --> :($a < $c) || :($b < $c)
    max(a , b::Number) < c::Number --> :($a < $c) && :($b < $c)
    c::Number < max(a , b::Number) --> :($c < $a) || :($c < $b)
    min(a * b::Number, c::Number) => b % c == 0 && b > 0 ? :(min($a, $b / $c) * $c) : :(min($a * $b, $c))
    min(a * b::Number, d * c::Number) => b % c == 0 && b > 0 ? :(min($a, $d * ($c/$b))*$b) : :(min($a * $b, $d * $c))
    min(a * b::Number, c::Number) => b % c == 0 && b < 0 ? :(max($a, $b / $c) * $c) : :(min($a * $b, $c))
    min(a * b::Number, d * c::Number) => b % c == 0 && b < 0 ? :(max($a, $d * ($c/$b))*$b) : :(min($a * $b, $d * $c))
    max(a * c::Number, b * c::Number) => c < 0 ? :(min($a, $b) * $c) : :(max($a, $b) * $c)

    # modulo.rs
    a % 0 --> 0
    a % a --> 0
    a % 1 --> 0

    # a % b::Number --> b > 0 ? :(($a + $b) % $b) : :($a % $b)
    (a * -1) % b --> -1 * (a % b)
    -1 * (a % b) --> (a * -1) % b
    (a - b) % 2 --> (a + b) % 2

    ((a * b::Number) + d) % c::Number => b % c == 0 ? :($b % $c) : :((($a * $b) + $d) % $c)
    (b::Number * a) % c::Number => b % c == 0 ? 0 : :(($b * $a) % $c)
    
    # mul.rs
    # a * b --> b * a
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
end

hidden_size = 128 
heuristic = ExprModel(
    Flux.Chain(Dense(length(symbols_to_index) + 2, hidden_size,relu), Dense(hidden_size,hidden_size)),
    Mill.SegmentedSumMax(hidden_size),
    Flux.Chain(Dense(3*hidden_size, hidden_size,relu), Dense(hidden_size, hidden_size)),
    Flux.Chain(Dense(hidden_size, hidden_size,relu), Dense(hidden_size, 1))
    )

epochs = 100
optimizer = ADAM()
training_samples = Vector{TrainingSample}()
pc = Flux.params([heuristic.head_model, heuristic.aggregation, heuristic.joint_model, heuristic.heuristic])
max_steps = 1000
max_depth = 10
n = 1

df = DataFrame([[], [], [], [], []], ["Epoch", "Id", "Simplified Expr", "Proof", "Length Reduced"])
# @load "training_samplesk1000_v3.jld2" training_samples
if isfile("models/tre1e_search_heuristic.bson")
    BSON.@load "models/tree_search_heuristic.bson" heuristic
elseif isfile("data/training_data/tr2aining_samplesk1000_v3.jld2")
    @load "data/training_data/training_samplesk1000_v3.jld2" training_samples
else
    train_heuristic!(heuristic, train_data, training_samples, max_steps, max_depth, all_symbols, theory)
    # @load "data/training_data/training_samplesk1000_v3.jld2" training_samples

    test_training_samples(training_samples, train_data, theory)

    for ep in 1:epochs 
        # train_heuristic!(heuristic, train_data, training_samples, max_steps, max_depth)
        for sample in training_samples
            if isnothing(sample.training_data) 
                continue
            end
            grad = gradient(pc) do
                # o = heuristic(sample.training_data)
                a = heuristic_loss(heuristic, sample.training_data, sample.hp, sample.hn)
                # a = loss(heuristic, sample.training_data, sample.hp, sample.hn)
                @show a
                # if isnan(a)
                #     println(sample.expression)
                # end
                return a
            end
            Flux.update!(optimizer, pc, grad)
        end
        # @show avarage_length_reduction
        length_reduction, proofs, simp_expressions = test_heuristic(heuristic, test_data[1:10], max_steps, max_depth)
        # @show length_reduction
        # @show proofs
        # @show length_reduction
        new_df_rows = [(ep, ind, s[1], s[2], s[3]) for (ind,s) in enumerate(zip(simp_expressions,  proofs, length_reduction))]
        for row in new_df_rows
            push!(df, row)
        end
    end
    BSON.@save "models/tree_search_heuristic.bson" heuristic
    CSV.write("stats/data100.csv", df)
end
