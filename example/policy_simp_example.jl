using Metatheory
using MyModule
using BSON
using CSV
using DataFrames
using JLD2
using Flux
using Mill
using Plots


train_data_path = "data/neural_rewrter/train.json"
test_data_path = "data/neural_rewrter/test.json"

train_data = isfile(train_data_path) ? load_data(train_data_path)[1:1000] : load_data(test_data_path)[1:1000]
test_data = load_data(test_data_path)[1:1000]
train_data = preprosses_data_to_expressions(train_data)
test_data = preprosses_data_to_expressions(test_data)

theory = @theory a b c begin
    a::Number + b::Number => a + b
    a::Number - b::Number => a - b
    a::Number * b::Number => a * b
    a::Number / b::Number => a / b
    #(a / b::Number) * c::Number => :($a * div(c, b))
    #(a * b::Number) / c::Number => :($a * div(b, c))


    a * (b * c) --> (a * b) * c
    (a * b) * c --> a * (b * c)
    a + (b + c) --> (a + b) + c
    (a + b) + c --> a + (b + c)
    a + (b + c) --> (a + c) + b
    (a + b) + c --> (a + c) + b
    (a - b) + b --> a
    (-a + b) + a --> b


    a + b --> b + a
    a * (b + c) --> (a * b) + (a * c)
    (a + b) * c --> (a * c) + (b * c)
    (a / b) * c --> (a * c) / b
    (a / b) * b --> a
    (a * b) / b --> a



    #-a --> -1 * a
    #a - b --> a + -b
    1 * a --> a
    a + a --> 2*a

    0 * a --> 0
    a + 0 --> a
    a - a --> 0

    a <= a --> 1
    a + b <= c + b --> a <= c
    a + b <= b + c --> a <= c
    a * b <= c * b --> a <= c
    a / b <= c / b --> a <= c
    a - b <= c - b --> a <= c
    a::Number <= b::Number => a<=b ? 1 : 0
    a <= b - c --> a + c <= b
    #a <= b - c --> a - b <= -c
    a <= b + c --> a - c <= b
    a <= b + c --> a - b <= c

    a + c <= b --> a <= b - c
    a - c <= b --> a <= b + c

    min(a, min(a, b)) --> min(a, b)
    min(min(a, b), b) --> min(a, b)
    max(a, max(a, b)) --> max(a, b)
    max(max(a, b), b) --> max(a, b)

    max(a, min(a, b)) --> a
    min(a, max(a, b)) --> a
    max(min(a, b), b) --> b
    min(max(a, b), b) --> b

    min(a + b::Number, a + c::Number) => b < c ? :($a + $b) : :($a + $c)
    max(a + b::Number, a + c::Number) => b < c ? :($a + $c) : :($a + $b)
    min(a - b::Number, a - c::Number) => b < c ? :($a - $c) : :($a - $b)
    max(a - b::Number, a - c::Number) => b < c ? :($a - $b) : :($a - $c)
    min(a * b::Number, a * c::Number) => b < c ? :($a * $b) : :($a * $c)
    max(a + b::Number, a + c::Number) => b < c ? :($a + $c) : :($a + $b)
    max(a, b::Number) <= c::Number => b > c ? 0 : :($max($a, $b) <= $c)
    #min(a, b::Number) <= c::Number => b < c ? 0 : 1
    #min(a + b, a + c) --> min(b, c)
end

hidden_size = 128 
policy = ExprModel(
    Flux.Chain(Dense(length(symbols_to_index) + 2, hidden_size,relu), Dense(hidden_size,hidden_size)),
    Mill.SegmentedSumMax(hidden_size),
    Flux.Chain(Dense(3*hidden_size, hidden_size,relu), Dense(hidden_size, hidden_size)),
    Flux.Chain(Dense(hidden_size, hidden_size,relu), Dense(hidden_size, 1))
    )

optimizer = ADAM()
training_samples = Vector{TrainingSample}()
@load "data/training_data/training_samplesk1000_v3.jld2" training_samples
pc = Flux.params([policy.head_model, policy.aggregation, policy.joint_model, policy.heuristic])
df = DataFrame([[], [], [], [], []], ["Epoch", "Id", "Simplified Expr", "Proof", "Length Reduced"])
optimizer = Flux.ADAM()
epoch = 100
# yk = ones(length(theory))
plot_loss = [[] for _ in 1:epoch]
plot_reduction = []
for ep in 1:epoch
    @show ep
    for (ind,i) in enumerate(training_samples[9:9])
        td = copy(i.initial_expr)
        # hn = copy(i.hn)
        # for k in size(hn)[2]:-1:1
        #     hn[:, begin:k-1] .= hn[:, begin:k-1] .- hn[:, k]
        # end
        # gd = gradient(pc) do
        #     loss = policy_loss_func(heuristic, i.training_data, j, i.hp, hn)
        #     @show loss
        #     return loss
        # end
        # Flux.update!(optimizer, pc, gd)
        # @show td
        # @show i.proof
        tmp_loss = []
        for (j,jr) in i.proof
            applicable_rules = filter(r -> r[2] != td, execute(td, theory))
            # applicable_rules = execute(td, theory)
            ee = []
            final_index = 0
            # @show length(applicable_rules)
            for (ind, (r, new_expr)) in enumerate(applicable_rules)
                if r[1] == j && r[2] == jr
                    final_index = ind
                end
                push!(ee, ex2mill(new_expr, symbols_to_index, all_symbols))
            end
            ds = reduce(catobs, ee) 
            # @show final_index

            gd = gradient(pc) do
                loss = policy_loss_func1(policy, ds, final_index)
                # @show loss
                return loss
            end

            td = applicable_rules[final_index][2]
            loss = policy_loss_func1(policy, ds, final_index)
            push!(tmp_loss, loss)
            # tmp = []
            # traverse_expr!(td, theory[j], 1, [], tmp)
            # @show tmp
            # if isempty(tmp[jr])
            #     td = theory[j](td) 
            # # if !isempty(tmp)
            # else
            #     # if isempty(tmp[jr])
            #     #     td = theory[j](td)
            #     # else
            #     my_rewriter!(tmp[jr], td, theory[j])
            #     # end
            # end
            # @show td
            Flux.update!(optimizer, pc, gd)
        end
        push!(plot_loss[ep], sum(tmp_loss))
        # @show td
    end
end

length_reduction, proof, simp_expressions = test_policy(policy, train_data[9:9], theory, 60, symbols_to_index, all_symbols)
@show simp_expressions
@show training_samples[9].expression
st = hcat(plot_loss...)
plot(1:epoch, transpose(st))
savefig("stats/policy_training_loss$(epoch)ep.png")
# # push!(plot_reduction, avarage_length_reduction)
# new_df_rows = [(1, ind, s[1], s[2], s[3]) for (ind,s) in enumerate(zip(simp_expressions,  proof, length_reduction))]
# for row in new_df_rows
#     push!(df, row)
# end
