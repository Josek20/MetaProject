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

old_all_symbols = [:+, :-, :/, :*, :<=, :>=, :min, :max, :<, :>, :select, :&&, ]#:||]# :(==), :!, :rem, :%]
old_symbols_to_index = Dict(i=>ind for (ind, i) in enumerate(old_all_symbols))

hidden_size = 256
policy = ExprModel(
    Flux.Chain(Dense(length(old_symbols_to_index) + 2, hidden_size,relu), Dense(hidden_size,hidden_size)),
    Mill.SegmentedSumMax(hidden_size),
    Flux.Chain(Dense(3*hidden_size, hidden_size,relu), Dense(hidden_size, hidden_size)),
    Flux.Chain(Dense(hidden_size, hidden_size,relu), Dense(hidden_size, 1))
    )

optimizer = ADAM()
policy_training_samples = Vector{PolicyTrainingSample}()
@load "data/training_data/policy_training_samplesk999_v5.jld2" policy_training_samples
pc = Flux.params([policy.head_model, policy.aggregation, policy.joint_model, policy.heuristic])
df = DataFrame([[], [], [], [], []], ["Epoch", "Id", "Simplified Expr", "Proof", "Length Reduced"])
optimizer = Flux.ADAM()
epoch = 60
function policy_sanity_check(training_samples, policy)
    count = 0
    tmp = []
    for (ind,i) in enumerate(training_samples[1:1])
        if isnothing(i.training_data)
            @show ind
            continue
        end 
        o = policy(i.training_data)

        # @show o[1, :]
        tmp = []
        for (iind, j) in enumerate(o[1, :])
            if j in tmp
                @show ind
                @show iind
                count += 1
            else
                push!(tmp, j)
            end
        end
    end
    # @show tmp
    @show count
    @assert count == 0
end
# policy_sanity_check(policy_training_samples, policy)
# yk = ones(length(theory))
plot_loss = [[] for _ in 1:epoch]
plot_reduction = []
sr = 1
er = 7
for ep in 1:epoch
    @show ep
    for (ind,i) in enumerate(policy_training_samples[sr:er])
        gd = gradient(pc) do
            loss = policy_loss_func(policy, i.training_data, i.hp, i.hn)
            return loss
        end
        loss = policy_loss_func(policy, i.training_data, i.hp, i.hn)
        push!(plot_loss[ep], loss)
        Flux.update!(optimizer, pc, gd)
    end
end

length_reduction, proof, simp_expressions = test_policy(policy, train_data[sr:er], theory, 60, old_symbols_to_index, old_all_symbols)
@show simp_expressions
# @show training_samples[9].expression
@show sum(length_reduction) / length(length_reduction)
st = hcat(plot_loss...)
plot(1:epoch, transpose(st))
savefig("stats/policy_training_loss$(epoch)ep.png")
policy_sanity_check(policy_training_samples, policy)
# # push!(plot_reduction, avarage_length_reduction)
# new_df_rows = [(1, ind, s[1], s[2], s[3]) for (ind,s) in enumerate(zip(simp_expressions,  proof, length_reduction))]
# for row in new_df_rows
#     push!(df, row)
# end
