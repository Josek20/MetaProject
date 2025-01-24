
using MyModule
using MyModule.Flux
using MyModule.Mill
using MyModule.LRUCache
using MyModule.Metatheory
using Serialization
using DataFrames
using Optimisers
using Statistics
using CSV

using MyModule: new_all_symbols, exp_size

using MyModule: extract_training_data3, interned_build_tree!, extract_smallest_terminal_node, Node, intern!, extract_training_data1, TrainingSample
using MyModule.DataStructures


function build_search_tree(heuristic, ex::Expr, max_steps, max_depth, all_symbols, symbols_to_index, theory)
    soltree = Dict{UInt64, Node}()
    open_list = PriorityQueue{Node, Float32}()
    close_list = Set{UInt64}()
    ex = intern!(ex)
    root = Node(ex, (0,0), nothing, 0, nothing)
    o = heuristic(root.ex)
    soltree[root.node_id] = root
    enqueue!(open_list, root, only(o))
    reached_goal = interned_build_tree!(soltree, heuristic, open_list, close_list, all_symbols, symbols_to_index, max_steps, max_depth, theory)
    smallest_node = extract_smallest_terminal_node(soltree, close_list)
    return(soltree, smallest_node, root)
end

function _extract_training_data(smallest_node, soltree, root, max_depth, initial_expr)
    big_vector, hp, hn, proof_vector, _ = extract_training_data3(smallest_node, soltree, root, max_depth)
    saturated = false
    simplified_expression = smallest_node.ex
    new_sample = TrainingSample(big_vector, saturated, simplified_expression, proof_vector, hp, hn, initial_expr)            
end


MyModule.get_value(x) = x

const EMPTY_DICT = Base.ImmutableDict{Int,Any}()

function (r::Metatheory.DynamicRule)(term)
  # n == 1 means that exactly one term of the input (term,) was matched
  success(bindings, n) =
    if n == 1
      bvals = [bindings[i] for i in 1:length(r.patvars)]
      bvals = map(MyModule.get_value, bvals)
      v = r.rhs_fun(term, nothing, bvals...)
      if isnothing(v)
        return nothing
      end 
      v = term isa MyModule.NodeID ? MyModule.intern!(v) : v
      return(v)
    end

  try
    return r.matcher(success, (term,), EMPTY_DICT)
  catch err
    rethrow(err)
    throw(RuleRewriteError(r, term))
  end
end

function training_data(n=typemax(Int))
    train_data_path = "../data/neural_rewrter/train.json"
    train_data = load_data(train_data_path)[10_001:70_000]
    train_data = filter(x->!occursin("select", x[1]), train_data)
    train_data = preprosses_data_to_expressions(train_data)
    train_data = sort(train_data, by=x->MyModule.exp_size(x))
    last(train_data, min(length(train_data), n))
end


function prepare_dataset(n=typemax(Int))
    samples = deserialize("../../data/training_data/size_heuristic_training_samples1.bin")
    samples = vcat(samples...)
    samples = sort(samples, by=x->MyModule.exp_size(x.initial_expr))
    # samples = sort(samples, by=x->length(x.proof))
    last(samples, min(length(samples), n))
end

Heaviside(x) = x > 0 ? 1 : 0
hinge0(x::T) where {T} = x > 0 ? x : zero(T)
hinge1(x) = hinge0(x + 1)
 
function get_batched_samples(samples, batch_size=10)
    batched_samples = []
    for i in 1:batch_size:length(samples)-1
        pds = reduce(catobs, [j[1] for j in samples[i:batch_size - 1 + i]])
        pds_size = cumsum([size(bd.data.head)[2] for (bd, _, _) in samples[i:batch_size - 1 + i]])
        pushfirst!(pds_size, 0)
        modified_hps = map(x->x[1] .+ x[2], zip([hp for (_, hp, _) in samples[i:batch_size - 1 + i]], pds_size))
        modified_hns = map(x->x[1] .+ x[2], zip([hn for (_, _, hn) in samples[i:batch_size - 1 + i]], pds_size))

        hps = vcat(modified_hps...)
        hns = vcat(modified_hns...)
        
        push!(batched_samples, (pds, hps, hns))
    end
    return batched_samples
end

function lossfun(m, (ds, I₊, I₋))
    o = vec(MyModule.heuristic(m, ds))
    mean(softplus.(o[I₊] - o[I₋]))
end

function hardloss(m, (ds, I₊, I₋))
    o = vec(MyModule.heuristic(m, ds))
    sum(Heaviside.(o[I₊] - o[I₋]))
end

function summarize(xs, ys)
    summary = (;  better = mean(x > y for (x,y) in zip(xs, ys)),
        worse = mean(x < y for (x,y) in zip(xs, ys)),
        same = mean(x == y for (x,y) in zip(xs, ys)), 
    )
end

function summarize(all_stats)
    for i in 1:length(all_stats) - 1
        println(summarize(all_stats[i], all_stats[i+1]))
    end
end

function train(model, samples)
    proof_neighborhood = 1
    optimizer=ADAM()
    opt_state = Flux.setup(optimizer, model)
    loss_stats = []
    all_stats = []
    push!(all_stats, [s.goal_size for s in samples])
    for outer_epoch in 1:3
        for ep in 1:20
            sum_loss = 0.0
            hard_loss = 0
            t = @elapsed for (i, s) in enumerate(samples)
                ds, I₊, I₋ = s.ds, s.hp, s.hn
                sa, grad = Flux.Zygote.withgradient(Base.Fix2(lossfun, (ds, I₊, I₋)), model)
                sum_loss += sa
                Optimisers.update!(opt_state, model, only(grad))
            end
            violations = [hardloss(model, sample) for sample in samples]
            @show (t, sum(violations), quantile(violations, 0:0.1:1))
        end

        @show MyModule.cache_status()
        t = @elapsed samples = map(enumerate(samples)) do (i, old_sample)
            MyModule.reset_inference_caches()

            # search_tree, solution = build_search_tree

            # best_node = find_the_solution

            t = @elapsed (soltree, smallest_node, root) = build_search_tree(model, old_sample.initial_expr, 1000, 50, new_all_symbols, sym_enc, theory)
            sa = old_sample.goal_size
            sb = MyModule.exp_size(smallest_node.ex)
            s = string(i,"  ", sa, "-->",sb, "  time to simplify: ", t)
            # @show MyModule.cache_status()
            if sa == sb
                s = Base.AnnotatedString(s, [(1:length(s), :face, :grey)])
                # println(s)
            elseif sa > sb
                s = Base.AnnotatedString(s, [(1:length(s), :face, :red)])
                println(s)
            else
                s = Base.AnnotatedString(s, [(1:length(s), :face, :green)])
                println(s)
            end
            if sb < sa
                new_sample = _extract_training_data(smallest_node, soltree, root, proof_neighborhood, old_sample.initial_expr)
                ns = (ds = MyModule.deduplicate(new_sample.training_data), hp = new_sample.hp, hn = new_sample.hn, initial_expr = old_sample.initial_expr, goal_size = sb)
                return(ns)
            else
                return(old_sample)
            end
        end

        push!(all_stats,  [s.goal_size for s in samples])
        println(mean.(all_stats))
        # @show summarize(stats)
        # @show (t, better_samples_counter, mean_size)
        # (;  better = mean(x.goal_size > y.goal_size for (x,y) in zip(new_samples, samples)),
        #     worse = mean(x.goal_size < y.goal_size for (x,y) in zip(new_samples, samples)),
        #     same = mean(x.goal_size == y.goal_size for (x,y) in zip(new_samples, samples)), 
        #     time = t1,
        # )
        # samples = new_samples
    # resolve all expressions
    # improve the current solutions
    # form the training samples
    end
end


hidden_size = 128
hidden_size = 128

function ffnn(idim, hidden_size, layers)
    layers == 1 && return Dense(idim, hidden_size, Flux.gelu)
    layers == 2 && return Flux.Chain(Dense(idim, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu))
end

head_model = ProductModel(
    (;head = ffnn(length(new_all_symbols), hidden_size, 1),
      args = ffnn(hidden_size, hidden_size, 1),  
        ),
    ffnn(2*hidden_size, hidden_size, 1)
    )

args_model = ProductModel(
    (;args = ffnn(hidden_size, hidden_size, 1),  
      position = Dense(2,hidden_size),  
        ),
    ffnn(2*hidden_size, hidden_size, 1)
    )

model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Flux.Chain(Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, 1)),
    );
    
training_samples = prepare_dataset();
epochs = 1000
samples = map(training_samples) do sample
    ds, hp, hn, _ = MyModule.get_training_data_from_proof(sample.proof, sample.initial_expr)
    (; ds = MyModule.deduplicate(ds),
       hp,
       hn,
       initial_expr = sample.initial_expr,
       goal_size = exp_size(sample.expression),
    )
end
train(model, samples)


# data = [i.initial_expr for i in training_samples]
# experiment_name = "test_v1_sorted_$(length(data))_$(epochs)_hidden_size_$(hidden_size)"
# serialize("models/trained_heuristic_$(experiment_name).bin", heuristic)
# #heuristic = deserialize("models/trained_heuristic_testing_new_training_api_sorted_filtered_955_10_hidden_size_64.bin")
# MyModule.reset_all_function_caches()
# df = map(data) do ex
#     simplified_expression, _, big_vector, saturated, hp, hn, root, proof_vector, _ = MyModule.interned_initialize_tree_search(heuristic, ex, 1000, 10, new_all_symbols, sym_enc, theory)
#     (; s₀ = MyModule.exp_size(root.ex), sₙ = MyModule.exp_size(simplified_expression), se = simplified_expression, pr = proof_vector)
# end |> DataFrame
# CSV.write("profile_results_trained_heuristic_$(experiment_name).csv", df)