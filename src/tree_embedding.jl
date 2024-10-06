my_sigmoid(x, k = 0.01, m = 0) = 1 / (1 + exp(-k * (x - m)))

cache_hits = 0
cache_misses = 0
function cached_inference!(ex::Expr, cache, model, all_symbols, symbols_to_ind)
    # global cache_hits, cache_misses
    # if haskey(cache,ex)
    #     cache_hits += 1
    # end
    get!(cache, ex) do
        # cache_misses += 1
        if ex.head == :call
            fun_name, args =  ex.args[1], ex.args[2:end]
        elseif ex.head in all_symbols
            fun_name = ex.head
            args = ex.args
        else
            error("unknown head $(ex.head)")
        end
        encoding = zeros(Float32, length(all_symbols), 1)
        encoding[symbols_to_index[fun_name]] = 1
        args = cached_inference!(args, cache, model, all_symbols, symbols_to_ind)
        tmp = vcat(model.head_model(encoding), args)
    end
end


function cached_inference!(ex::Symbol, cache, model, all_symbols, symbols_to_ind)
    # global cache_hits, cache_misses
    # if haskey(cache,ex)
    #     cache_hits += 1
    # end
    get!(cache, ex) do
        # cache_misses += 1
        encoding = zeros(Float32, length(all_symbols), 1)
        encoding[symbols_to_ind[ex]] = 1
        ds = ProductNode((
            head = ArrayNode(encoding),
            args = BagNode(missing, [0:-1])
        ))
        tmp = embed(model,ds.data.args)
        vcat(model.head_model(ds.data.head.data), tmp)
    end
end


function cached_inference!(ex::Number, cache, model, all_symbols, symbols_to_ind)
    # global cache_hits, cache_misses
    # if haskey(cache,ex)
    #     cache_hits += 1
    # end
    get!(cache, ex) do
        # cache_misses += 1
        encoding = zeros(Float32, length(all_symbols), 1)
        encoding[symbols_to_ind[:Number]] = my_sigmoid(ex)
        ds = ProductNode((
            head = ArrayNode(encoding),
            args = BagNode(missing, [0:-1])
        ))
        tmp = embed(model,ds.data.args)
        vcat(model.head_model(ds.data.head.data), tmp)
    end
end

_short_aggregation(::SegmentedSum, x) = x
_short_aggregation(::SegmentedSum, x, y) = x + y
_short_aggregation(::SegmentedMean, x) = x
_short_aggregation(::SegmentedMean, x, y) = (x + y) ./ 2
_short_aggregation(::SegmentedMax, x) = x
_short_aggregation(::SegmentedMax, x, y) = max.(x, y)
const const_left = [1, 0]
const const_right = [0, 1]

function cached_inference!(args::Vector, cache, model, all_symbols, symbols_to_ind)
    l = length(args)
    my_tmp = [cached_inference!(a, cache, model, all_symbols, symbols_to_ind) for a in args]
    # @show size(my_tmp[1])
    my_args = hcat(my_tmp...)
    tmp = vcat(my_args, Flux.onehotbatch(1:l, 1:2))
    tmp = model.joint_model(tmp)
    model.aggregation(tmp,  Mill.AlignedBags([1:l]))
    # _short_aggregation(model.aggregation, tmp)
end

# julia> @benchmark begin
#            cache = LRU(maxsize=10000)
#            m1 = map(x->MyModule.cached_inference!(x,cache, heuristic, new_all_symbols, sym_enc), exp_data[1:100])
#            o1 = map(x->MyModule.embed(heuristic,x), m1)
#        end
# BenchmarkTools.Trial: 92 samples with 1 evaluation.
#  Range (min … max):  48.447 ms … 100.228 ms  ┊ GC (min … max): 0.00% … 44.71%
#  Time  (median):     53.311 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   54.699 ms ±   9.041 ms  ┊ GC (mean ± σ):  3.31% ±  8.90%
#
#        █▆                                                       
#   ▅▆▂▂▄██▇▃▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁▂▁▁▁▂ ▁
#   48.4 ms         Histogram: frequency by time         97.8 ms <
#
#  Memory estimate: 16.13 MiB, allocs estimate: 213996.

# julia> @benchmark begin
#            cache = LRU(maxsize=10000)
#            m1 = map(x->MyModule.cached_inference!(x,cache, heuristic, new_all_symbols, sym_enc), exp_data[1:100])
#            o1 = map(x->MyModule.embed(heuristic,x), m1)
#        end
# BenchmarkTools.Trial: 92 samples with 1 evaluation.
#  Range (min … max):  50.961 ms … 70.206 ms  ┊ GC (min … max): 0.00% … 20.41%
#  Time  (median):     52.012 ms              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   53.896 ms ±  3.734 ms  ┊ GC (mean ± σ):  2.94% ±  5.08%
#
#    █▇                                                          
#   ▇███▆▃▃▃▁▃▃▁▁▃▁▁▄▅▃▄▃▅▃▃▃▁▁▃▃▁▁▁▁▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃ ▁
#   51 ms           Histogram: frequency by time          70 ms <
#
#  Memory estimate: 16.96 MiB, allocs estimate: 268270.

# function cached_inference!(args::Vector, cache, model, all_symbols, symbols_to_ind)
#     l = length(args)
#     if l == 1
#         tmp = cached_inference!(args[1], cache, model, all_symbols, symbols_to_ind)
#         # @show size(tmp)
#         tmp = vcat(tmp, const_left)
#         # @show size(tmp)
#         embeddding = model.joint_model(tmp)
#         # I think we can skip embeddings, since mean / max, will become identities.
#         return(embeddding)
#         # return model.aggregation(embeddding)
        
#     elseif l == 2
#         t₁ = vcat(cached_inference!(args[1], cache, model, all_symbols, symbols_to_ind), const_left)
#         t₂ = vcat(cached_inference!(args[2], cache, model, all_symbols, symbols_to_ind), const_right)
#         _short_aggregation(model.aggregation, model.joint_model(t₁), model.joint_model(t₂))
#     else
#         error("Unexpected number of arguments $l, file issue with the expression")
#     end
# end


function ex2mill(ex::Expr, symbols_to_index, all_symbols, variable_names::Dict, cache, model)
    if ex.head == :call
        fun_name, args =  ex.args[1], ex.args[2:end]
    elseif ex.head in all_symbols # Symbol("&&") || ex.head == Symbol("min")
        fun_name = ex.head
        args = ex.args
    else
        error("unknown head $(ex.head)")
    end
    n = length(symbols_to_index) + 1
    encoding = zeros(Float32, n+13, 1)

    encoding[symbols_to_index[fun_name]] = 1

    # ds = ProductNode(; 
    #     args=ProductNode((
    #         head = ArrayNode(encoding),
    #         args = args2mill(args, symbols_to_index, all_symbols, variable_names)
    #     )),
    #     position = ArrayNode(zeros(2,1))
    # )
    ds = ProductNode((
        head = ArrayNode(encoding),
        args = args2mill(args, symbols_to_index, all_symbols, variable_names, cache, model)
    ))
    # get!(cache, ex) do 
    #     embed(model, ds)
    # end
    return(ds)
end


function ex2mill(ex::Symbol, symbols_to_index, all_symbols, variable_names, cache, model)
    n = length(symbols_to_index) + 1
    encoding = zeros(Float32, n+13, 1)
    # encoding[n] = 1     # encoding = Flux.onehotbatch([n], 1:n)
    encoding[variable_names[ex] + length(symbols_to_index)] = 1


    ds = ProductNode((
        head = ArrayNode(encoding),
        args = BagNode(missing, [0:-1])
        ))
end


function ex2mill(ex::Number, symbols_to_index, all_symbols, variable_names, cache, model)
    n = length(symbols_to_index) + 1
    encoding = zeros(Float32, n+13, 1)
    # encoding[n] = 1     # encoding = Flux.onehotbatch([n], 1:n)
    encoding[n + 13] = my_sigmoid(ex)
    ds = ProductNode((
        head = ArrayNode(encoding),
        args = BagNode(missing, [0:-1])
        ))
end


function args2mill(args::Vector, symbols_to_index, all_symbols, variable_names, cache, model)
    isempty(args) && return(BagNode(missing, [0:-1]))
    l = length(args)
    BagNode(ProductNode((;
        args = reduce(catobs, [ex2mill(a, symbols_to_index, all_symbols, variable_names, cache, model) for a in args]),
        position = ArrayNode(Flux.onehotbatch(1:l, 1:2)),
        )),
        [1:l]
        )

end


struct ExprModel{HM,A,JM,H}
    head_model::HM
    aggregation::A
    joint_model::JM    
    heuristic::H
end


function (m::ExprModel)(ds::ProductNode)
    m.heuristic(embed(m, ds))
end


function embed(m::ExprModel, ds::ProductNode) 
    # @show ds.data
    if haskey(ds.data, :position)
        # @show size(m.head_model(ds.data.args.data.head.data))
        tmp = m.head_model(ds.data.args.data.head.data)
        tmp = vcat(tmp, embed(m, ds.data.args.data.args), ds.data.position.data)
    else
        o = m.head_model(ds.data.head.data)
        # tmp = vcat(o, zeros(Float32, 2, size(o)[2]))
        # tmp = vcat(tmp, embed(m, ds.data.args))
        tmp = vcat(o, embed(m, ds.data.args), zeros(Float32, 2, size(o)[2]))
    end
    # tmp = vcat(m.head_model(ds.data.args.head.data), embed(m, ds.data.args))
    m.joint_model(tmp)
end
# function embed(m::ExprModel, ds::ProductNode)
#     # tmp = vcat(m.head_model(ds.data.args.data.head.data), ds.data.position.data)
#     tmp = vcat(m.head_model(ds.data.args.data.head.data), ds.data.position.data, embed(m, ds.data.args.data.args))   
#     m.joint_model(tmp)
# end


function embed(m::ExprModel, ds::BagNode)
    tmp = embed(m, ds.data)
    m.aggregation(tmp, ds.bags)
end


function embed(m::ExprModel, ds::Matrix)
    tmp = vcat(ds, zeros(Float32, 2, 1))
    m.heuristic(m.joint_model(tmp))
end


embed(m::ExprModel, ds::Missing) = missing


logistic(x) = log(1 + exp(x))
hinge(x) = max(0, 1 - x)
loss01(x) = x > 0


function loss(heuristic, big_vector, hp=nothing, hn=nothing, surrogate::Function = logistic)
    o = heuristic(big_vector) 
    p = (o * hp) .* hn

    diff = p - o[1, :] .* hn
    filtered_diff = filter(!=(0), diff)
    # return sum(log.(1 .+ exp.(filtered_diff)))
    # return mean(softmax(filtered_diff))
    return sum(surrogate.(filtered_diff))
end


function heuristic_loss(heuristic, data, in_solution, not_in_solution)
    loss_t = 0.0
    heuristics = heuristic(data)
    in_solution = findall(x->x!=0,sum(in_solution, dims=2)[:, 1]) 
    for (ind,i) in enumerate(in_solution)
        a = findall(x->x!=0,not_in_solution[:, ind])
        for j in a
            if heuristics[i] >= heuristics[j]
                loss_t += heuristics[i] - heuristics[j] + 1
            end
        end
    end

    return loss_t
end


function check_loss(heuristics, data, hp, hn)
    fl1 = heuristic_loss(heuristics, data, hp, hn)
    fl2 = loss(heuristics, data, hp, hn)
    @assert fl1 == fl2
end
