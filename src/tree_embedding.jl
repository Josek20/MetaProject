my_sigmoid(x, k = 0.01, m = 0) = 1 / (1 + exp(-k * (x - m)))


# function cached_inference!(ex::ExprWithHash, cache, model, all_symbols, symbols_to_ind)
#     get!(cache, ex) do
#         if isa(ex.ex, Expr)
#             cached_inference1!(ex, cache, model, all_symbols, symbols_to_ind)
#         elseif isa(ex.ex, Number)
#             cached_inference3!(ex, cache, model, all_symbols, symbols_to_ind)
#         elseif isa(ex.ex, Symbol)
#             cached_inference2!(ex, cache, model, all_symbols, symbols_to_ind)
#         end
#     end
# end


function cached_inference!(ex::ExprWithHash, ex_v::Expr, cache, model, all_symbols, symbols_to_ind)
    get!(cache, ex) do
        if ex.head == :call
            fun_name, args =  ex.head, ex.args
        elseif ex.head in all_symbols
            fun_name = ex.head
            args = ex.args
        else
            error("unknown head $(ex.head)")
        end
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_index[fun_name]] = 1
        args = cached_inference!(args, cache, model, all_symbols, symbols_to_ind)
        if isa(model, ExprModel)
            tmp = model.head_model(encoding)
        else
            tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
        end
        tmp = vcat(tmp, args)
    end
end


function cached_inference!(ex::ExprWithHash, ex_v::Symbol, cache, model, all_symbols, symbols_to_ind)
    get!(cache, ex) do
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_ind[ex.ex]] = 1

        if isa(model, ExprModel)
            tmp = model.head_model(encoding)
            a = vcat(tmp, model.aggregation.:ψ)
        else
            tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
            a = vcat(tmp, model.expr_model.aggregation.:ψ)
        end
        return a
    end
end


function cached_inference!(ex::ExprWithHash, ex_v::Number, cache, model, all_symbols, symbols_to_ind)
    get!(cache, ex.ex) do
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_ind[:Number]] = my_sigmoid(ex.ex)

        if isa(model, ExprModel)
            tmp = model.head_model(encoding)
            a = vcat(tmp, model.aggregation.:ψ)
        else
            tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
            a = vcat(tmp, model.expr_model.aggregation.:ψ)
        end
        return a
    end
end


function cached_inference!(ex::Expr, cache, model, all_symbols, symbols_to_ind)
    get!(cache, ex) do
        if ex.head == :call
            fun_name, args =  ex.args[1], ex.args[2:end]
        elseif ex.head in all_symbols
            fun_name = ex.head
            args = ex.args
        else
            error("unknown head $(ex.head)")
        end
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_index[fun_name]] = 1
        args = cached_inference!(args, cache, model, all_symbols, symbols_to_ind)
        if isa(model, ExprModel)
            tmp = model.head_model(encoding)
        else
            tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
        end
        tmp = vcat(tmp, args)
    end
end


function cached_inference!(ex::Symbol, cache, model, all_symbols, symbols_to_ind)
    get!(cache, ex) do
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_ind[ex]] = 1

        if isa(model, ExprModel)
            tmp = model.head_model(encoding)
            a = vcat(tmp, model.aggregation.:ψ)
        else
            tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
            a = vcat(tmp, model.expr_model.aggregation.:ψ)
        end
        return a
    end
end


function cached_inference!(ex::Number, cache, model, all_symbols, symbols_to_ind)
    get!(cache, ex) do
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_ind[:Number]] = my_sigmoid(ex)

        if isa(model, ExprModel)
            tmp = model.head_model(encoding)
            a = vcat(tmp, model.aggregation.:ψ)
        else
            tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
            a = vcat(tmp, model.expr_model.aggregation.:ψ)
        end
        return a
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
    my_args = hcat(my_tmp...)
    tmp = vcat(my_args, Flux.onehotbatch(1:l, 1:2))
    if isa(model, ExprModel)
        tmp = model.joint_model(tmp)
        a = model.aggregation(tmp,  Mill.AlignedBags([1:l])) 
    else
        tmp = model.expr_model.joint_model(tmp, model.model_params.joint_model)
        a = model.expr_model.aggregation(tmp,  Mill.AlignedBags([1:l])) 
    end
    return a[:,1]
end


function cached_inference!(args::Vector{MyModule.ExprWithHash}, cache, model, all_symbols, symbols_to_ind)
    l = length(args)
    my_tmp = [cached_inference!(a, a.ex, cache, model, all_symbols, symbols_to_ind) for a in args]
    my_args = hcat(my_tmp...)
    tmp = vcat(my_args, Flux.onehotbatch(1:l, 1:2))
    if isa(model, ExprModel)
        tmp = model.joint_model(tmp)
        a = model.aggregation(tmp,  Mill.AlignedBags([1:l])) 
    else
        tmp = model.expr_model.joint_model(tmp, model.model_params.joint_model)
        a = model.expr_model.aggregation(tmp,  Mill.AlignedBags([1:l])) 
    end
    return a[:,1]
end


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


struct InitExprModelDense{HM, JM, H}
    head_model::HM
    joint_model::JM 
    heuristic::H 
end


function InitExprModelDense(m::ExprModel)
    hm = SimpleChains.init_params(m.head_model)
    jm = SimpleChains.init_params(m.joint_model)
    h = SimpleChains.init_params(m.heuristic)
    return InitExprModelDense(hm, jm, h) 
end


struct ExprModelSimpleChains
    expr_model::ExprModel
    model_params::InitExprModelDense
end


function ExprModelSimpleChains(m::ExprModel) 
    ie = InitExprModelDense(m)
    return ExprModelSimpleChains(m, ie)
end


function (m::ExprModelSimpleChains)(ds::ProductNode)
    m.expr_model(ds, m.model_params)
end


function (m::ExprModel)(ds::ProductNode)
    m.heuristic(embed(m, ds))
end


function (m::ExprModel)(ds::ProductNode, params::InitExprModelDense)
    m.heuristic(embed(m, ds, params), params.heuristic)
end


function embed(m::ExprModel, ds::ProductNode, params::InitExprModelDense) 
    # @show ds.data
    if haskey(ds.data, :position)
        # @show size(m.head_model(ds.data.args.data.head.data))
        tmp = m.head_model(ds.data.args.data.head.data, params.head_model)
        tmp = vcat(tmp, embed(m, ds.data.args.data.args, params), ds.data.position.data)
    else
        o = m.head_model(ds.data.head.data, params.head_model)
        # tmp = vcat(o, zeros(Float32, 2, size(o)[2]))
        # tmp = vcat(tmp, embed(m, ds.data.args))
        tmp = vcat(o, embed(m, ds.data.args, params), zeros(Float32, 2, size(o)[2]))
    end
    # tmp = vcat(m.head_model(ds.data.args.head.data), embed(m, ds.data.args))
    m.joint_model(tmp, params.joint_model)
end


function embed(m::ExprModel, ds::BagNode,params::InitExprModelDense)
    tmp = embed(m, ds.data, params)
    m.aggregation(tmp, ds.bags)
end


embed(m::ExprModel, ds::Missing, params::InitExprModelDense) = missing


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
# cached_inference!(ex::Expr, cache, model, all_symbols, symbols_to_ind)

function (m::ExprModel)(ex::Expr, cache, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
    ds = cached_inference!(ex, cache, m, all_symbols, symbols_to_index)
    tmp = vcat(ds, zeros(Float32, 2))
    m.heuristic(m.joint_model(tmp))
end


# function (m::ExprModelSimpleChains)(ex::ExprWithHash, cache, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
#     ds = cached_inference!(ex, ex.ex, cache, m, all_symbols, symbols_to_index)
#     tmp = vcat(ds, zeros(Float32, 2))
#     m.expr_model.heuristic(m.expr_model.joint_model(tmp, m.model_params.joint_model), m.model_params.heuristic)
# end


function (m::ExprModelSimpleChains)(ex::Expr, cache, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
    # By switching between this 
    # ds = cached_inference!(ExprWithHash(ex), ex, cache, m, all_symbols, symbols_to_index)
    # and this
    ds = cached_inference!(ex, cache, m, all_symbols, symbols_to_index)
    
    tmp = vcat(ds, zeros(Float32, 2))
    m.expr_model.heuristic(m.expr_model.joint_model(tmp, m.model_params.joint_model), m.model_params.heuristic)
end
# function embed(m::ExprModel, ds::Vector)
#     tmp = vcat(ds, zeros(Float32, 2))
#     m.heuristic(m.joint_model(tmp))
# end


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


function loss(heuristic::ExprModel, params::InitExprModelDense, big_vector, hp, hn, surrogate::Function = logistic)
    o = heuristic(big_vector, params)
    p = (o * hp) .* hn

    diff = p - o[1, :] .* hn
    filtered_diff = filter(!=(0), diff)
    # return sum(log.(1 .+ exp.(filtered_diff)))
    # return mean(softmax(filtered_diff))
    return sum(surrogate.(filtered_diff))
end