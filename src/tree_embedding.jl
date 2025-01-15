my_sigmoid(x, k = 0.01, m = 0) = 1 / (1 + exp(-k * (x - m)))


function cached_inference!(ex::ExprWithHash, ::Type{Expr}, cache, model, all_symbols, symbols_to_ind)
    get!(cache, ex) do
        args = ex.args
        fun_name = ex.head
        args = cached_inference!(args, cache, model, all_symbols, symbols_to_ind)
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_index[fun_name]] = 1
        if isa(model, ExprModel)
            tmp = model.head_model(encoding)
        else
            tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
        end
        tmp = vcat(tmp, args)
    end
end


function cached_inference!(ex::ExprWithHash, ::Type{Symbol}, cache, model, all_symbols, symbols_to_ind)
    get!(cache, ex) do
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_ind[ex.head]] = 1

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


function cached_inference!(ex::ExprWithHash, ::Type{Int}, cache, model, all_symbols, symbols_to_ind)
    get!(cache, ex) do
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_ind[:Number]] = my_sigmoid(ex.head)

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
        args = cached_inference!(args, cache, model, all_symbols, symbols_to_ind)
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_index[fun_name]] = 1
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


const const_left = [1, 0]
const const_right = [0, 1]
const const_one = [0, 0]
const const_third = [1, 1]
const const_both = [1 0; 0 1]
const const_nth = [1 0 1; 0 1 1]


function cached_inference!(args::Vector, cache, model, all_symbols, symbols_to_ind)
    l = length(args)
    my_tmp = [cached_inference!(a, cache, model, all_symbols, symbols_to_ind) for a in args]
    my_args = hcat(my_tmp...)
    if l == 2
        tmp = vcat(my_args, const_both)
    elseif l == 3
        tmp = vcat(my_args, const_nth)
    else
        tmp = vcat(my_args, const_left)
    end

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
    my_tmp = []
    for i in args
        if isa(i.head, Symbol) && symbols_to_ind[i.head] <= 18
            pl = cached_inference!(i, Expr, cache, model, all_symbols, symbols_to_ind)
        else
            if isa(i.head, Symbol)
                pl = cached_inference!(i, Symbol, cache, model, all_symbols, symbols_to_ind)
            else
                pl = cached_inference!(i, Int, cache, model, all_symbols, symbols_to_ind)
            end
        end
        push!(my_tmp, pl)
    end
    my_args = hcat(my_tmp...)
    if l == 2
        tmp = vcat(my_args, const_both)
    elseif l == 3
        tmp = vcat(my_args, const_nth)
    else
        tmp = vcat(my_args, const_one)
    end
    if isa(model, ExprModel)
        tmp = model.joint_model(tmp)
        a = model.aggregation(tmp,  Mill.AlignedBags([1:l])) 
    else
        tmp = model.expr_model.joint_model(tmp, model.model_params.joint_model)
        a = model.expr_model.aggregation(tmp,  Mill.AlignedBags([1:l])) 
    end
    return a[:,1]
end


function not_int_cache(ex::Vector, cache)
    in_cache = []
    not_in_cache = []
    in_cache_order = []
    not_in_cache_order = []
    for (ind,i) in enumerate(ex)
        if haskey(cache, i)
            push!(in_cache, cache[i])
            push!(in_cache_order, ind)
        else
            push!(not_in_cache, i)
            push!(not_in_cache_order, ind)
        end
    end
    return in_cache, not_in_cache, in_cache_order, not_in_cache_order
end


# function not_int_cache(ex::Vector, cache)
#     in_cache_order = [ind for (ind, i) in enumerate(ex) if haskey(cache, i)]
#     not_in_cache_order = [ind for (ind, i) in enumerate(ex) if !haskey(cache, i)]

#     in_cache = [cache[ex[i]] for i in in_cache_order]
#     not_in_cache = [ex[i] for i in not_in_cache_order]

#     return in_cache, not_in_cache, in_cache_order, not_in_cache_order
# end


function batched_cached_inference!(exs::Vector,::Type{Expr}, cache, model, all_symbols, symbols_to_ind)
    in_cache, not_in_cache, in_cache_order, not_in_cache_order = not_int_cache(exs, cache)
    fun_indeces = []
    args_bags = []
    sym_args = Union{Symbol, Int, ExprWithHash}[]
    expr_args = Expr[]
    sym_bag_ids = Int[]
    expr_bag_ids = Int[]
    for (ind1,ex) in enumerate(not_in_cache)
        if ex.head == :call
            fun_name, args =  ex.args[1], ex.args[2:end]
        elseif ex.head in all_symbols
            fun_name = ex.head
            args = ex.args
        else
            error("unknown head $(ex.head)")
        end
        push!(fun_indeces, (ind1, symbols_to_ind[fun_name]))
        for (ind2,i) in enumerate(args)
            if !isa(i, Expr)
                push!(sym_args, i)
                push!(sym_bag_ids, ind1)
            else
                push!(expr_args, i)
                push!(expr_bag_ids, ind1)
            end
        end
    end
    args = batched_cached_inference!(sym_args, expr_args, cache, model, all_symbols, symbols_to_ind, vcat(sym_bag_ids, expr_bag_ids))
    encoding = sparse([i[2] for i in fun_indeces], [i[1] for i in fun_indeces], ones(Float32, length(fun_indeces)), length(all_symbols), length(exs))
    if isa(model, ExprModel)
        tmp = model.head_model(encoding)
    else
        tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
    end

    a = vcat(tmp, args)
    for (ind, i) in enumerate(not_in_cache_order)
        cache[exs[i]] = a[:, i]
    end
    for (cached,i) in zip(in_cache, in_cache_order)
        a[:, i] = cached
    end
    a
end


function batched_cached_inference!(args::Vector, ::Type{Symbol}, cache, model, all_symbols, symbols_to_ind)
    in_cache, not_in_cache, in_cache_order, not_in_cache_order = not_int_cache(args, cache)
    sym_indices = []
    values = Float32[]
    for (ind,i) in enumerate(not_in_cache)
        if isa(i, Number)
            push!(values, my_sigmoid(i))
        else
            push!(values, 1)
        end
        push!(sym_indices, (ind, isa(i, Symbol) ? symbols_to_ind[i] : symbols_to_ind[:Number]))
    end
    encoding = sparse([i[2] for i in sym_indices], [i[1] for i in sym_indices], values, length(all_symbols), length(args))
    if isa(model, ExprModel)
        tmp = model.head_model(encoding)
        # a = vcat(tmp, model.aggregation.:ψ)
        a = vcat(tmp, zeros(Float32, length(model.aggregation.:ψ), length(args)))
    else
        tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
        # a = vcat(tmp, model.expr_model.aggregation.:ψ)
        a = vcat(tmp, zeros(Float32, length(model.aggregation.:ψ), length(args)))
    end
    for (ind, i) in enumerate(not_in_cache_order)
        cache[args[i]] = a[:, i]
    end
    for (cached,i) in zip(in_cache, in_cache_order)
        a[:, i] = cached
    end
    return a
end


function batched_cached_inference!(sym_args::Vector, expr_args::Vector, cache, model, all_symbols, symbols_to_ind, args_bags)
    expr_my_tmp = []
    sym_my_tmp = []
    if !isempty(sym_args)
        sym_my_tmp = batched_cached_inference!(sym_args, Symbol, cache, model, all_symbols, symbols_to_ind)
    end
    if !isempty(expr_args)
        expr_my_tmp = batched_cached_inference!(expr_args, Expr, cache, model, all_symbols, symbols_to_ind)
    end
    if isempty(expr_my_tmp)
        my_args = sym_my_tmp
    elseif isempty(sym_my_tmp)
        my_args = expr_my_tmp
    else
        my_args = hcat(sym_my_tmp, expr_my_tmp)
    end
    # my_bags = vcat(sym_bag_ids, expr_bag_ids)


    tmp = vcat(my_args, zeros(2, size(my_args)[2]))
    bag_check = Dict()
    for (ind,bg) in enumerate(args_bags)
        if haskey(bag_check, bg)
            bag_check[bg] += 1
        else
            bag_check[bg] = 1
        end
        if bag_check[bg] == 1
            tmp[end-1:end,ind] = const_left
        elseif bag_check[bg] == 2
            tmp[end-1:end,ind] = const_right
        else
            tmp[end-1:end,ind] = const_third
        end
    end
    if isa(model, ExprModel)
        tmp = model.joint_model(tmp)
        a = model.aggregation(tmp,  Mill.ScatteredBags(args_bags)) 
    else
        tmp = model.expr_model.joint_model(tmp, model.model_params.joint_model)
        a = model.expr_model.aggregation(tmp,  Mill.ScatteredBags(args_bags)) 
    end
    return a
end



struct ExprModel{HM,A,JM,H}
    head_model::HM
    aggregation::A
    args_model::JM    
    heuristic::H
end

Flux.@layer ExprModel


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

function heuristic(m::ExprModel, ds)
    m.heuristic(m(ds))
end

function (m::ExprModel)(ds::ProductNode{<:NamedTuple{(:head,:args)}})
    head_model = m.head_model
    h = vcat(head_model.ms.head(ds.data.head),
        head_model.ms.args.m(m(ds.data.args)),
        )
    head_model.m(h)
end


function (m::ExprModel)(ds::ProductNode{<:NamedTuple{(:args,:position)}})
    args_model = m.args_model
    h = vcat(
        args_model.ms.args.m(m(ds.data.args)),
        args_model.ms.position(ds.data.position),
    )
    args_model.m(h)
end


function (m::ExprModel)(ds::BagNode)
    m.aggregation(m(ds.data), ds.bags)
end

function (m::ExprModel)(ds::DeduplicatingNode)
    # DeduplicatedMatrix(m(ds.x), ds.ii) # this might be slightly faster but might hit some corner cases
    m(ds.x)[:,ds.ii] # this is safer
end

function (m::ExprModel)(ds::BagNode{<:Missing})
    repeat(m.aggregation.ψ, 1, numobs(ds))
end

function (m::ExprModel)(ex::Union{Expr, Int}, cache, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
    ds = cached_inference!(ex, cache, m, all_symbols, symbols_to_index)
    tmp = vcat(ds, zeros(Float32, 2))
    m.heuristic(m.joint_model(tmp))
end


function (m::ExprModel)(ex::ExprWithHash, cache, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
    ds = cached_inference!(ex, typeof(ex.ex), cache, m, all_symbols, symbols_to_index)
    tmp = vcat(ds, zeros(Float32, 2))
    m.heuristic(m.joint_model(tmp))
end


function (m::ExprModel)(ex::Vector, cache, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
    ds = batched_cached_inference!(ex, Expr, cache, m, all_symbols, symbols_to_index)
    tmp = vcat(ds, zeros(Float32, 2, length(ex)))
    m.heuristic(m.joint_model(tmp))
end


function (m::ExprModel)(ex::Vector{ExprWithHash}, cache, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
    ds = batched_cached_inference!(ex, Expr, cache, m, all_symbols, symbols_to_index)
    tmp = vcat(ds, zeros(Float32, 2, length(ex)))
    m.heuristic(m.joint_model(tmp))
end

# function (m::ExprModel)(ex::Expr,::Type{Symbol}, cache, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
#     ds = cached_inference!(ExprWithHash(ex), typeof(ex), cache, m, all_symbols, symbols_to_index)
#     tmp = vcat(ds, zeros(Float32, 2))
#     m.heuristic(m.joint_model(tmp))
# end


function (m::ExprModelSimpleChains)(ex::ExprWithHash, cache, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
    ds = cached_inference!(ex, typeof(ex.ex), cache, m, all_symbols, symbols_to_index)
    tmp = vcat(ds, zeros(Float32, 2))
    m.expr_model.heuristic(m.expr_model.joint_model(tmp, m.model_params.joint_model), m.model_params.heuristic)
end


function (m::ExprModelSimpleChains)(ex::Expr, cache, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
    ds = cached_inference!(ex, cache, m, all_symbols, symbols_to_index)
    tmp = vcat(ds, zeros(Float32, 2))
    m.expr_model.heuristic(m.expr_model.joint_model(tmp, m.model_params.joint_model), m.model_params.heuristic)
end


# function (m::ExprModelSimpleChains)(ex::ExprWithHash, cache, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
#     ds = cached_inference!(ex, typeof(ex.ex), cache, m, all_symbols, symbols_to_index)
#     tmp = vcat(ds, zeros(Float32, 2))
#     m.expr_model.heuristic(m.expr_model.joint_model(tmp, m.model_params.joint_model), m.model_params.heuristic)
# end

function simple_to_flux(m1::ExprModelSimpleChains, m2::ExprModel)
    simple_weights = SimpleChains.weights(m1.expr_model.joint_model, m1.model_params.joint_model)
    for i in 1:length(m2.joint_model.layers)
        m2.joint_model.layers[i].weight .= simple_weights[i]
    end
    simple_weights = SimpleChains.weights(m1.expr_model.head_model, m1.model_params.head_model)
    for i in 1:length(m2.head_model.layers)
        m2.head_model.layers[i].weight .= simple_weights[i]
    end
    simple_weights = SimpleChains.weights(m1.expr_model.heuristic, m1.model_params.heuristic)
    for i in 1:length(m2.heuristic.layers)
        m2.heuristic.layers[i].weight .= simple_weights[i]
    end
    return m2
end


function flux_to_simple(m1::ExprModelSimpleChains, m2::ExprModel)
    simple_weights = SimpleChains.weights(m1.expr_model.joint_model, m1.model_params.joint_model)
    simple_biases = SimpleChains.biases(m1.expr_model.joint_model, m1.model_params.joint_model)
    for i in 1:length(m2.joint_model.layers)
        simple_weights[i] .= m2.joint_model.layers[i].weight
        simple_biases[i] .= m2.joint_model.layers[i].bias 
    end
    simple_weights = SimpleChains.weights(m1.expr_model.head_model, m1.model_params.head_model)
    simple_biases = SimpleChains.biases(m1.expr_model.head_model, m1.model_params.head_model)
    for i in 1:length(m2.head_model.layers)
        simple_weights[i] .= m2.head_model.layers[i].weight
        simple_biases[i] .= m2.head_model.layers[i].bias 
    end
    simple_weights = SimpleChains.weights(m1.expr_model.heuristic, m1.model_params.heuristic)
    simple_biases = SimpleChains.biases(m1.expr_model.heuristic, m1.model_params.heuristic)
    for i in 1:length(m2.heuristic.layers)
        simple_weights[i] .= m2.heuristic.layers[i].weight
        simple_biases[i] .= m2.heuristic.layers[i].bias
    end 
    return m1
end

embed(m::ExprModel, ds::Missing) = missing


logistic(x) = log(1 + exp(x))
hinge(x) = min(0, max(1, x + 1))
# loss01(x) = x + 1 > 0 ? x + 1 : 0


function batched_loss(heuristic, big_vector, hp, hn, bags_of_batches, surrogate::Function = hinge)
    o = heuristic(big_vector)
    p = (o * hp) .* hn

    diff = p - o[1, :] .* hn
    diff = map(bg->filter(!=(0), diff[bg]), bags_of_batches)
    filtered_diff = filter(!=(0), diff)
    return sum(surrogate.(filtered_diff))
end


# function loss(heuristic, big_vector::Vector, hp=nothing, hn=nothing, surrogate::Function = softplus, cache = LRU(maxsize=10000))
#     o = map(x->only(heuristic(x, cache)), big_vector)
#     # @show size(o)
#     p = (transpose(o) * hp) .* hn

#     diff = p - o .* hn
#     filtered_diff = filter(!=(0), diff)
#     return sum(surrogate.(filtered_diff))
# end


function loss(heuristic, big_vector::ProductNode, hp::Vector, hn::Vector, surrogate::Function = hinge)
    o = heuristic(big_vector)
    diff = o[hp] .- o[hn]
    # return sum(surrogate.(diff))
    # return sum(filter(x->x>=0, diff))
    return sum(count(>(0), diff))
end


# function loss(heuristic, big_vector::ProductNode, hp::Matrix, hn::Matrix, surrogate::Function = softplus)
#     o = heuristic(big_vector)
#     p = (o * hp) .* hn

#     diff = p - o[1, :] .* hn
#     filtered_diff = filter(!=(0), diff)
#     return sum(surrogate.(filtered_diff))
# end


# function loss(heuristic, big_vector::ProductNode, hp::Vector, hn::Vector, surrogate::Function = softplus)
#     o = vec(heuristic(big_vector))
#     diff = o[hp] - o[hn]
#     # return sum(surrogate.(diff))
#     return sum(filter(x->x>=0, diff))
# end


function loss(heuristic, big_vector::ProductNode, hp, hn, surrogate::Function = softplus, agg::Function = mean)
    o = vec(heuristic(big_vector))
    return agg(surrogate.(o[hp] - o[hn]))
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


# function loss(heuristic::ExprModel, params::InitExprModelDense, big_vector, hp, hn, surrogate::Function = logistic)
#     o = heuristic(big_vector, params)
#     p = (o * hp) .* hn

#     diff = p - o[1, :] .* hn
#     filtered_diff = filter(!=(0), diff)
#     # return sum(log.(1 .+ exp.(filtered_diff)))
#     # return mean(softmax(filtered_diff))
#     return sum(surrogate.(filtered_diff))
# end


struct MySimpleChainsLoss{T,Y<:AbstractVector{T}}<:SimpleChains.AbstractLoss{T}
    targets::Y
end

SimpleChains.target(loss::MySimpleChainsLoss) = loss.targets
(loss::MySimpleChainsLoss)(x::AbstractArray) = MySimpleChainsLoss(x)

function (loss::MySimpleChainsLoss)(prev_layer_out::AbstractArray{T}, p::Ptr, pu) where {T}
    total_loss = loss(big_vector, hp, hn)
    return total_loss, p, pu
end

function SimpleChains.layer_output_size(::Val{T}, sl::MySimpleChainsLoss, s::Tuple) where {T}
    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)
end
function SimpleChains.forward_layer_output_size(::Val{T}, sl::MySimpleChainsLoss, s) where {T}
    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)
end

# Todo: implement back prop for your model 
function SimpleChains.chain_valgrad!(
    __,
    previous_layer_output::AbstractArray{T},
    layers::Tuple{MySimpleChainsLoss},
    _::Ptr,
    pu::Ptr{UInt8},
) where {T}
    loss = getfield(layers, 1)
    total_loss = calculate_loss(loss, previous_layer_output)
    y = loss.targets

    # Store the backpropagated gradient in the previous_layer_output array.
    for i in eachindex(y)
        sign_arg = 2 * y[i] - 1
        # Get the value of the last logit
        logit_i = previous_layer_output[i]
        previous_layer_output[i] = -(sign_arg * inv(1 + exp(sign_arg * logit_i)))
    end

    return total_loss, previous_layer_output, pu
end