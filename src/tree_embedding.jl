const const_left = [1, 0]
const const_right = [0, 1]
const const_one = [0, 0]
const const_third = [1, 1]
const const_both = [1 0; 0 1]
const const_nth = [1 0 1; 0 1 1]
my_sigmoid(x, k = 0.01, m = 0) = 1 / (1 + exp(-k * (x - m)))


@my_cache function cached_inference!(ex::NodeID, ::Type{Expr}, model, all_symbols, symbols_to_ind)
    # get!(cache, ex) do
        node = nc[ex]
        fun_name = node.head
        args = cached_inference!([node.left, node.right], model, all_symbols, symbols_to_ind)
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_index[fun_name]] = 1
        # if isa(model, ExprModel)
        # tmp = model.head_model.ms.head.m(encoding)
        head_model = model.head_model
        h = vcat(head_model.ms.head.m(encoding),
            head_model.ms.args.m(args),
            )
        head_model.m(h)
        # else
        #     tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
        # end
        # tmp = vcat(tmp, args)
    # end
end


@my_cache function cached_inference!(ex::NodeID, ::Type{Symbol}, model, all_symbols, symbols_to_ind)
    # get!(cache, ex) do
        node = nc[ex]
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_ind[node.head]] = 1
        zero_bag = repeat(model.aggregation.ψ, 1, 1)
        # if isa(model, ExprModel)
        head_model = model.head_model
        h = vcat(head_model.ms.head.m(encoding),
            head_model.ms.args.m(zero_bag),
            )
        head_model.m(h)
        # tmp = model.head_model.ms.head.m(encoding)
        # a = vcat(tmp, model.aggregation.:ψ) 
        # else
        #     tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
        #     a = vcat(tmp, model.expr_model.aggregation.:ψ)
        # end
        # return a
    # end
end


@my_cache function cached_inference!(ex::NodeID, ::Type{Int}, model, all_symbols, symbols_to_ind)
    # get!(cache, ex) do
        node = nc[ex]
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_ind[:Number]] = my_sigmoid(node.v)

        zero_bag = repeat(model.aggregation.ψ, 1, 1)
        head_model = model.head_model
        h = vcat(head_model.ms.head.m(encoding),
            head_model.ms.args.m(zero_bag),
            )
        head_model.m(h)
        # if isa(model, ExprModel)
        # tmp = model.head_model.ms.head.m(encoding)
        # a = vcat(tmp, model.aggregation.:ψ) 
        # else
        #     tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
        #     a = vcat(tmp, model.expr_model.aggregation.:ψ)
        # end
        # return a
    # end
end


function cached_inference!(args::Vector{NodeID}, model, all_symbols, symbols_to_ind)
    left_node, right_node = args
    tmp = []
    if left_node != nullid
        inference_type = nc[left_node].head ∈ (:float, :integer) ? Int : Symbol
        inference_type = nc[left_node].iscall ? Expr : inference_type 
        left_infer = cached_inference!(left_node, inference_type, model, all_symbols, symbols_to_ind)
        # tmp_left = vcat(left_infer, const_left)
        tmp_left = left_infer
        push!(tmp, tmp_left)
    end
    if right_node != nullid
        inference_type = nc[right_node].head ∈ (:float, :integer) ? Int : Symbol
        inference_type = nc[right_node].iscall ? Expr : inference_type 
        right_infer = cached_inference!(right_node, inference_type, model, all_symbols, symbols_to_ind)
        # tmp_right = vcat(right_infer, const_right)
        tmp_right = right_infer
        push!(tmp, tmp_right)
    end
    l = length(tmp)
    tmp = hcat(tmp...)
    args_model = model.args_model
    h = vcat(
        args_model.ms.args.m(tmp),
        args_model.ms.position.m(l == 2 ? const_both : const_left),
    )
    positional_encoding = args_model.m(h)
    model.aggregation(positional_encoding, Mill.AlignedBags([1:l]))
    # else
    #     tmp = model.expr_model.args_model(tmp, model.model_params.args_model)
    #     a = model.expr_model.aggregation(tmp,  Mill.AlignedBags([1:l])) 
    # end
    # return a[:,1]
end


@my_cache function cached_inference!(ex::ExprWithHash, ::Type{Expr}, model, all_symbols, symbols_to_ind)
    # get!(cache, ex) do
        args = ex.args
        fun_name = ex.head
        args = cached_inference!(args, model, all_symbols, symbols_to_ind)
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_index[fun_name]] = 1
        head_model = model.head_model
        h = vcat(head_model.ms.head.m(encoding),
            head_model.ms.args.m(args),
            )
        head_model.m(h)
        # if isa(model, ExprModel)
        #     tmp = model.head_model(encoding)
        # else
        #     tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
        # end
        # tmp = vcat(tmp, args)
    # end
end


@my_cache function cached_inference!(ex::ExprWithHash, ::Type{Symbol}, model, all_symbols, symbols_to_ind)
    # get!(cache, ex) do
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_ind[ex.head]] = 1
        zero_bag = repeat(model.aggregation.ψ, 1, 1)
        head_model = model.head_model
        h = vcat(head_model.ms.head.m(encoding),
            head_model.ms.args.m(zero_bag),
            )
        head_model.m(h)
        # if isa(model, ExprModel)
        #     tmp = model.head_model(encoding)
        #     a = vcat(tmp, model.aggregation.:ψ) 
        # else
        #     tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
        #     a = vcat(tmp, model.expr_model.aggregation.:ψ)
        # end
        # return a
    # end
end


@my_cache function cached_inference!(ex::ExprWithHash, ::Type{Int}, model, all_symbols, symbols_to_ind)
    # get!(cache, ex) do
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_ind[:Number]] = my_sigmoid(ex.head)
        zero_bag = repeat(model.aggregation.ψ, 1, 1)
        head_model = model.head_model
        h = vcat(head_model.ms.head.m(encoding),
            head_model.ms.args.m(zero_bag),
            )
        head_model.m(h)
        # if isa(model, ExprModel)
        #     tmp = model.head_model(encoding)
        #     a = vcat(tmp, model.aggregation.:ψ)
        # else
        #     tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
        #     a = vcat(tmp, model.expr_model.aggregation.:ψ)
        # end
        # return a
    # end
end


@my_cache function cached_inference!(ex::Expr, model, all_symbols, symbols_to_ind)
    # get!(cache, ex) do
        if ex.head == :call
            fun_name, args =  ex.args[1], ex.args[2:end]
        elseif ex.head in all_symbols
            fun_name = ex.head
            args = ex.args
        else
            error("unknown head $(ex.head)")
        end
        args = cached_inference!(args, model, all_symbols, symbols_to_ind)
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_index[fun_name]] = 1
        head_model = model.head_model
        h = vcat(head_model.ms.head.m(encoding),
            head_model.ms.args.m(args),
            )
        head_model.m(h)
        # if isa(model, ExprModel)
        #     tmp = model.head_model(encoding)
        # else
        #     tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
        # end
        # tmp = vcat(tmp, args)
    # end
end


@my_cache function cached_inference!(ex::Symbol, model, all_symbols, symbols_to_ind)
    # get!(cache, ex) do
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_ind[ex]] = 1
        zero_bag = repeat(model.aggregation.ψ, 1, 1)
        head_model = model.head_model
        h = vcat(head_model.ms.head.m(encoding),
            head_model.ms.args.m(zero_bag),
            )
        head_model.m(h)
        # if isa(model, ExprModel)
        #     tmp = model.head_model(encoding)
        #     a = vcat(tmp, model.aggregation.:ψ)
        # else
        #     tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
        #     a = vcat(tmp, model.expr_model.aggregation.:ψ)
        # end
        # return a
    # end
end


@my_cache function cached_inference!(ex::Number, model, all_symbols, symbols_to_ind)
    # get!(cache, ex) do
        encoding = zeros(Float32, length(all_symbols))
        encoding[symbols_to_ind[:Number]] = my_sigmoid(ex)
        zero_bag = repeat(model.aggregation.ψ, 1, 1)
        head_model = model.head_model
        h = vcat(head_model.ms.head.m(encoding),
            head_model.ms.args.m(zero_bag),
            )
        head_model.m(h)
        # if isa(model, ExprModel)
        #     tmp = model.head_model(encoding)
        #     a = vcat(tmp, model.aggregation.:ψ)
        # else
        #     tmp = model.expr_model.head_model(encoding, model.model_params.head_model)
        #     a = vcat(tmp, model.expr_model.aggregation.:ψ)
        # end
        # return a
    # end
end


function cached_inference!(args::Vector, model, all_symbols, symbols_to_ind)
    l = length(args)
    my_tmp = [cached_inference!(a, model, all_symbols, symbols_to_ind) for a in args]
    my_args = hcat(my_tmp...)
    args_model = model.args_model
    if l == 3
        t = const_nth
    elseif l == 2
        t = const_both
    else
        t = const_left
    end
    # t = l == 2 ? const_both : const_left
    h = vcat(
        args_model.ms.args.m(my_args),
        args_model.ms.position.m(t),
    )
    positional_encoding = args_model.m(h)
    model.aggregation(positional_encoding, Mill.AlignedBags([1:l]))
    # if l == 2
    #     tmp = vcat(my_args, const_both)
    # elseif l == 3
    #     tmp = vcat(my_args, const_nth)
    # else
    #     tmp = vcat(my_args, const_left)
    # end

    # if isa(model, ExprModel)
    #     tmp = model.args_model(tmp)
    #     a = model.aggregation(tmp,  Mill.AlignedBags([1:l])) 
    # else
    #     tmp = model.expr_model.args_model(tmp, model.model_params.args_model)
    #     a = model.expr_model.aggregation(tmp,  Mill.AlignedBags([1:l])) 
    # end
    # return a[:,1]
end


function cached_inference!(args::Vector{MyModule.ExprWithHash}, cache, model, all_symbols, symbols_to_ind)
    l = length(args)
    my_tmp = []
    for i in args
        if isa(i.head, Symbol) && symbols_to_ind[i.head] <= 18
            pl = cached_inference!(i, Expr, model, all_symbols, symbols_to_ind)
        else
            if isa(i.head, Symbol)
                pl = cached_inference!(i, Symbol, model, all_symbols, symbols_to_ind)
            else
                pl = cached_inference!(i, Int, model, all_symbols, symbols_to_ind)
            end
        end
        push!(my_tmp, pl)
    end
    my_args = hcat(my_tmp...)
    args_model = model.args_model
    h = vcat(
        args_model.ms.args.m(my_args),
        args_model.ms.position.m(l == 2 ? const_both : const_left),
    )
    positional_encoding = args_model.m(h)
    model.aggregation(positional_encoding, Mill.AlignedBags([1:l]))
    # if l == 2
    #     tmp = vcat(my_args, const_both)
    # elseif l == 3
    #     tmp = vcat(my_args, const_nth)
    # else
    #     tmp = vcat(my_args, const_one)
    # end
    # if isa(model, ExprModel)
    #     tmp = model.args_model(tmp)
    #     a = model.aggregation(tmp,  Mill.AlignedBags([1:l])) 
    # else
    #     tmp = model.expr_model.args_model(tmp, model.model_params.args_model)
    #     a = model.expr_model.aggregation(tmp,  Mill.AlignedBags([1:l])) 
    # end
    # return a[:,1]
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
        tmp = model.args_model(tmp)
        a = model.aggregation(tmp,  Mill.ScatteredBags(args_bags)) 
    else
        tmp = model.expr_model.args_model(tmp, model.model_params.args_model)
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
    args_model::JM 
    heuristic::H 
end


function InitExprModelDense(m::ExprModel)
    hm = SimpleChains.init_params(m.head_model)
    jm = SimpleChains.init_params(m.args_model)
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


function (m::ExprModel)(ex::NodeID, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
    ds = cached_inference!(ex, Expr, m, all_symbols, symbols_to_index)
    # tmp = vcat(ds, zeros(Float32, 2))
    m.heuristic(ds)
end


function (m::ExprModel)(ex::Union{Expr, Int}, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
    ds = cached_inference!(ex, m, all_symbols, symbols_to_index)
    # tmp = vcat(ds, zeros(Float32, 2))
    # m.heuristic(m.args_model(tmp))
    m.heuristic(ds)
end


function (m::ExprModel)(ex::ExprWithHash, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
    ds = cached_inference!(ex, typeof(ex.ex), m, all_symbols, symbols_to_index)
    # tmp = vcat(ds, zeros(Float32, 2))
    # m.heuristic(m.args_model(tmp))
    m.heuristic(ds)
end


function (m::ExprModel)(ex::Vector, cache, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
    ds = batched_cached_inference!(ex, Expr, cache, m, all_symbols, symbols_to_index)
    # tmp = vcat(ds, zeros(Float32, 2, length(ex)))
    # m.heuristic(m.args_model(tmp))
    m.heuristic(ds)
end


function (m::ExprModel)(ex::Vector{ExprWithHash}, cache, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
    ds = batched_cached_inference!(ex, Expr, cache, m, all_symbols, symbols_to_index)
    # tmp = vcat(ds, zeros(Float32, 2, length(ex)))
    # m.heuristic(m.args_model(tmp))
    m.heuristic(ds)
end

# function (m::ExprModel)(ex::Expr,::Type{Symbol}, cache, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
#     ds = cached_inference!(ExprWithHash(ex), typeof(ex), cache, m, all_symbols, symbols_to_index)
#     tmp = vcat(ds, zeros(Float32, 2))
#     m.heuristic(m.args_model(tmp))
# end




# function (m::ExprModelSimpleChains)(ex::ExprWithHash, cache, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
#     ds = cached_inference!(ex, typeof(ex.ex), cache, m, all_symbols, symbols_to_index)
#     tmp = vcat(ds, zeros(Float32, 2))
#     m.expr_model.heuristic(m.expr_model.args_model(tmp, m.model_params.args_model), m.model_params.heuristic)
# end

function simple_to_flux(m1::ExprModelSimpleChains, m2::ExprModel)
    simple_weights = SimpleChains.weights(m1.expr_model.args_model, m1.model_params.args_model)
    for i in 1:length(m2.args_model.layers)
        m2.args_model.layers[i].weight .= simple_weights[i]
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
    simple_weights = SimpleChains.weights(m1.expr_model.args_model, m1.model_params.args_model)
    simple_biases = SimpleChains.biases(m1.expr_model.args_model, m1.model_params.args_model)
    for i in 1:length(m2.args_model.layers)
        simple_weights[i] .= m2.args_model.layers[i].weight
        simple_biases[i] .= m2.args_model.layers[i].bias 
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

function loss(heuristic, big_vector::ProductNode, hp, hn, surrogate::Function = softplus, agg::Function = mean)
    o = vec(heuristic(big_vector))
    return agg(surrogate.(o[hp] - o[hn]))
end
