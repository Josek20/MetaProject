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


@my_cache function interned_cached_inference_simple_chains!(ex::NodeID, ::Type{Expr}, model::ExprModelSimpleChains, all_symbols, symbols_to_ind)
    node = nc[ex]
    fun_name = node.head
    args = interned_cached_inference_simple_chains!([node.left, node.right], model, all_symbols, symbols_to_ind)
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_index[fun_name]] = 1
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(args),
        )
    head_model.m(h)
end


@my_cache function interned_cached_inference_simple_chains!(ex::NodeID, ::Type{Symbol}, model::ExprModelSimpleChains, all_symbols, symbols_to_ind)
    node = nc[ex]
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_ind[node.head]] = 1
    zero_bag = repeat(model.aggregation.ψ, 1, 1)
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(zero_bag),
        )
    head_model.m(h)
end


@my_cache function interned_cached_inference_simple_chains!(ex::NodeID, ::Type{Int}, model::ExprModelSimpleChains, all_symbols, symbols_to_ind)
    node = nc[ex]
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_ind[:Number]] = my_sigmoid(node.v)

    zero_bag = repeat(model.aggregation.ψ, 1, 1)
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(zero_bag),
        )
    head_model.m(h)
end


function interned_cached_inference_simple_chains!(args::Vector{NodeID}, model, all_symbols, symbols_to_ind)
    left_node, right_node = args
    tmp = []
    if left_node != nullid
        inference_type = nc[left_node].head ∈ (:float, :integer) ? Int : Symbol
        inference_type = nc[left_node].iscall ? Expr : inference_type 
        left_infer = interned_cached_inference_simple_chains!(left_node, inference_type, model, all_symbols, symbols_to_ind)
        tmp_left = left_infer
        push!(tmp, tmp_left)
    end
    if right_node != nullid
        inference_type = nc[right_node].head ∈ (:float, :integer) ? Int : Symbol
        inference_type = nc[right_node].iscall ? Expr : inference_type 
        right_infer = interned_cached_inference_simple_chains!(right_node, inference_type, model, all_symbols, symbols_to_ind)
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
end


@my_cache function hashed_expr_cached_inference_simple_chains!(ex::ExprWithHash, ::Type{Expr}, model::ExprModelSimpleChains, all_symbols, symbols_to_ind)
    args = ex.args
    fun_name = ex.head
    args = hashed_expr_cached_inference_simple_chains!(args, model, all_symbols, symbols_to_ind)
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_index[fun_name]] = 1
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(args),
        )
    head_model.m(h)
end


@my_cache function hashed_expr_cached_inference_simple_chains!(ex::ExprWithHash, ::Type{Symbol}, model::ExprModelSimpleChains, all_symbols, symbols_to_ind)
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_ind[ex.head]] = 1
    zero_bag = repeat(model.aggregation.ψ, 1, 1)
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(zero_bag),
        )
    head_model.m(h)
end


@my_cache function hashed_expr_cached_inference_simple_chains!(ex::ExprWithHash, ::Type{Int}, model::ExprModelSimpleChains, all_symbols, symbols_to_ind)
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_ind[:Number]] = my_sigmoid(ex.head)
    zero_bag = repeat(model.aggregation.ψ, 1, 1)
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(zero_bag),
        )
    head_model.m(h)
end


function hashed_expr_cached_inference_simple_chains!(args::Vector{MyModule.ExprWithHash}, model::ExprModelSimpleChains, all_symbols, symbols_to_ind)
    l = length(args)
    my_tmp = []
    for i in args
        if isa(i.head, Symbol) && symbols_to_ind[i.head] <= 18
            pl = hashed_expr_cached_inference_simple_chains!(i, Expr, model, all_symbols, symbols_to_ind)
        else
            if isa(i.head, Symbol)
                pl = hashed_expr_cached_inference_simple_chains!(i, Symbol, model, all_symbols, symbols_to_ind)
            else
                pl = hashed_expr_cached_inference_simple_chains!(i, Int, model, all_symbols, symbols_to_ind)
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
end


@my_cache function expr_cached_inference_simple_chains!(ex::Expr, model, all_symbols, symbols_to_ind)
    if ex.head == :call
        fun_name, args =  ex.args[1], ex.args[2:end]
    elseif ex.head in all_symbols
        fun_name = ex.head
        args = ex.args
    else
        error("unknown head $(ex.head)")
    end
    args = expr_cached_inference_simple_chains!(args, model, all_symbols, symbols_to_ind)
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_index[fun_name]] = 1
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(args),
        )
    head_model.m(h)
end


@my_cache function expr_cached_inference_simple_chains!(ex::Symbol, model, all_symbols, symbols_to_ind)
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_ind[ex]] = 1
    zero_bag = repeat(model.aggregation.ψ, 1, 1)
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(zero_bag),
        )
    head_model.m(h)
end


@my_cache function expr_cached_inference_simple_chains!(ex::Number, model, all_symbols, symbols_to_ind)
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_ind[:Number]] = my_sigmoid(ex)
    zero_bag = repeat(model.aggregation.ψ, 1, 1)
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(zero_bag),
        )
    head_model.m(h)
end


function expr_cached_inference_simple_chains!(args::Vector{Union{Expr, Symbol, Number}}, model, all_symbols, symbols_to_ind)
    l = length(args)
    my_tmp = [expr_cached_inference_simple_chains!(a, model, all_symbols, symbols_to_ind) for a in args]
    my_args = hcat(my_tmp...)
    args_model = model.args_model
    if l == 3
        t = const_nth
    elseif l == 2
        t = const_both
    else
        t = const_left
    end
    h = vcat(
        args_model.ms.args.m(my_args),
        args_model.ms.position.m(t),
    )
    positional_encoding = args_model.m(h)
    model.aggregation(positional_encoding, Mill.AlignedBags([1:l]))
end


function (m::ExprModelSimpleChains)(ex::ExprWithHash, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
    ds = hashed_expr_cached_inference_simple_chains!(ex, typeof(ex.ex), m, all_symbols, symbols_to_index)
    # tmp = vcat(ds, zeros(Float32, 2))
    # m.expr_model.heuristic(m.expr_model.args_model(tmp, m.model_params.args_model), m.model_params.heuristic)
end


function (m::ExprModelSimpleChains)(ex::Expr, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
    ds = expr_cached_inference_simple_chains!(ex, m, all_symbols, symbols_to_index)
    # tmp = vcat(ds, zeros(Float32, 2))
    # m.expr_model.heuristic(m.expr_model.args_model(tmp, m.model_params.args_model), m.model_params.heuristic)
end


function (m::ExprModelSimpleChains)(ex::NodeID, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
    ds = interned_cached_inference_simple_chains!(ex, typeof(ex.ex), m, all_symbols, symbols_to_index)
    # tmp = vcat(ds, zeros(Float32, 2))
    # m.expr_model.heuristic(m.expr_model.args_model(tmp, m.model_params.args_model), m.model_params.heuristic)
end
