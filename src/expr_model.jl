struct ExprModel{HM,A,JM,H}
    head_model::HM
    aggregation::A
    args_model::JM    
    heuristic::H
end


Flux.@layer ExprModel


const const_left = [1, 0]
const const_right = [0, 1]
const const_one = [0, 0]
const const_third = [1, 1]
const const_both = [1 0; 0 1]
const const_nth = [1 0 1; 0 1 1]


my_sigmoid(x, k = 0.01, m = 0) = 1 / (1 + exp(-k * (x - m)))


@my_cache function get_head_and_args(ex::NodeID)
    node = nc[ex]
    fun_name = node.head
    (;args=[node.left, node.right], fun_name=fun_name)
end


@my_cache function get_head_and_args(ex::ExprWithHash)
    args = ex.args
    fun_name = ex.head
    (;args=args, fun_name=fun_name)
end


@my_cache function get_head_and_args(ex::Expr)
    if ex.head == :call
        fun_name, args =  ex.args[1], ex.args[2:end]
    elseif ex.head in all_symbols
        fun_name = ex.head
        args = ex.args
    else
        error("unknown head $(ex.head)")
    end
    (;args=args, fun_name=fun_name)
end


@my_cache function get_leaf_args(ex::NodeID)
    node = nc[ex]
    symbol_index = node.head ∈ (:integer, :float) ? :Number : node.head
    encoding_value = node.head ∈ (:integer, :float) ? my_sigmoid(node.v) : 1
    (;symbol_index=symbol_index,encoding_value=encoding_value)
end


@my_cache function get_leaf_args(ex::Union{Symbol, Int})
    symbol_index = ex isa Symbol ? ex : :Number
    encoding_value = ex isa Symbol ? 1 : my_sigmoid(node.v)
    (;symbol_index=symbol_index,encoding_value=encoding_value)
end


@my_cache function get_leaf_args(ex::ExprWithHash)
    symbol_index = ex.head isa Symbol ? ex : :Number
    encoding_value = ex.head isa Symbol ? 1 : my_sigmoid(node.v)
    (;symbol_index=symbol_index,encoding_value=encoding_value)
end


@my_cache function general_cached_inference(ex::Union{Expr, NodeID, ExprWithHash}, ::Type{Expr}, model, all_symbols, symbols_to_ind)
    args, fun_name = get_head_and_args(ex)
    args = general_scached_inference!(args, model, all_symbols, symbols_to_ind)
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_index[fun_name]] = 1
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(args),
        )
    head_model.m(h)
end


@my_cache function general_cached_inference(ex::Union{Symbol, NodeID, ExprWithHash, Int}, ::Union{Type{Symbol}, Type{Int}}, model, all_symbols, symbols_to_ind)
    symbol_index, encoding_value = get_leaf_args(ex)
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_ind[symbol_index]] = encoding_value
    zero_bag = repeat(model.aggregation.ψ, 1, 1)
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(zero_bag),
        )
    head_model.m(h)
end

function some_hashed()
    
    for i in args
        if isa(i.head, Symbol) && symbols_to_ind[i.head] <= 18
            pl = hashed_expr_cached_inference!(i, Expr, model, all_symbols, symbols_to_ind)
        else
            if isa(i.head, Symbol)
                pl = hashed_expr_cached_inference!(i, Symbol, model, all_symbols, symbols_to_ind)
            else
                pl = hashed_expr_cached_inference!(i, Int, model, all_symbols, symbols_to_ind)
            end
        end
        push!(my_tmp, pl)
    end
end

function some_node()
    tmp = []
    if left_node != nullid
        inference_type = nc[left_node].head ∈ (:float, :integer) ? Int : Symbol
        inference_type = nc[left_node].iscall ? Expr : inference_type 
        left_infer = interned_cached_inference!(left_node, inference_type, model, all_symbols, symbols_to_ind)
        tmp_left = left_infer
        push!(tmp, tmp_left)
    end
    if right_node != nullid
        inference_type = nc[right_node].head ∈ (:float, :integer) ? Int : Symbol
        inference_type = nc[right_node].iscall ? Expr : inference_type 
        right_infer = interned_cached_inference!(right_node, inference_type, model, all_symbols, symbols_to_ind)
        tmp_right = right_infer
        push!(tmp, tmp_right)
    end
end

function some_expr()
    my_tmp = [expr_cached_inference!(a, model, all_symbols, symbols_to_ind) for a in args]
end
    
@my_cache function general_cached_inference(args::Vector, model, all_symbols, symbols_to_ind)
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



@my_cache LRU(maxsize=100_000) function interned_cached_inference!(ex::NodeID, ::Type{Expr}, model, all_symbols, symbols_to_ind)
    node = nc[ex]
    fun_name = node.head
    args = interned_cached_inference!([node.left, node.right], model, all_symbols, symbols_to_ind)
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_index[fun_name]] = 1
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(args),
        )
    head_model.m(h)
end


@my_cache LRU(maxsize=100_000) function interned_cached_inference!(ex::NodeID, ::Type{Symbol}, model, all_symbols, symbols_to_ind)
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


@my_cache LRU(maxsize=100_000) function interned_cached_inference!(ex::NodeID, ::Type{Int}, model, all_symbols, symbols_to_ind)
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


function interned_cached_inference!(args::Vector{NodeID}, model, all_symbols, symbols_to_ind)
    left_node, right_node = args
    tmp = []
    if left_node != nullid
        inference_type = nc[left_node].head ∈ (:float, :integer) ? Int : Symbol
        inference_type = nc[left_node].iscall ? Expr : inference_type 
        left_infer = interned_cached_inference!(left_node, inference_type, model, all_symbols, symbols_to_ind)
        tmp_left = left_infer
        push!(tmp, tmp_left)
    end
    if right_node != nullid
        inference_type = nc[right_node].head ∈ (:float, :integer) ? Int : Symbol
        inference_type = nc[right_node].iscall ? Expr : inference_type 
        right_infer = interned_cached_inference!(right_node, inference_type, model, all_symbols, symbols_to_ind)
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


@my_cache function hashed_expr_cached_inference!(ex::ExprWithHash, ::Type{Expr}, model, all_symbols, symbols_to_ind)
    args = ex.args
    fun_name = ex.head
    args = hashed_expr_cached_inference!(args, model, all_symbols, symbols_to_ind)
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_index[fun_name]] = 1
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(args),
        )
    head_model.m(h)
end


@my_cache function hashed_expr_cached_inference!(ex::ExprWithHash, ::Type{Symbol}, model, all_symbols, symbols_to_ind)
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_ind[ex.head]] = 1
    zero_bag = repeat(model.aggregation.ψ, 1, 1)
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(zero_bag),
        )
    head_model.m(h)
end


@my_cache function hashed_expr_cached_inference!(ex::ExprWithHash, ::Type{Int}, model, all_symbols, symbols_to_ind)
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_ind[:Number]] = my_sigmoid(ex.head)
    zero_bag = repeat(model.aggregation.ψ, 1, 1)
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(zero_bag),
        )
    head_model.m(h)
end


function hashed_expr_cached_inference!(args::Vector{MyModule.ExprWithHash}, model, all_symbols, symbols_to_ind)
    l = length(args)
    my_tmp = []
    for i in args
        if isa(i.head, Symbol) && symbols_to_ind[i.head] <= 18
            pl = hashed_expr_cached_inference!(i, Expr, model, all_symbols, symbols_to_ind)
        else
            if isa(i.head, Symbol)
                pl = hashed_expr_cached_inference!(i, Symbol, model, all_symbols, symbols_to_ind)
            else
                pl = hashed_expr_cached_inference!(i, Int, model, all_symbols, symbols_to_ind)
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


@my_cache function expr_cached_inference!(ex::Expr, model, all_symbols, symbols_to_ind)
    if ex.head == :call
        fun_name, args =  ex.args[1], ex.args[2:end]
    elseif ex.head in all_symbols
        fun_name = ex.head
        args = ex.args
    else
        error("unknown head $(ex.head)")
    end
    args = expr_cached_inference!(args, model, all_symbols, symbols_to_ind)
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_index[fun_name]] = 1
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(args),
        )
    head_model.m(h)
end


@my_cache function expr_cached_inference!(ex::Symbol, model, all_symbols, symbols_to_ind)
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_ind[ex]] = 1
    zero_bag = repeat(model.aggregation.ψ, 1, 1)
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(zero_bag),
        )
    head_model.m(h)
end


@my_cache function expr_cached_inference!(ex::Number, model, all_symbols, symbols_to_ind)
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_ind[:Number]] = my_sigmoid(ex)
    zero_bag = repeat(model.aggregation.ψ, 1, 1)
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(zero_bag),
        )
    head_model.m(h)
end


function expr_cached_inference!(args::Vector, model, all_symbols, symbols_to_ind)
    l = length(args)
    my_tmp = [expr_cached_inference!(a, model, all_symbols, symbols_to_ind) for a in args]
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
    node = MyModule.nc[ex]
    inference_type = node.head ∈ (:integer, :float) ? Int : Symbol
    inference_type = node.iscall ? Expr : inference_type 
    ds = interned_cached_inference!(ex, inference_type, m, all_symbols, symbols_to_index)
    m.heuristic(ds)
end


function (m::ExprModel)(ex::Union{Expr, Int}, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
    ds = expr_cached_inference!(ex, m, all_symbols, symbols_to_index)
    m.heuristic(ds)
end


function (m::ExprModel)(ex::ExprWithHash, all_symbols=new_all_symbols, symbols_to_index=sym_enc)
    ds = hashed_expr_cached_inference!(ex, typeof(ex.ex), m, all_symbols, symbols_to_index)
    m.heuristic(ds)
end