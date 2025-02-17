struct ExprModel{HM,A,JM,H}
    head_model::HM
    aggregation::A
    args_model::JM    
    heuristic::H
end


Flux.@layer ExprModel


@my_cache LRU(maxsize=100_000) function general_cached_inference(ex, ::Type{Expr}, model; all_symbols=new_all_symbols, symbols_to_ind=sym_enc)
    args, fun_name = get_head_and_args(ex)
    args = general_cached_inference(args, model, all_symbols=all_symbols, symbols_to_ind=symbols_to_ind)
    encoding = zeros(Float32, length(all_symbols))
    encoding[symbols_to_index[fun_name]] = 1
    head_model = model.head_model
    h = vcat(head_model.ms.head.m(encoding),
        head_model.ms.args.m(args),
        )
    head_model.m(h)
end


@my_cache LRU(maxsize=100_000) function general_cached_inference(ex, ::Type{Symbol}, model; all_symbols=new_all_symbols, symbols_to_ind=sym_enc)
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


function general_cached_inference(args::Vector, model; all_symbols=new_all_symbols, symbols_to_ind=sym_enc)
    l = length(args)

    tmp = []
    left_inference_type = get_inference_type(args[1])
    left = general_cached_inference(args[1], left_inference_type, model, all_symbols=all_symbols, symbols_to_ind=symbols_to_ind)
    push!(tmp, left)
    if l == 2
        right_inference_type = get_inference_type(args[2])
        right = general_cached_inference(args[2], right_inference_type, model, all_symbols=all_symbols, symbols_to_ind=symbols_to_ind)
        push!(tmp, right)
    end
    
    tmp = hcat(tmp...)
    args_model = model.args_model
    h = vcat(
        args_model.ms.args.m(tmp),
        args_model.ms.position.m(l == 2 ? const_both : const_left),
    )
    positional_encoding = args_model.m(h)
    model.aggregation(positional_encoding, Mill.AlignedBags([1:l]))
end


function (m::ExprModel)(x)
    inference_type = get_inference_type(x)
    ds = general_cached_inference(x, inference_type, m)
    m.heuristic(ds)
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

