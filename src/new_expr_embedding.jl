function expr_to_prefix!(ex::Expr, all_symbols, prefix::Vector=[])
    h = ex.head
    if h == :call
        push!(prefix, ex.args[1])
        for i in ex.args[2:end]
            expr_to_prefix!(i, all_symbols, prefix)
        end
    elseif h in all_symbols
        push!(prefix, h)
        for i in ex.args
            expr_to_prefix!(i, all_symbols, prefix)
        end
    end
    return prefix
end


function expr_to_prefix!(ex::Union{Symbol, Number}, all_symbols, prefix::Vector)
    push!(prefix, ex)
end


function embed_prefix_expr(prefix_ex::Vector, symb2ind::Dict{Symbol, Int})
    n = length(symb2ind)
    p = length(prefix_ex)
    encoding = zeros(Float32, n, p)
    for (ind,i) in enumerate(prefix_ex)
        if isa(i, Number)
            encoding[symb2ind[:Number], ind] = MyModule.my_sigmoid(i)
        else
            encoding[symb2ind[i], ind] = 1
        end
    end
    return encoding
end


function simple_expression_encoding(ex::Expr, all_symbols, symb2ind)
    expr_pref = expr_to_prefix!(ex, all_symbols)
    return embed_prefix_expr(expr_pref, symb2ind)
end


function ()
    
end