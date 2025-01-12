@inbounds function update_stats!(stats::Vector, depth::Int, ind::Int, numbers::Vector, position::Vector, bags::Vector)
    if depth > length(stats)
        push!(stats, [])
        push!(numbers, [])
        push!(position, [])
        push!(bags, [])
    end
    # @show stats, numbers
    @inbounds push!(stats[depth], ind)
    return stats, numbers, position, bags
end


function count_expr_stats(ex::Union{Expr,Symbol,Number}, sym_enc::Dict)
    depth = 1
    allocation_depth = 20
    stats = [[] for _ in 1:allocation_depth]
    numbers = [[] for _ in 1:allocation_depth]
    position = [[] for _ in 1:allocation_depth]
    bags = [UnitRange{Int64}[] for _ in 1:allocation_depth]
    stats, numbers = count_expr_stats!(stats, ex, depth, sym_enc, numbers, position, bags)
    last_element = findfirst(==([]), stats)
    tmp = isnothing(last_element) ? length(stats) : last_element - 1
    return stats[1:tmp], numbers[1:tmp], position[1:tmp], bags[1:tmp]
end


function count_expr_stats!(stats, ex::Expr, depth, sym_enc::Dict, numbers, position, bags)
    if ex.head == :call
        update_stats!(stats, depth, sym_enc[ex.args[1]], numbers, position, bags)
        tmp = findlast(x->x!=0:-1,bags[depth])
        if isempty(bags[depth]) || isnothing(tmp)
            push!(bags[depth], 1:length(ex.args)-1)
        else
            nstart = bags[depth][tmp].stop + 1
            push!(bags[depth], nstart:nstart + length(ex.args) - 2)
        end
        for i in 2:length(ex.args)
            stats, numbers, position,bags = count_expr_stats!(stats, ex.args[i], depth + 1, sym_enc, numbers, position, bags)
            push!(position[depth + 1], i - 1)
        end
    elseif ex.head in all_symbols 
        update_stats!(stats, depth, sym_enc[ex.head], numbers, position, bags)
        sa = length(ex.args)
        tmp = findlast(x->x!=0:-1,bags[depth])
        if isempty(bags[depth]) || isnothing(tmp)
            push!(bags[depth], 1:sa)
        else
            nstart = bags[depth][tmp].stop + 1
            push!(bags[depth], nstart:nstart + sa - 1)
        end
        for (ind,a) in enumerate(ex.args)
            stats, numbers, position, bags = count_expr_stats!(stats, a, depth + 1, sym_enc, numbers, position, bags)
            push!(position[depth + 1], ind)
        end
    end
    return stats, numbers, position, bags
end


function count_expr_stats!(stats, ex::Union{Symbol, Number}, depth, sym_enc::Dict, numbers, position, bags)
    if isa(ex, Symbol)
        update_stats!(stats, depth, sym_enc[ex], numbers, position, bags)
        push!(bags[depth], 0:-1)
    else
        update_stats!(stats, depth, sym_enc[:Number], numbers, position, bags)
        push!(numbers[depth], MyModule.my_sigmoid(ex))
        push!(bags[depth], 0:-1)
    end
    return stats, numbers, position, bags
end


# Todo : add for a single symbol or a number

function unfold_allocation(stats::Vector, numbers::Vector, position::Vector, exbags::Vector, encoding_length=13 + 1 + 18)
    depth = 1
    sz1 = length(stats[depth])
    ne,nb = sz1, sz1
    am = unfold_allocation(stats, depth + 1, numbers, position, exbags, encoding_length)
    arr_data = zeros(Float32, encoding_length, ne)
    cols = collect(1:ne)
    arr_data[stats[depth] .+ (cols .- 1) .* encoding_length] .= 1
    # tmp = ProductNode(;
    #     args=ProductNode((
    #         head = ArrayNode(arr_data),
    #         args = BagNode(
    #             am
    #             , AlignedBags(exbags[depth])),
    #     )),
    #     position=ArrayNode(zeros(Float32,2,1))
    # )
    tmp = ProductNode((
        head = ArrayNode(arr_data),
        args = BagNode(
            am
            , AlignedBags(exbags[depth])),
    ))
    return tmp
end


function unfold_allocation(stats::Vector, depth::Int, numbers::Vector, position::Vector, exbags::Vector, encoding_length=13 + 1 + 18)
    if depth > length(stats)
        return missing
    end
    sz1 = length(stats[depth])
    ne,nb = sz1, sz1
    am = unfold_allocation(stats::Vector, depth + 1, numbers::Vector, position::Vector, exbags::Vector, encoding_length)
    # @show depth
    # @show position[depth]
    pos = ArrayNode(Flux.onehotbatch(position[depth], 1:2))
    arr_data = zeros(Float32, encoding_length, ne)
    cols = collect(1:ne)
    arr_data[stats[depth] .+ (cols .- 1) .* encoding_length] .= 1
    if !isempty(numbers[depth])
        arr_data[encoding_length, arr_data[encoding_length,:] .!= 0] .= numbers[depth] 
    end
    tmp = ProductNode((;
        args = ProductNode((
            head = ArrayNode(arr_data),
            args = BagNode(
                am
                , AlignedBags(exbags[depth])),
        )),
        position = pos
    ))
    return tmp
end


function handle_bags(bg, new_bg)
    bg_len = length(bg)
    for (ind, g) in enumerate(new_bg)
        if ind > bg_len 
            push!(bg, g)
        else
            max_bg = maximum(bg[ind])
            if max_bg == 0:-1
                append!(bg[ind], g)
            else
                adjusted_g = [j == 0:-1 ? j : j .+ max_bg.stop for j in g]
                append!(bg[ind], adjusted_g)
            end
        end
    end
    return bg
end

function no_reduce_multiple_fast_ex2mill(expression_vector::Vector, sym_enc)
    if isa(expression_vector[1], ExprWithHash)
        stats = map(x-> count_expr_stats(x.ex, sym_enc), expression_vector)
    else
        stats = map(x-> count_expr_stats(x, sym_enc), expression_vector)
    end
    st, nm, ps, bg = stats[1]
    for (_,i) in enumerate(stats[2:end])
        min_length = min(length(st), length(i[1]))

        # Concatenate corresponding elements of v1 and v2
        result1 = map(vcat, st[1:min_length], i[1][1:min_length])
        result2 = map(vcat, nm[1:min_length], i[2][1:min_length])
        result3 = map(vcat, ps[1:min_length], i[3][1:min_length])
        for (ind1, g) in enumerate(i[4])
            if length(bg) < ind1
                push!(bg, g)
            else
                max_bg = maximum(bg[ind1])
                if max_bg == 0:-1
                    append!(bg[ind1], g)
                else
                    tmp = []
                    for j in g
                        if j == 0:-1
                            push!(tmp, j)
                        else
                            push!(tmp, j .+ max_bg.stop)
                        end
                    end
                    append!(bg[ind1], tmp)
                end
            end
        end
        if length(st) > min_length
            append!(result1, st[min_length+1:end])
            append!(result2, nm[min_length+1:end])
            append!(result3, ps[min_length+1:end])
        elseif length(i[1]) > min_length
            append!(result1, i[1][min_length+1:end])
            append!(result2, i[2][min_length+1:end])
            append!(result3, i[3][min_length+1:end])
        end
        st = result1
        nm = result2
        ps = result3
    end
    a = unfold_allocation(st, nm, ps, bg)
    return a
end


function no_reduce_multiple_fast_ex2mill1(expression_vector::Vector, sym_enc)
    stats = map(x-> count_expr_stats(x, sym_enc), expression_vector)
    st, nm, ps, bg = stats[1]
    for (_,i) in enumerate(stats[2:end])
        min_length = min(length(st), length(i[1]))

        # Concatenate corresponding elements of v1 and v2
        st = [vcat(st[j], i[1][j]) for j in 1:min_length]
        nm = [vcat(nm[j], i[2][j]) for j in 1:min_length]
        ps = [vcat(ps[j], i[3][j]) for j in 1:min_length]
        bg = handle_bags(bg, i[4])
        if length(st) > min_length
            append!(st, st[min_length + 1:end])
            append!(nm, nm[min_length + 1:end])
            append!(ps, ps[min_length + 1:end])
        elseif length(i[1]) > min_length
            append!(st, i[1][min_length + 1:end])
            append!(nm, i[2][min_length + 1:end])
            append!(ps, i[3][min_length + 1:end])
        end
    end
    a = unfold_allocation(st, nm, ps, bg)
    return a
end


function multiple_fast_ex2mill(expression_vector::Vector, sym_enc)
    stats = map(x-> count_expr_stats(x, sym_enc), expression_vector)
    a = map(x->unfold_allocation(x...), stats)
    return a
end


function single_fast_ex2mill(ex, sym_enc=sym_enc)
    st, nm, ps, bg = count_expr_stats(ex, sym_enc)
    a = unfold_allocation(st, nm, ps, bg)
    return a
end
