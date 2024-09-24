struct SymbolsEncoding{E,M}
    enc::E 
    met::M
end

var_symbols = [Symbol("v$(i-1)") for (i,j) in enumerate(1:number_of_variables)]
new_all_symbols = vcat(all_symbols, var_symbols)

function create_all_symbols_encoding(all_symbols, encoding_length=13 + 1 + 18)
    enc = zeros(Float32, encoding_length, encoding_length)
    enc1 = Flux.onehotbatch(1:encoding_length, 1:encoding_length)
    meta = Dict(i=>ind for (ind, i) in enumerate(all_symbols))
    meta[:Number] = encoding_length
    return SymbolsEncoding(enc1, meta)
end


@inline function update_stats!(stats, depth, δne, δnb)
    if depth > length(stats)
      push!(stats, (0,0))
    end
    @inbounds ne, nb = stats[depth]
    ne += δne
    nb += δnb
    @inbounds stats[depth] = (ne, nb)
    return stats
end


function counter1(ex::Expr)
    depth = 1
    stats = fill((0,0), 10)
    stats = counter!(stats, ex, depth)
    last_element = findfirst(==((0,0)), stats)
    tmp = isnothing(last_element) ? length(stats) : last_element - 1
    return stats[1:tmp]
end


function counter!(stats, ex::Expr, depth)
    if ex.head == :call
        update_stats!(stats, depth, 1, 1)
        for i in 2:length(ex.args)
            stats = counter!(stats, ex.args[i], depth +1)
        end
    elseif ex.head in all_symbols 
        update_stats!(stats, depth, 1, 1)
        for a in ex.args
            stats = counter!(stats, a, depth + 1)
        end
    end
    return stats
end


function counter!(stats, ex::Union{Symbol, Number}, depth)
    update_stats!(stats, depth, 1, 1)
    return stats
end


function unfold_allocation(stats::Vector, encoding_length=13 + 1 + 18)
    depth = 1
    ne,nb = stats[depth]
    am = unfold_allocation(stats, depth + 1, encoding_length)
    # ma = ismissing(am) ? fill(0:-1, nb) : fill(1:2, nb)
    tmp = ProductNode((
        head = ArrayNode(zeros(Float32, encoding_length, ne)),
        args = BagNode(
            am
            , fill(0:-1, nb)),
    ))
    return tmp
end


function unfold_allocation(stats, depth, encoding_length=13 + 1 + 18)
    if depth > length(stats)
        return missing
    end
    ne,nb = stats[depth]
    am = unfold_allocation(stats, depth + 1, encoding_length)
    # # ma = ismissing(am) ? fill(0:-1, nb) : fill(1:2, nb)
    # tmp2 = repeat([1,2], div(nb, 2))
    # tmp1 = isodd(nb) ?  append!([1], tmp2) : tmp2
    pos = ArrayNode(Flux.onehotbatch(ne , 1:2))
    tmp = ProductNode((;
        args = ProductNode((
            head = ArrayNode(zeros(Float32, encoding_length, ne)),
            args = BagNode(
                am
                , fill(0:-1, nb)),
        )),
        position = pos
    ))
    return tmp
end


@inbounds function update_stats1!(stats::Vector, depth, ind, numbers, position, bags)
    if depth > length(stats)
        push!(stats, [])
        push!(numbers, [])
        push!(position, [])
        push!(bags, [])
    end
    # @show stats, numbers
    push!(stats[depth], ind)
    return stats, numbers, position, bags
end


function counter2(ex::Expr, sym_enc::SymbolsEncoding)
    depth = 1
    stats = [[] for _ in 1:10]
    numbers = [[] for _ in 1:10]
    position = [[] for _ in 1:10]
    bags = [UnitRange{Int64}[] for _ in 1:10]
    stats, numbers = counter2!(stats, ex, depth, sym_enc, numbers, position, bags)
    last_element = findfirst(==([]), stats)
    tmp = isnothing(last_element) ? length(stats) : last_element - 1
    return stats[1:tmp], numbers[1:tmp], position[1:tmp], bags[1:tmp]
end


function counter2!(stats, ex::Expr, depth, sym_enc::SymbolsEncoding, numbers, position, bags)
    if ex.head == :call
        update_stats1!(stats, depth, sym_enc.met[ex.args[1]], numbers, position, bags)
        tmp = findlast(x->x!=0:-1,bags[depth])
        if isempty(bags[depth]) || isnothing(tmp)
            push!(bags[depth], 1:length(ex.args)-1)
        else
            # tmp =findlast(x->x!=0:-1,bags[depth])
            # @show bags[depth], tmp
            nstart = bags[depth][tmp].stop + 1
            push!(bags[depth], nstart:nstart + length(ex.args) - 2)
        end
        for i in 2:length(ex.args)
            stats, numbers, position,bags = counter2!(stats, ex.args[i], depth + 1, sym_enc, numbers, position, bags)
            push!(position[depth + 1], i - 1)
        # push!(bags[depth], 1:length(ex.args)-1)

        end
    elseif ex.head in all_symbols 
        update_stats1!(stats, depth, sym_enc.met[ex.head], numbers, position, bags)
        sa = length(ex.args)
        tmp =findlast(x->x!=0:-1,bags[depth])
        if isempty(bags[depth]) || isnothing(tmp)
            push!(bags[depth], 1:sa)
        else
            nstart = bags[depth][tmp].stop + 1
            push!(bags[depth], nstart:nstart + sa - 1)
        end
        for (ind,a) in enumerate(ex.args)
            stats, numbers, position, bags = counter2!(stats, a, depth + 1, sym_enc, numbers, position, bags)
            push!(position[depth + 1], ind)
            # push!(bags[depth], 1:sa)
        end
    end
    return stats, numbers, position, bags
end


my_sigmoid(x, k=0.01, m=0) = 1/(1 + exp(-k*(x-m)))
function counter2!(stats, ex::Union{Symbol, Number}, depth, sym_enc::SymbolsEncoding, numbers, position, bags)
    if isa(ex, Symbol)
        update_stats1!(stats, depth, sym_enc.met[ex], numbers, position, bags)
        push!(bags[depth], 0:-1)
    else
        update_stats1!(stats, depth, sym_enc.met[:Number], numbers, position, bags)
        push!(numbers[depth], my_sigmoid(ex))
        push!(bags[depth], 0:-1)
    end
    return stats, numbers, position, bags
end


function create_bags(n)                                  
    ranges = UnitRange{Int}[]  # Initialize an empty array of UnitRange                                                                
                                                                
        if isodd(n)                                             
        # If n is odd, start with 1:1                           
        push!(ranges, 1:1)                                      
        start = 2  # Start the next range from 2                
    else                                                        
        start = 1  # Start from 1 if n is even                  
    end                                                         
                                                                
    # Create ranges in pairs (i:i+1)                            
    for i in start:2:(n + (isodd(n) ? -1 : 0))                  
        push!(ranges, i:i+1)                                    
    end                                                         
                                                                
    return ranges                                               
end


# Todo : add for a single symbol or a number

function unfold_allocation2(stats::Vector, numbers::Vector, position::Vector, exbags::Vector, encoding_length=13 + 1 + 18)
    depth = 1
    sz1 = length(stats[depth])
    ne,nb = sz1, sz1
    am = unfold_allocation2(stats, depth + 1, numbers, position, exbags, encoding_length)
    # ma = fill(0:-1, nb)
    arr_data = zeros(Float32, encoding_length, ne)
    cols = collect(1:ne)
    arr_data[stats[depth] .+ (cols .- 1) .* encoding_length] .= 1
    tmp = ProductNode((
        head = ArrayNode(arr_data),
        args = BagNode(
            am
            , AlignedBags(exbags[depth])),
    ))
    return tmp
end


function unfold_allocation2(stats::Vector, depth, numbers::Vector, position::Vector, exbags::Vector, encoding_length=13 + 1 + 18)
    if isempty(stats) || depth > length(stats)
        return missing
    end
    sz1 = length(stats[depth])
    ne,nb = sz1, sz1
    am = unfold_allocation2(stats::Vector, depth + 1, numbers::Vector, position::Vector, exbags::Vector, encoding_length)
    # ma = fill(0:-1, nb)
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


function multiple_fast_ex2mill(expression_vector::Vector{Expr}, sym_enc)
    stats = map(x-> counter2(x, sym_enc), expression_vector)
    a = map(x->unfold_allocation2(x[1], x[2], x[3]), stats)
    return a
end


function single_fast_ex2mill(ex, sym_enc)
    st, nm, ps = counter2(ex, sym_enc)
    a = unfold_allocation2(st, nm, ps)
    return a
end


