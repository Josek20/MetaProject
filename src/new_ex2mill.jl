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


function counter(ex::Expr)
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

function unfold_allocation(stats, first::Bool, encoding_length=14 + 2 + 18)
    ne,nb = stats[1]
    am = unfold_allocation(stats[2:end])
    ma = ismissing(am) ? fill(0:-1, nb) : fill(1:2, nb)
    tmp = ProductNode((
        head = ArrayNode(zeros(Float32, encoding_length, ne)),
        args = BagNode(
            am
            , ma),
    ))
    return tmp
end


function unfold_allocation(stats, encoding_length=14 + 2 + 18)
    if isempty(stats)
        return missing
    end
    ne,nb = stats[1]
    am = unfold_allocation(stats[2:end])
    ma = ismissing(am) ? fill(0:-1, nb) : fill(1:2, nb)
    tmp = ProductNode((;
        args = ProductNode((
            head = ArrayNode(zeros(Float32, encoding_length, ne)),
            args = BagNode(
                am
                , ma),
        )),
        position = ArrayNode(Flux.onehotbatch(fill(1:2, nb), 1:2)),
    ))
    return tmp
end