# Initialize model and example tree
all_symbols = [:+, :-, :/, :*, :<=, :>=, :min, :max, :<, :>, :select, :&&, :||, :(==), :!, :rem, :%]
symbols_to_index = Dict(i=>ind for (ind, i) in enumerate(all_symbols))

function ex2mill(ex::Expr, symbols_to_index, all_symbols, position)
    if ex.head == :call
        fun_name, args =  ex.args[1], ex.args[2:end]
    elseif ex.head in all_symbols # Symbol("&&") || ex.head == Symbol("min")
        fun_name = ex.head
        args = ex.args
    else
        error("unknown head $(ex.head)")
    end
    n = length(symbols_to_index) + 2
    # encoding = Flux.onehotbatch([symbols_to_index[fun_name]], 1:n) * position
    # encoding = Flux.onehotbatch([1, symbols_to_index[fun_name]], 1:n)
    encoding = Flux.onehotbatch([symbols_to_index[fun_name]], 1:n)
    # emb_ind = popfirst!(position)
    # pos_encoding = [parse(Int, i) for i in bitstring(Int16(0))]
    # encoding = vcat(encoding, pos_encoding)
    ds = ProductNode((
        head = ArrayNode(encoding),
        args = args2mill(args, symbols_to_index, all_symbols, position)
    ))
    return(ds)
end

function ex2mill(ex::Symbol, symbols_to_index, all_symbols, position)
    n = length(symbols_to_index) + 2
    # encoding = Flux.onehotbatch([n-1], 1:n) * position
    encoding = Flux.onehotbatch([n-1], 1:n)
    # emb_ind = popfirst!(position)
    # pos_encoding = [parse(Int, i) for i in bitstring(Int16(0))]
    # encoding = vcat(encoding, pos_encoding)
    # encoding = Flux.onehotbatch([n-1], 1:n)
    ds = ProductNode((
        head = ArrayNode(encoding),
        args = BagNode(missing, [0:-1])
        ))
end

function ex2mill(ex::Number, symbols_to_index, all_symbols, position)
    n = length(symbols_to_index) + 2
    # encoding = Flux.onehotbatch([n-2], 1:n) * position
    # @show ex
    encoding = Flux.onehotbatch([n-2], 1:n) # * Float32(ex)
    # emb_ind = popfirst!(position)
    # pos_encoding = [parse(Int, i) for i in bitstring(Int16(ex))]
    # encoding = vcat(encoding, pos_encoding)
    # encoding = Flux.onehotbatch([n-2], 1:n)
    ds = ProductNode((
        head = ArrayNode(encoding),
        args = BagNode(missing, [0:-1])
        ))
end

function args2mill(args::Vector, symbols_to_index, all_symbols, position)
    isempty(args) && return(BagNode(missing, [0:-1]))
    BagNode(
        reduce(catobs, map(a -> ex2mill(a, symbols_to_index, all_symbols, position), args)),
        [1:length(args)]
        )
end


struct ExprModel{HM,A,JM,H}
    head_model::HM
    aggregation::A
    joint_model::JM    
    heuristic::H
end


function get_product_node_matrix!(x::ProductNode, pmatrix) 
    m = x.data.head 
    push!(pmatrix, m)
    get_product_node_matrix!(x.data.args, pmatrix)
end


function get_product_node_matrix!(x::BagNode, pmatrix) 
    get_product_node_matrix!(x.data, pmatrix)
end


function get_product_node_matrix!(x::Missing, pmatrix) 
    return
end


# function Base.==(x::ProductNode, y::ProductNode)
#      
# end
function (m::ExprModel)(ds::ProductNode)
    m.heuristic(embed(m, ds, zeros(1, size(ds.data.head.data)[2])))
end

function embed(m::ExprModel, ds::ProductNode, pos_em::Matrix)
    # sds = size(ds.data.head.data)[2]
    # pos_emb = repeat([1,2], sds ÷ 2)#[collect(1:2) for _ in 1:sds ÷ 2]
    # # pos_emb = hcat([bitstring(Int16(i)) for i in collect(1:sds)]...)
    # # if isempty(pos_emb)
    # #     pos_emb = 
    # # end
    # @show pos_emb
    # @show sds
    # reshaped = isempty(pos_emb) ? zeros(1, sds) : reshape(pos_emb, (1, sds))
    # @show size(reshaped)
    # @show size(ds.data.head.data)
    # stmp = vcat(ds.data.head.data, pos_em)
    # @show size(stmp)
    # tmp1, tmp2 = m.head_model(stmp), embed(m, ds.data.args, pos_em)
    # @show tmp1
    # tmp = vcat(tmp1, tmp2)
    tmp = vcat(m.head_model(ds.data.head.data), embed(m, ds.data.args, pos_em))
    # @show size(tmp)
    # @show ds.data.head.data
    # @show tmp2
    m.joint_model(tmp)
end

function embed(m::ExprModel, ds::BagNode, pos_em::Matrix)
    tmp = embed(m, ds.data, pos_em)
    m.aggregation(tmp, ds.bags)
end

embed(m::ExprModel, ds::Missing, pos_em::Matrix) = missing

function loss(heuristic, big_vector, hp=nothing, hn=nothing)
    o = heuristic(big_vector) 
    p = (o * hp) .* hn

    diff = p - o[1, :] .* hn
    filtered_diff = filter(x-> x != 0, diff)
    return sum(log.(1 .+ exp.(diff)))
end

# :(min(min(v1 * 4, 67) + v0 * 71, 137) <= (1018 + v1 * 4) + v0 * 71)
# :(min(min(v1 * 4, 67) + v0 * 71, 137) <= (v0 * 71 + 1018) + v1 * 4)

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


