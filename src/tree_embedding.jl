# Initialize model and example tree
all_symbols = [:+, :-, :/, :*, :<=, :>=, :min, :max, :<, :>, :select, :&&, :||, :(==), :!, :rem, :%, :(!=)]
symbols_to_index = Dict(i=>ind for (ind, i) in enumerate(all_symbols))

number_of_variables = 13
var_one_enc = Flux.onehotbatch(collect(1:number_of_variables), 1:number_of_variables)
variable_names = Dict(Symbol("v$(i-1)")=>j for (i,j) in enumerate(1:number_of_variables))

function ex2mill(ex::Expr, symbols_to_index, all_symbols, variable_names::Dict)
    if ex.head == :call
        fun_name, args =  ex.args[1], ex.args[2:end]
    elseif ex.head in all_symbols # Symbol("&&") || ex.head == Symbol("min")
        fun_name = ex.head
        args = ex.args
    else
        error("unknown head $(ex.head)")
    end
    n = length(symbols_to_index) + 2
    encoding = zeros(Float32, n+14, 1)

    encoding[symbols_to_index[fun_name]] = 1

    ds = ProductNode((
        head = ArrayNode(encoding),
        args = args2mill(args, symbols_to_index, all_symbols, variable_names)
    ))
    return(ds)
end

function ex2mill(ex::Symbol, symbols_to_index, all_symbols, variable_names)
    n = length(symbols_to_index) + 2
    encoding = zeros(Float32, n+14, 1)
    encoding[n] = 1     # encoding = Flux.onehotbatch([n], 1:n)
    encoding[variable_names[ex] + n] = 1


    ds = ProductNode((
        head = ArrayNode(encoding),
        args = BagNode(missing, [0:-1])
        ))
end


my_sigmoid(x, k = 0.01, m = 0) = 1 / (1 + exp(-k * (x - m)))


function ex2mill(ex::Number, symbols_to_index, all_symbols, variable_names)
    n = length(symbols_to_index) + 2
    encoding = zeros(Float32, n+14, 1)
    encoding[n] = 1     # encoding = Flux.onehotbatch([n], 1:n)
    encoding[n + 1] = my_sigmoid(ex)
    ds = ProductNode((
        head = ArrayNode(encoding),
        args = BagNode(missing, [0:-1])
        ))
end


function args2mill(args::Vector, symbols_to_index, all_symbols, variable_names)
    isempty(args) && return(BagNode(missing, [0:-1]))
    l = length(args)
    BagNode(ProductNode((;
        args = reduce(catobs, [ex2mill(a, symbols_to_index, all_symbols, variable_names) for a in args]),
        position = ArrayNode(Flux.onehotbatch(1:l, 1:2)),
        )),
        [1:l]
        )
end


struct ExprModel{HM,A,JM,H}
    head_model::HM
    aggregation::A
    joint_model::JM    
    heuristic::H
end


function (m::ExprModel)(ds::ProductNode)
    m.heuristic(embed(m, ds))
end


function embed(m::ExprModel, ds::ProductNode) 
    # @show ds.data
    if haskey(ds.data, :position)
        # @show size(m.head_model(ds.data.args.data.head.data))
        tmp = vcat(m.head_model(ds.data.args.data.head.data), ds.data.position.data)
        tmp = vcat(tmp, embed(m, ds.data.args.data.args))
    else
        o = m.head_model(ds.data.head.data)
        tmp = vcat(o, zeros(Float32, 2, size(o)[2]))
        tmp = vcat(tmp, embed(m, ds.data.args))
    end
    # tmp = vcat(m.head_model(ds.data.args.head.data), embed(m, ds.data.args))
    m.joint_model(tmp)
end


function embed(m::ExprModel, ds::BagNode)
    tmp = embed(m, ds.data)
    m.aggregation(tmp, ds.bags)
end


embed(m::ExprModel, ds::Missing) = missing


logistic(x) = log(1 + exp(x))
hinge(x) = max(0, 1 - x)
loss01(x) = x > 0


function loss(heuristic, big_vector, hp=nothing, hn=nothing, surrogate::Function = logistic)
    o = heuristic(big_vector) 
    p = (o * hp) .* hn

    diff = p - o[1, :] .* hn
    filtered_diff = filter(!=(0), diff)
    # return sum(log.(1 .+ exp.(filtered_diff)))
    return mean(softmax(filtered_diff))
    # return sum(surrogate.(filtered_diff))
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
