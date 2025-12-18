using CSV
using Graphs
using DataFrames
using Plots
using D3Trees
using Optimisers
using Base.Threads
using Random
using Statistics
using MyModule
using MyModule.Flux
using MyModule.Mill
using MyModule: all_expand, exp_size, Node, NodeID, intern!, expr, DeduplicatingNode, AbstractModel
using Serialization
using TimerOutputs
using Base.Iterators
using TimerOutputs
const TO = TimerOutput()
reset_timer!(TO)

Random.seed!(42)
include("rl_pipeline.jl")
include("my_env.jl")
include("sampler.jl")


include("learner.jl")

include("plot_soltree_radial.jl")

Base.only(t::Tuple{Float32, Float32}) = t
Base.only(t::Matrix{Float32}) = size(t)[1] > 1 ? Tuple(vec(t)) : first(t)
Base.vec(t::Tuple{Float32, Float32}) = t
Base.isless(a::Tuple{Float32, Float32}, b::Tuple{Float32, Float32}) = a[1] < b[1] || (a[1] == b[1] && a[2] < b[2])
Base.round(t::Tuple{Float32, Float32}, m::RoundingMode{:Nearest}; digits::Int64) = t
function get_data(;path="train.json")
    train_data_path = "./data/neural_rewrter/$(path)"
    train_data = load_data(train_data_path)[1:1_000]
    train_data = filter(x->!occursin("select", x[1]), train_data)
    train_data = preprosses_data_to_expressions(train_data)
    sorted_data = sort(train_data, by=x->MyModule.exp_size(x))
    data = sorted_data
    return Vector{Expr}(data)
end

data = get_data()

hidden_size=64
input_size = 64

function ffnn(idim, hidden_size, layers)
    layers == 1 && return Dense(idim, hidden_size, Flux.gelu)
    layers == 2 && return Chain(Dense(idim, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu))
end

head_model = ProductModel(
    (;head = ffnn(length(new_all_symbols), hidden_size, 1),
    args = ffnn(hidden_size, hidden_size, 1),  
        ),
    ffnn(2*hidden_size, hidden_size, 1)
    )

args_model = ProductModel(
    (;args = ffnn(hidden_size, hidden_size, 1),  
    position = Dense(2,hidden_size),  
        ),
    ffnn(2*hidden_size, hidden_size, 1)
    )

sampler = Tree2TreeSampler(max_steps=100, max_depth=100, epsilon=0.0, eps_decay=0.80, is_directed=true, n_best=-1, batch=128, gamma=1)

model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
    );
target_model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
    );
target_model = deepcopy(model)
learner = DummyLerner(Flux.mse, model, max_iter=10, lr=0.001)
env = MyTreeEnv(data[1], model)
pipeline = SimpleRLPipeline(env, model, sampler, learner, target_model)


########################################################
#        Experiments
########################################################
function value_with_nn()
    MyModule.reset_all_function_caches()
    loss_over_time = []
    for (ind,i) in enumerate(data[1:500])
        # i = data[1]
        soltree, best_node, root, all_trees_history = MyModule.initialize_tree_search_tree(intern!(i), model; max_expansions=100, max_depth=sampler.max_depth, epsilon=0.5, gamma=sampler.gamma)
        # best_node
        # break
        r(t::Vector{UInt64}, tₙ::Vector{UInt64}) = minimum(map(x->exp_size(soltree[x].ex), t)) - minimum(map(x->exp_size(soltree[x].ex), tₙ))

        update_time = @elapsed history_res = map(1:10) do k   
            res = map(enumerate(all_trees_history[1: end - 1])) do (ind, t)
                target_v = r(t, all_trees_history[ind + 1]) + sampler.gamma * maximum(map(sₙ -> only(pipeline.target_model(soltree[sₙ].ex)), all_trees_history[ind + 1]))
                # tmp = [exp_size(soltree[i].ex) for i in t]
                # best_s_id = argmin(tmp)
                tmp = [only(pipeline.target_model(soltree[i].ex)) for i in t]
                best_s_id = argmax(tmp)
                return (;rewards=target_v, input_values=soltree[t[best_s_id]].ex)
            end
            tmp1 = [i.input_values for i in res]
            tmp2 = [i.rewards for i in res]
            input_values = get_input_values(tmp1)
            target = get_target(tmp2, pipeline.sampler)

            learning_time = 0
            loss = 0
            learning_time += @elapsed loss += compute_gradient!(target, input_values, pipeline.model, pipeline.learner)
            if mod(k, 5) == 0
                pipeline.target_model = deepcopy(pipeline.model)
            end
            push!(loss_over_time, loss)
            # loss_over_time += loss
            @show loss
            clean_cache(pipeline.model)
            _, smallest_node, _, _ = MyModule.initialize_tree_search_tree(intern!(i), pipeline.model; max_expansions=10, max_depth=sampler.max_depth, epsilon=0, gamma=sampler.gamma)
            println("Best solution :$(exp_size(smallest_node.ex)) | $(exp_size(best_node.ex))")
            s1 = Dict()
            for (k,v) in soltree
                s1[v.ex] = only(pipeline.model(v.ex))
            end
            return(s1)
        end
        history = Dict() 
    
        expr_values_vectors = map(values(soltree)) do i 
            res = map(x->x[i.ex], history_res)
            history[i.node_id] = res
            return(;ex=i.ex, values=res, nid=i.node_id)
        end
        Plots.plot(map(x->x.values,expr_values_vectors), legend=false)
        savefig("stats/tree_state_lr001/expr_$(ind).png")
    end
    # end

    # plot(loss_over_time, legend=false)
    # # plot()
    # plot(map(x->x.values,expr_values_vectors), legend=false)
    # savefig("stats/tree_state/expr_$(ind).png")
    # fig, _, _ = plot_radial_interactive_tree(soltree, history)
    # fig
end
# value_with_nn()

new_soltree = deserialize("test_tree.bin")
new_soltree1 = Dict(k=>MyModule.Node(MyModule.intern!(v.ex), v.rule_index, v.children, v.parent, v.depth, v.node_id) for (k,v) in new_soltree)
new_soltree = deserialize("test_tree1.bin")
new_soltree = Dict(k=>MyModule.Node(MyModule.intern!(v.ex), v.rule_index, v.children, v.parent, v.depth, v.node_id) for (k,v) in new_soltree)

all_nodes = collect(values(new_soltree))
index_root = findfirst(x->x.depth==0, all_nodes)
root = all_nodes[index_root]

all_nodes1 = collect(values(new_soltree1))
index_root1 = findfirst(x->x.depth==0, all_nodes1)
root1 = all_nodes1[index_root1]

# soltree, smallest_node, root, soltree1 = MyModule.initialize_tree_search_epsilon(intern!(data[end]), model; max_expansions=1000, max_depth=sampler.max_depth, epsilon=0.5, gamma=sampler.gamma)
# # smallest_node.ex
# new_soltree = soltree
function tree_max()
    function enumetrate_all_subtree(node::Node, soltree::Dict)
        result = Set[]
        push!(result, Set(node.node_id))
        isempty(node.children) && return result

        children_subrees = map(node.children) do child
            enumetrate_all_subtree(soltree[child], soltree)
        end

        for combination in collect(Iterators.product(children_subrees...))
            new_set = Set(node.node_id)
            for s in combination
                union!(new_set, s)
            end
            push!(result, new_set)
        end
        return result
    end


    # soltree, smallest_node, root, soltree1 = MyModule.initialize_policy_tree_search(intern!(data[end]), model, max_expansions=1000, max_depth=100, gamma=0.9)
    # @show MyModule.TO
    # break
    tmp = enumetrate_all_subtree(root, new_soltree)
    println("get all subtrees")
    γ = 0.99
    α = 0.5
    V_iter = Dict(i=>0f0 for i in tmp)
    V_td = Dict(i=>0f0 for i in tmp)
    V_app = Dict(i=>0f0 for i in tmp)
    V_iter1 = Dict(i=>0f0 for i in keys(new_soltree)) 
    V_td1 = Dict(i=>0f0 for i in keys(new_soltree)) 
    # V_ = Dict(i=>0f0 for i in keys(new_soltree)) 
    r(t::Vector{UInt64}, tₙ::Vector{UInt64}, soltree::Dict) = minimum(map(x->exp_size(soltree[x].ex), t)) - minimum(map(x->exp_size(soltree[x].ex), tₙ))
    r(t::Set{UInt64}, tₙ::Set{UInt64}, soltree::Dict) = r(collect(t), collect(tₙ), soltree)

    # for (ind,i) in enumerate(tmp[1:end-1])
    plot_td = [collect(values(V_td))]
    plot_iter = [collect(values(V_iter))]
    plot_td1 = [collect(values(V_td1))]
    plot_iter1 = [collect(values(V_iter1))]
    plot_app = [map(x->only(model(new_soltree[x].ex)), collect(keys(new_soltree)))]
    function get_tr(current_tree, possible_next_trees, new_soltree, values)
        map(possible_next_trees) do tₓ
            # check if tₓ is leaf
            # @show tₓ
            is_leaf = true
            for j in tₓ
                # !isempty(intersect(next_child, i)) && continue
                j in current_tree && continue
                next_child = Set(new_soltree[j].children)
                if !isempty(next_child)
                    # @show j
                    is_leaf = false
                end
            end
            # subtree 
            # vl = is_leaf ? 0 : maximum(x->values[x], tₓ)
            # fring
            # if is_leaf
            #     vl = 0
            # else
            mask = .!in.(tₓ, Ref(current_tree))
            @show tₓ, collect(tₓ), collect(tₓ)[mask], mask
            vl = maximum(x->values[x], collect(tₓ)[mask])
            # end
            r(current_tree, tₓ, new_soltree) + γ * vl
        end
    end
    for _ in 1:100
        rew = []
        input_nodes = []
        leafs_set = Set()
        for (ind,i) in enumerate(tmp)
            nodes = collect(i)
            possible_next_trees = Set[]
            for j in i
                if isempty(new_soltree[j].children)
                    push!(leafs_set, j)
                    continue
                end
                
                next_child = Set(new_soltree[j].children)
                !isempty(intersect(next_child, i)) && continue
                # @show j
                push!(possible_next_trees, union(next_child, i))
            end
            isempty(possible_next_trees) && continue
            # Value iteration tabular
            V_iter[i] = maximum(r(i, tₓ, new_soltree) + γ * get!(V_iter, tₓ, 0f0) for tₓ in possible_next_trees)
            # TD(0) tabular
            V_td[i] += α * (maximum(r(i, tₓ, new_soltree) + γ * get!(V_td, tₓ, 0f0) for tₓ in possible_next_trees) - V_td[i])
            # @show i
            # v_iter 1
            # tr1 = maximum(r(i, tₓ, new_soltree) + γ * (haskey(V_iter, tₓ) ? maximum(x->V_iter1[x], tₓ) ) for tₓ in possible_next_trees)
            tr1 = get_tr(i, possible_next_trees, new_soltree, V_iter1) 
            # tr1 = maximum(r(i, tₓ, new_soltree) + γ * maximum(x->V_iter1[x], tₓ) for tₓ in possible_next_trees)
            max_id1 = argmax(x->V_iter1[x], i)
            # max_id1 = argmax(x->exp_size(new_soltree[x].ex), i)
            V_iter1[max_id1] = maximum(tr1)
            @show values(V_iter1)
            @show max_id1, maximum(tr1)
            # v_td 1
            tr1 = maximum(r(i, tₓ, new_soltree) + γ * maximum(x->V_td1[x], tₓ) for tₓ in possible_next_trees)
            # tr1 = get_tr(i, possible_next_trees, new_soltree, V_td1)
            # max_id1 = argmax(x->V_td1[x], i)
            # @show max_id1, maximum(tr1)
            V_td1[max_id1] += α * (maximum(tr1) - V_td1[max_id1])
            # approximate
            tr = maximum(r(i, tₓ, new_soltree) + γ * maximum(x->only(model(new_soltree[x].ex)), tₓ) for tₓ in possible_next_trees)
            push!(rew, tr)
            max_id = argmax(map(s->only(model(new_soltree[s].ex)), nodes))
            push!(input_nodes, MyModule.expr(MyModule.nc,new_soltree[nodes[max_id]].ex))
        end
        push!(plot_iter, collect(values(V_iter)))
        push!(plot_td, collect(values(V_td)))
        push!(plot_iter1, collect(values(V_iter1)))
        push!(plot_td1, collect(values(V_td1)))
        # clean_cache(model)
        # ds =  MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_nodes, sym_enc))
        # loss = compute_gradient!(rew, ds, model, learner)
        # push!(plot_app, map(x->only(model(new_soltree[x].ex)), collect(keys(new_soltree))))
        # @show values(V_iter)  
        # @show  V_iter[Set(0x7356776a85238df8)]
        # @show  V_td[Set(0x7356776a85238df8)]
        # @show V_iter1[0x7356776a85238df8]
        # @show V_td1[0x7356776a85238df8]
        # @show loss
        # @show only(model(new_soltree[0x7356776a85238df8].ex))
    end

    # Plots.plot(hcat(plot_td...)', legend=false)
    # Plots.plot(hcat(plot_iter...)', legend=false)
    # Plots.plot(hcat(plot_td1...)', legend=false)
    Plots.plot(hcat(plot_iter1...)', legend=false)
    # Plots.plot(hcat(plot_app...)', legend=false)

    # history = Dict(i=>plot_td[ind] for (ind,i) in enumerate(keys(new_soltree))) 
    # fig, _, _ = plot_radial_interactive_tree(new_soltree, history)
    # fig
end

############################################
#               Q learning
# Q(T, a) = r(T, T') + γ * max_a'(Q(T', a'))
############################################
function tabular_q_learning()
    function get_Q!(node::Node, soltree::Dict, V_qiter::Dict, V_qtd::Dict; α=0.5, γ=0.99)
        next_children = soltree[node.node_id].children
        isempty(next_children) && return
        if node.depth == 0
            get_full_tree = [node.node_id]
        else
            current_node_id = node.node_id
            # get_full_tree = map(1:node.depth) do _
            #     tmp = current_node_id
            #     current_node_id = soltree[current_node_id].parent
            #     return soltree[current_node_id].children + current_node_id
            # end
            get_full_tree = []
            for d in node.depth:-1:1
                current_node_id = soltree[current_node_id].parent
                append!(get_full_tree, soltree[current_node_id].children)
            end
            push!(get_full_tree, current_node_id)
        end
        @show get_full_tree
        rew = minimum(x->exp_size(soltree[x].ex), get_full_tree) - minimum(x->exp_size(soltree[x].ex), next_children)
        new_target = rew + γ * maximum(x->V_qiter[x], next_children)
        V_qiter[node.node_id] = new_target
        V_qtd[node.node_id] += α * (new_target - V_qtd[node.node_id])
        for child in next_children
            get_Q!(soltree[child], soltree, V_qiter, V_qtd)
        end
        return
    end

    V_qiter = Dict(i=>0f0 for i in keys(new_soltree))
    V_qtd = Dict(i=>0f0 for i in keys(new_soltree))
    plot_qiter_values = [collect(values(V_qiter))]
    plot_qtd_values = [collect(values(V_qtd))]
    states_history = Dict(i=>[] for i in keys(new_soltree))
    for ep in 1:500
        get_Q!(root, new_soltree, V_qiter, V_qtd)
        # @show V_q
        push!(plot_qiter_values, collect(values(V_qiter)))
        push!(plot_qtd_values, collect(values(V_qtd)))
        for (k,v) in V_qtd
            push!(states_history[k], v)
        end
    end
    Plots.plot(hcat(plot_qiter_values...)', legend=false)
    Plots.plot(hcat(plot_qtd_values...)', legend=false)
    fig, _, _ = plot_radial_interactive_tree(new_soltree, V_qiter)
    fig
end



# function tabular_qtarget_learning()
function get_Q_targets!(node::Node, soltree::Dict, model::ExprModel, targets::Dict; α=0.5, γ=0.99)
    next_children = soltree[node.node_id].children
    if isempty(next_children)
        targets[node.node_id] = 0f0
        return
    end
    if node.depth == 0
        get_full_tree = [node.node_id]
    else
        @timeit TO "get full tree" begin
        
            current_node_id = node.node_id
            get_full_tree = []
            for d in node.depth:-1:1
                current_node_id = soltree[current_node_id].parent
                append!(get_full_tree, soltree[current_node_id].children)
            end
            push!(get_full_tree, current_node_id)
        end
    end
    # @show get_full_tree
    @timeit TO "get reward" rew = minimum(x->exp_size(soltree[x].ex), get_full_tree) - minimum(x->exp_size(soltree[x].ex), next_children)
    @timeit TO "get new target value" new_target = rew + γ * maximum(x->only(model(soltree[x].ex)), next_children)
    targets[node.node_id] = new_target
    for child in next_children
        get_Q_targets!(soltree[child], soltree, model, targets)
    end
    return
end

# plot_app_values = [zeros(Float32, length(new_soltree))]
# plot_app_values1 = [zeros(Float32, length(new_soltree1))]
# targets_history = Dict(i=>[] for i in collect(keys(new_soltree)))
# targets_history1 = Dict(i=>[] for i in collect(keys(new_soltree1)))
# loss_history = []
# for ep in 1:100
#     targets = Dict()
#     get_Q_targets!(root, new_soltree, model, targets)
#     targets1 = Dict()
#     get_Q_targets!(root1, new_soltree1, model, targets1)
#     # @show targets
#     push!(plot_app_values, collect(values(targets)))
#     push!(plot_app_values1, collect(values(targets1)))
#     for (k,v) in targets_history
#         push!(targets_history[k], only(model(new_soltree[k].ex)))
#     end
#     for (k,v) in targets_history1
#         push!(targets_history1[k], only(model(new_soltree1[k].ex)))
#     end
#     clean_cache(target_model)
#     if mod(ep, 5) == 0
#         target_model = deepcopy(model)
#     end
#     # clean_cache(model)
#     input_nodes = map(x->MyModule.expr(MyModule.nc, new_soltree[x].ex), collect(keys(targets)))
#     input_nodes1 = map(x->MyModule.expr(MyModule.nc, new_soltree1[x].ex), collect(keys(targets1)))
#     ds =  MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_nodes, sym_enc))
#     ds1 =  MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_nodes1, sym_enc))
#     loss = map(1:10) do i
#         loss = 0
#         for (tr, ds) in [(targets, ds), (targets1, ds1)]
#             loss += compute_gradient!(collect(values(tr)), ds, model, learner)
#         end
#         loss
#     end
#     push!(loss_history, sum(loss))
#     # @show loss
#     # @show V_q 
# end
# Plots.plot(hcat(plot_app_values[1:end]...)', legend=false)
# Plots.plot(hcat(plot_app_values1[1:end]...)', legend=false)
# Plots.plot(loss_history, legend=false)
# history = Dict(i=>only(model(new_soltree1[i].ex)) for (ind,i) in enumerate(keys(new_soltree1))) 
# fig, _, _ = plot_radial_interactive_tree(new_soltree1, Dict())
# fig

function tabular_reinforce()
    function get_solutions_traj_actions!(node::Node, soltree, targets)
        if node.depth == 0
            return
        end
        current_node_id = node.parent
        get_full_tree = UInt64[]
        possible_actions = UInt64[]
        for d in soltree[node.parent].depth:-1:1
            tmp = current_node_id
            current_node_id = soltree[current_node_id].parent
            tmp = filter(x->tmp != x, soltree[current_node_id].children)
            append!(possible_actions, tmp)
            append!(get_full_tree, soltree[current_node_id].children)
        end
        push!(get_full_tree, current_node_id)
        # @show length(possible_actions)
        # @show length(get_full_tree)
        if isa(targets, Dict)
            targets[get_full_tree] = Dict(i=>0 for i in possible_actions)
        else
            push!(targets, (node.parent, possible_actions, get_full_tree))
        end
        get_solutions_traj_actions!(soltree[node.parent], soltree, targets)
    end
    targets = []
    soltree = new_soltree
    smallest_node = MyModule.extract_smallest_node(soltree)
    get_solutions_traj_actions!(smallest_node, soltree, targets)
    # targets
    # length.(collect(values(targets)))
    # length.(collect(keys(targets)))
    targets_table = Dict(t=>zeros(Float32, length(a)+1) for (pa, a, t) in targets)
    # targets_table = Dict(t=>Dict(k=>0 for k in vcat(pa, a)) for (pa, a, t) in targets)
    γ = 0.99
    α = 0.5
    for ep in 1:10
        Gₜ = 0
        for j in collect(values(targets_table))
            @show j
        end
        for (ind, (pos_action, possible_actions, tree)) in enumerate(targets)
            if ind == 1
                rew = minimum(x->exp_size(soltree[x].ex), tree) - exp_size(smallest_node.ex)
            else
                rew = minimum(x->exp_size(soltree[x].ex), tree) - minimum(x->exp_size(soltree[x].ex), targets[ind - 1][3])
            end
            Gₜ = rew + γ * Gₜ
            @show Gₜ
            # update solution action
            grad = Flux.softmax(collect(targets_table[tree]))
            targets_table[tree][1] += α * Gₜ * (1 - grad[1])
            # update non solution actions
            for i in 2:length(possible_actions) + 1
                targets_table[tree][i] += α * Gₜ * ( - grad[i])
            end
        end
    end


    ###########################################
    # state is node
    ###########################################
    # targets_table = Dict(pa=>zeros(Float32, length(a)+1) for (pa, a, t) in targets)
    targets_table = Dict(pa=>Dict(k=>0f0 for k in vcat(pa, a)) for (pa, a, t) in targets)
    γ = 0.99
    α = 0.5
    for ep in 1:10
        Gₜ = 0
        for j in collect(values(targets_table))
            @show values(j)
        end
        for (ind, (pos_action, possible_actions, tree)) in enumerate(targets)
            if ind == 1
                rew = minimum(x->exp_size(soltree[x].ex), tree) - exp_size(smallest_node.ex)
            else
                rew = minimum(x->exp_size(soltree[x].ex), tree) - minimum(x->exp_size(soltree[x].ex), targets[ind - 1][3])
            end
            Gₜ = rew + γ * Gₜ
            @show Gₜ
            # update solution action
            keys_ = collect(keys(targets_table[pos_action]))
            softmax_vals = Flux.softmax(collect(values(targets_table[pos_action])))
            targets_table[pos_action] = Dict(keys_[i] => softmax_vals[i] for i in eachindex(keys_))
            grad = (1 - targets_table[pos_action][pos_action])
            targets_table[pos_action][pos_action] += α * Gₜ * grad
            # update non solution actions
            for i in possible_actions
                targets_table[pos_action][i] += α * Gₜ * ( - targets_table[pos_action][i])
            end
        end
    end
    
    ###########################################
    # state is node
    ###########################################
    targets_table = Dict(pa=>[0f0, Dict(k=>0f0 for k in a)] for (pa, a, t) in targets)
    γ = 0.99
    α = 0.5
    for ep in 1:1000
        Gₜ = 0
        for (k,j) in collect(values(targets_table))
            @show k, values(j)
        end
        for (ind, (pos_action, possible_actions, tree)) in enumerate(targets)
            if ind == 1
                rew = minimum(x->exp_size(soltree[x].ex), tree) - exp_size(smallest_node.ex)
            else
                rew = minimum(x->exp_size(soltree[x].ex), tree) - minimum(x->exp_size(soltree[x].ex), targets[ind - 1][3])
            end
            Gₜ = rew + γ * Gₜ
            @show Gₜ
            # update solution action
            keys_ = collect(keys(targets_table[pos_action][2]))
            tmp1 = collect(values(targets_table[pos_action][2]))
            if isempty(tmp1)
                softmax_vals = Flux.softmax([targets_table[pos_action][1]])
                grads1 = softmax_vals[1]
                # targets_table[pos_action][1] += α * Gₜ
                # targets_table[pos_action][2] = Dict(keys_[i] => softmax_vals[i+1] for i in eachindex(keys_))
            else
                softmax_vals = Flux.softmax(vcat(targets_table[pos_action][1], tmp1))
                grads1 = softmax_vals[1]
                grads2 = Dict(keys_[i] => softmax_vals[i+1] for i in eachindex(keys_))
            end
            grad = (1 - grads1)
            targets_table[pos_action][1] += α * Gₜ * grad
            # update non solution actions
            for i in possible_actions
                targets_table[pos_action][2][i] += α * Gₜ * ( - grads2[i])
            end
        end
    end

    # targets_table = Dict(pa=>[0f0, Dict(k=>0f0 for k in a)] for (pa, a, t) in targets)
    targets_table = Dict(i=>0f0 for i in keys(soltree))
    γ = 0.99
    α = 0.5
    for ep in 1:10
        Gₜ = 0
        # for (k,j) in collect(values(targets_table))
        #     @show k, values(j)
        # end
        @show collect(values(targets_table))
        for (ind, (pos_action, possible_actions, tree)) in enumerate(targets)
            if ind == 1
                rew = minimum(x->exp_size(soltree[x].ex), tree) - exp_size(smallest_node.ex)
            else
                rew = minimum(x->exp_size(soltree[x].ex), tree) - minimum(x->exp_size(soltree[x].ex), targets[ind - 1][3])
            end
            Gₜ = rew + γ * Gₜ
            @show Gₜ
            # update solution action
            if isempty(possible_actions)
                softmax_vals = Flux.softmax([targets_table[pos_action]])
                grads1 = softmax_vals[1]
            else
                softmax_vals = Flux.softmax(vcat(targets_table[pos_action], [targets_table[x] for x in possible_actions]))
                grads1 = softmax_vals[1]
            end
            grad = (1 - grads1)
            targets_table[pos_action] += α * Gₜ * grad
            # update non solution actions
            for (ind,i) in enumerate(possible_actions)
                targets_table[i] += α * Gₜ * ( - softmax_vals[ind+1])
            end
        end
    end

    #######################################
    #  using NN to app
    #######################################
    γ = 0.99
    α = 0.5
    for ep in 1:50
        Gₜ = 0
        inputs_vec = []
        rews = []
        for (ind, (pos_action, possible_actions, tree)) in enumerate(targets)
            if ind == 1
                rew = minimum(x->exp_size(soltree[x].ex), tree) - exp_size(smallest_node.ex)
            else
                rew = minimum(x->exp_size(soltree[x].ex), tree) - minimum(x->exp_size(soltree[x].ex), targets[ind - 1][3])
            end
            Gₜ = rew + γ * Gₜ
            @show Gₜ
            input_values = [expr(MyModule.nc, soltree[i].ex) for i in vcat(pos_action, possible_actions)]
            input_values = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_values, sym_enc))
            push!(inputs_vec, input_values)
            push!(rews, Gₜ)
        end
        sa, grad = Flux.Zygote.withgradient(model) do oq
            loss = 0
            for (ind, (inp, Gₜ)) in enumerate(zip(inputs_vec, rews))
                outs = vec(MyModule.heuristic(oq, inp))
                prob1 = Flux.softmax(outs)[1]
                logp = log(prob1 + 1e-8) 
                loss -= logp * Gₜ 
            end
            @show loss
            return loss
        end
        @show sa
        Optimisers.update!(learner.params, model, grad[1])
    end
    clean_cache(model)
    history = Dict(i=>only(model(soltree[i].ex)) for (ind,i) in enumerate(keys(soltree))) 
    fig, _, _ = plot_radial_interactive_tree(soltree, history)
    fig
end
function get_solutions_traj_actions!(node::Node, soltree, targets)
    if node.depth == 0
        return
    end
    current_node_id = node.parent
    get_full_tree = UInt64[]
    possible_actions = UInt64[]
    for d in soltree[node.parent].depth:-1:1
        tmp = current_node_id
        current_node_id = soltree[current_node_id].parent
        tmp = filter(x->tmp != x, soltree[current_node_id].children)
        append!(possible_actions, tmp)
        append!(get_full_tree, soltree[current_node_id].children)
    end
    push!(get_full_tree, current_node_id)
    # @show length(possible_actions)
    # @show length(get_full_tree)
    if isa(targets, Dict)
        targets[get_full_tree] = Dict(i=>0 for i in possible_actions)
    else
        push!(targets, (node.parent, possible_actions, get_full_tree))
    end
    get_solutions_traj_actions!(soltree[node.parent], soltree, targets)
end
targets = []
soltree = new_soltree
smallest_node = MyModule.extract_smallest_node(soltree)
get_solutions_traj_actions!(smallest_node, soltree, targets)
policy = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
    );
policy_old = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
    );
policy_old = deepcopy(policy)
value_model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Chain(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1))
    );

learner = PolicyLerner(Flux.mse, policy, max_iter=10, lr=0.001)
vlearner = DummyLerner(Flux.mse, value_model, max_iter=10, lr=0.001)
γ = 0.99
ϵ = 0.2
loss1 = []
loss2 = []
for ep in 1:100
    Gₜ = 0
    inputs_vec = []
    rews = []
    for (ind, (pos_action, possible_actions, tree)) in enumerate(targets)
        if ind == 1
            rew = minimum(x->exp_size(soltree[x].ex), tree) - exp_size(smallest_node.ex)
        else
            rew = minimum(x->exp_size(soltree[x].ex), tree) - minimum(x->exp_size(soltree[x].ex), targets[ind - 1][3])
        end
        Gₜ = rew + γ * Gₜ
        @show Gₜ
        Aₜ = Gₜ - only(value_model(soltree[pos_action].ex))
        @show Aₜ
        input_values = [expr(MyModule.nc, soltree[i].ex) for i in vcat(pos_action, possible_actions)]
        input_values = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_values, sym_enc))
        push!(inputs_vec, input_values)
        # push!(rews, Gₜ)
        push!(rews, Aₜ)
    end
    clean_cache(value_model)
    input_values = [expr(MyModule.nc, soltree[i].ex) for (i, _, _) in targets]
    v_inputs = MyModule.deduplicate(MyModule.no_reduce_multiple_fast_ex2mill(input_values, sym_enc))
    sa, grad = Flux.Zygote.withgradient(value_model) do vm
        outs = vec(MyModule.heuristic(vm, v_inputs))
        Flux.mse(outs, rews)
    end
    push!(loss1, sa)
    Optimisers.update!(vlearner.params, value_model, grad[1])
    sa, grad = Flux.Zygote.withgradient(policy) do oq
        # loss = 0
        # for (ind, (inp, Aₜ)) in enumerate(zip(inputs_vec, rews))
        loss = map(enumerate(zip(inputs_vec, rews))) do (ind, (inp, Aₜ))
            outs = vec(MyModule.heuristic(oq, inp))
            outs_old = vec(MyModule.heuristic(policy_old, inp))
            prob_old = Flux.softmax(outs_old)[1]
            prob = Flux.softmax(outs)[1]
            ratio = exp(log(prob) - log(prob_old))
            clipped_ratio = clamp(ratio, 1 - ϵ, 1 + ϵ)
            policy_loss = min(ratio * Aₜ, clipped_ratio * Aₜ)
            -policy_loss
        end
        return mean(loss)
    end
    @show sa
    push!(loss2, sa)
    if mod(ep, 2) == 0
        policy_old = deepcopy(policy)
    end
    Optimisers.update!(learner.params, policy, grad[1])
end

clean_cache(policy)
@show Flux.softmax(vec(MyModule.heuristic(policy, inputs_vec[1])))
plot([loss1, loss2], label=["value loss" "policy loss"])
# history = Dict(i=>only(policy(soltree[i].ex)) for (ind,i) in enumerate(keys(soltree))) 
# fig, _, _ = plot_radial_interactive_tree(soltree, history)
# fig

# for i in values(soltree)
#     @show i.ex, only(model(i.ex))
# end
# soltree = new_soltree1
# smallest_node = MyModule.extract_smallest_node(soltree)
# current_node_id = smallest_node.parent
# get_full_tree = []
# possible_actions = []
# for d in soltree[smallest_node.parent].depth:-1:1
#     tmp = current_node_id
#     current_node_id = soltree[current_node_id].parent
#     tmp = filter(x->tmp != x, soltree[current_node_id].children)
#     append!(possible_actions, tmp)
#     append!(get_full_tree, soltree[current_node_id].children)
# end
# push!(get_full_tree, current_node_id)
# push!(possible_actions, smallest_node.parent)
# @show length(get_full_tree)
# @show length(possible_actions)
# get all possible subtrees as states, and all posible actions 
# current_node_id = node.node_id
# get_full_tree = []
# for d in node.depth:-1:1
#     current_node_id = soltree[current_node_id].parent
#     append!(get_full_tree, soltree[current_node_id].children)
# end
# push!(get_full_tree, current_node_id)
# Example Data
# E = length(collect(values(states_history))[1]) # epochs
# N = length(states_history)    # states

# # Simulated value history: (epochs × states)
# method1_values = hcat(values(states_history)...)
# method2_values = hcat(values(targets_history)...)

# # Define line styles for the two methods
# method_styles = [:solid, :dash]

# # Choose a color palette (one color per state)
# colors = distinguishable_colors(N)

# # Start plotting
# plt = Plots.plot(title="State Values Over Time (Both Methods)", xlabel="Epoch", ylabel="Value")

# for state in 1:N
#     # Plot for Method 1 (solid)
#     Plots.plot!(plt, 1:E, method1_values[:, state],
#           label="State $state (TD(0) tabular)", color=colors[state], linestyle=method_styles[1], legend=false)
    
#     # Plot for Method 2 (dashed)
#     Plots.plot!(plt, 1:E, method2_values[:, state],
#           label="State $state (NN tabular)", color=colors[state], linestyle=method_styles[2], legend=false)
# end

# display(plt)
