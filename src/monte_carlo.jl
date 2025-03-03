using MyModule
using MyModule.Mill
using MyModule.Flux
using MyModule.DataStructures
using Optimisers
using Serialization
using BenchmarkTools
using ProfileCanvas

using MyModule: load_data, preprosses_data_to_expressions
using MyModule: exp_size, initialize_tree_search, all_expand, intern!, Node, build_tree, expand_node!, NodeID, TreePolicyModel, Node, push_to_tree!, extract_smallest_node, OnlyNode

experiment_name = "monte_carlo_alpha_zero"
train_data_path = "./data/neural_rewrter/train.json"
train_data = load_data(train_data_path)[1:1_000]
train_data = filter(x->!occursin("select", x[1]), train_data)
train_data = preprosses_data_to_expressions(train_data)
sorted_data = sort(train_data, by=x->MyModule.exp_size(x))
data = sorted_data

input_dim = 512
hidden_dim = 256
max_steps = 50
hidden_size = 64

function ffnn(idim, hidden_size, layers)
    layers == 1 && return Dense(idim, hidden_size, Flux.gelu)
    layers == 2 && return Flux.Chain(Dense(idim, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu))
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

value_model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Flux.Chain(Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, 1)),
    );


policy_model = ExprModel(
    head_model,
    Mill.SegmentedSum(hidden_size),
    args_model,
    Flux.Chain(Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, hidden_size, Flux.gelu), Dense(hidden_size, length(theory)), softmax)
    );



has_terminal(ex::Vector{NodeID}, terminal_node = intern!(:1)) = terminal_node ∈ ex
has_terminal(ex::Vector{Expr}, terminal_node = :(1)) = terminal_node ∈ ex

function get_all_subtrees!(ex::NodeID, all_subtrees::Vector, parent=[])
    node = MyModule.nc[ex]
    for (ind,i) in enumerate([node.left, node.right])
        if !(MyModule.nc[i].iscall) && MyModule.nc[i].head ∉ [:&&, :||]
            continue
        end
        if i == MyModule.nullid
            continue
        end
        pos = vcat(parent, ind)
        push!(all_subtrees, (i, pos))
        get_all_subtrees!(i, all_subtrees, pos)
    end
end

function my_rewrite!(ex::NodeID, pos, new_exp_part::NodeID)
    if isempty(pos)
        return new_exp_part
    end
    node = MyModule.nc[ex]

    if pos[1] == 1
        new_part = my_rewrite!(node.left, pos[2:end], new_exp_part)
        # node.left = new_part
        new_node = OnlyNode(node.head, node.iscall, node.v, new_part, node.right)
    else        
        new_part = my_rewrite!(node.right, pos[2:end], new_exp_part)
        # node.right = new_part
        new_node = OnlyNode(node.head, node.iscall, node.v, node.left, new_part)
    end
    return get!(MyModule.nc, new_node)
end

function rollout(ex, model; max_expansions=10)
    soltree = Dict{UInt64, Node}()
    current = Node(ex, (), hash(ex), 0)
    soltree[current.node_id] = current
    smallest_node = current
    smallest_node_size = exp_size(ex)
    for step in 1:max_expansions
        ex = current.ex
        subtrees = [(ex, [])]
        get_all_subtrees!(ex, subtrees)
        tree_action = []
        for (st, pos) in subtrees
            new_ex = [(intern!(r(st)), pos, ind, st) for (ind,r) in enumerate(theory)]
            # filter empty rewrites
            filtered_ex = filter(x->!isnothing(x[1]), new_ex)
            new_ex = map(x->(x..., my_rewrite!(ex, x[2], x[1])), filtered_ex)
            # filter repeated 
            filtered_ex = filter(x->!haskey(soltree, hash(x[5])), new_ex)
            append!(tree_action, filtered_ex)
        end
        isempty(tree_action) && break
        root_embedding = MyModule.general_cached_inference(ex, Expr, model)
        o = map(tree_action) do (_, _, rule_id, subtree, _)
            subtree_embedding = MyModule.general_cached_inference(subtree, Expr, model)
            model.heuristic(subtree_embedding)[rule_id]
        end
        min_index = argmax(o)
        _, pos, rule_id, subtree, current_exp = tree_action[min_index]
        current_size = exp_size(current_exp)
        current = Node(current_exp, (pos, rule_id), current.node_id, current.depth + 1)
        soltree[current.node_id] = current
        if smallest_node_size > current_size
            smallest_node_size = current_size
            smallest_node = current
        end
    end
    return soltree, smallest_node, []
end


function rollout_expand!(parent::Node, soltree, open_list, model; theory=theory)
    ex = parent.ex
    new_ex, rules_applied = all_expand(ex, theory)
    new_nodes = map(x->Node(x[1], x[2], parent.node_id, parent.depth + 1), zip(new_ex, rules_applied))
    new_nodes = filter(x->push_to_tree!(soltree, x), new_nodes)
    isempty(new_nodes) && return
    o = model(ex)
    for ((p, rid), n) in zip(rules_applied, new_nodes)
        enqueue!(open_list, n, o[rid])
    end
    nodes_ids = map(x->x.node_id, new_nodes)
    append!(parent.children, nodes_ids)
end


function build_rollout_tree!(soltree, open_list, close_list, model; max_expansions=1000, max_depth=10)
    expansions = 0
    while !isempty(open_list)
        expansions == max_expansions && break
        node, _ = dequeue_pair!(open_list)
        push!(close_list, node.node_id)

        node.depth == max_depth && continue
        
        rollout_expand!(node, soltree, open_list, model)
        expansions += 1
    end
end


function initialize_rollout_tree_search(ex, model; max_expansions=1000, max_depth=10)
    open_list = PriorityQueue{Node, Float32}(Base.Order.Reverse)
    close_list = Set{UInt64}()

    soltree = Dict{UInt64, Node}()
    root = Node(ex, (), hash(ex), 0)
    soltree[root.node_id] = root
    enqueue!(open_list, root, 1)
    build_rollout_tree!(soltree, open_list, close_list, model, max_expansions=max_expansions, max_depth=max_depth)

    smallest_node = extract_smallest_node(soltree)
    return(soltree, smallest_node, root)
end


function backpropagate_stats!(n, nodes_stats, soltree)
    if n.parent == n.node_id
        return
    end
    nodes_stats[n.parent][1] += 1
    nodes_stats[n.parent][2] += nodes_stats[n.node_id][2] + nodes_stats[n.node_id][3]
    backpropagate_stats!(soltree[n.parent], nodes_stats, soltree)
end


function monte_carlo_expand!(parent, soltree, open_list)
    new_ex, rules_applied = all_expand(parent.ex, theory)
    filtered_new_ex = filter(x->parent.ex != x, new_ex)
    
    new_nodes = map(x->Node(x[1], x[2], parent.node_id, parent.depth + 1), zip(filtered_new_ex, rules_applied))
    # tmp = filter(x->push_to_tree!(soltree, x), new_nodes)
    for x in new_nodes
        res = push_to_tree!(soltree, x)
        # if res && x.node_id != parent.node_id && x.node_id ∉ parent.children && x.parent == parent.node_id
        if res && x.node_id ∉ parent.children
            push!(parent.children, x.node_id)
            enqueue!(open_list, x, 0)
            return 
        end
    end
    # if isempty(filtered_new_ex)
    _, _ = dequeue_pair!(open_list)
    monte_carlo_expand!(first(open_list)[1], soltree, open_list)
    # end
end


function monte_carlo_search!(root, value_model, policy_model, open_list, rolled_out, nodes_stats, soltree, close_list; max_expansions=10)
    root_node = root
    # @show max_expansions
    initial_size = exp_size(root_node.ex)
    for step in 1:max_expansions
        game_state, _ = first(open_list)
        # @show step
        # @show game_state.ex
        monte_carlo_expand!(game_state, soltree, open_list)
        # game_states = filter(x-> x.node_id ∉ close_list, collect(keys(open_list)))
        next_game_states = collect(keys(open_list))
        has_terminal(map(x->x.ex, next_game_states)) && break
        @assert length(next_game_states) != 0
        filtered_game_states = filter(x->x.node_id ∉ rolled_out, next_game_states)
        # @show length(filtered_game_states)
        if length(filtered_game_states) != 0
            for i in filtered_game_states
                push!(rolled_out, i.node_id)
            end
            trees = [rollout(gs.ex, policy_model, max_expansions=30) for gs in filtered_game_states]
            # trees = [initialize_rollout_tree_search(gs.ex, policy_model, max_expansions=20) for gs in filtered_game_states]
            
            for (rt, (st, sm, cl)) in zip(filtered_game_states, trees)
                reward = initial_size - exp_size(sm.ex)
                # nodes_stats[rt.node_id] = (;visited_count=1,value_sum=value_model(sm.ex),reward=reward)
                nodes_stats[rt.node_id] = [1f0, only(value_model(sm.ex)), reward]
                backpropagate_stats!(rt, nodes_stats, soltree)
            end
            for g_state in next_game_states
                v = nodes_stats[g_state.node_id][2]
                n = nodes_stats[g_state.node_id][1]
                N = nodes_stats[g_state.parent][1]
                uct = v ÷ n + 1.2 * √log(N) ÷ n
                open_list[g_state] = uct
            end
        end        
        push!(close_list, game_state.node_id)
    end
end


function init_monte_carlo(game_state, value_model, policy_model; max_expansions=10)
    soltree = Dict{UInt64, Node}()
    # values_nodes = Dict{UInt64, Float32}()
    # nodes_stats = Dict{UInt64, NamedTuple}()
    nodes_stats = Dict{UInt64, Vector{Float32}}()
    open_list = PriorityQueue{Node, Float32}(Base.Order.Reverse)
    close_list = Set{UInt64}()
    rolled_out = Set{UInt64}()
    root = Node(game_state, (), hash(game_state), 0)
    soltree[root.node_id] = root
    # n_visited[root.node_id] = 1
    # values_nodes[root.node_id] = value_model(root.ex)
    # nodes_stats[root.node_id] = (;visited_count=0,reward=0,value_sum=0)
    nodes_stats[root.node_id] = [0f0,0f0,0f0]
    push!(close_list, root.node_id)
    expand_node!(root, soltree, open_list, value_model)
    rollout_nodes = keys(open_list)
    for i in rollout_nodes
        push!(rolled_out, i.node_id)
    end
    
    trees = [rollout(gs.ex, policy_model, max_expansions=30) for gs in rollout_nodes]
    # trees = [initialize_rollout_tree_search(gs.ex, policy_model, max_expansions=20) for gs in rollout_nodes]
            
    for (rt, (st, sm, cl)) in zip(rollout_nodes, trees)
        reward = exp_size(root.ex) - exp_size(sm.ex)
        # nodes_stats[rt.node_id] = (;visited_count=1,value_sum=value_model(sm.ex),reward=reward)
        nodes_stats[rt.node_id] = [1f0, only(value_model(sm.ex)), reward]
        backpropagate_stats!(rt, nodes_stats, soltree)
    end
    for g_state in rollout_nodes
        v = nodes_stats[g_state.node_id][2]
        n = nodes_stats[g_state.node_id][1]
        N = nodes_stats[g_state.parent][1]
        uct = v ÷ n + 1.2 * √log(N) ÷ n
        open_list[g_state] = uct
    end
    # open_list[root] = 0
    push!(rolled_out, root.node_id)
    monte_carlo_search!(root, value_model, policy_model, open_list, rolled_out, nodes_stats, soltree, close_list, max_expansions=max_expansions)
    smallest_node = MyModule.extract_smallest_node(soltree)
    nodes_in_proof, proof = MyModule.extract_proof(smallest_node, soltree)
    # return vcat(root.ex, map(x->x.ex, nodes_in_proof)), proof
    return map(x->x.ex, nodes_in_proof), proof
end


@assert 0 == 1
max_epochs = 1
inner_epochs = 10
optimizer=ADAM()
value_opt_state = Flux.setup(optimizer, value_model)
policy_opt_state = Flux.setup(optimizer, policy_model)
sqnorm(x) = sum(abs2, x)
game_data = [(;ds=nothing, reward=0, initial_expr=i, proof=[]) for i in data]
batch_size = 100
@elapsed for epoch in 1:max_epochs
    @show epoch
    empty!(MyModule.memoize_cache(MyModule.exp_size))
    empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
    empty!(MyModule.memoize_cache(MyModule.all_expand))
    t_total = @elapsed game_data = map(enumerate(game_data)) do (ind, i)
        ex = i.initial_expr
        @show ex
        nodes_in_proof, proof = init_monte_carlo(intern!(ex), value_model, policy_model, max_expansions=50)
        new_reward = exp_size(nodes_in_proof[end])
        if i.reward < new_reward || isnothing(i.ds)
            all_expr = map(x->MyModule.expr(MyModule.nc,x), nodes_in_proof)
            # return(;ds=MyModule.no_reduce_multiple_fast_ex2mill(all_expr), reward=new_reward, initial_expr=ex)
            return(;ds=all_expr, reward=new_reward, initial_expr=ex, proof=proof)
        else
            return(i)
        end
    end
    processed_training_data = []

    for i in 1:batch_size:length(game_data)
        game_data_batch = game_data[i: (i+batch_size < length(game_data) ? i+batch_size-1 : length(game_data))]
        ds = []
        I₊ = Float32[]
        Pₚ = []
        for x in game_data_batch
            rw = zeros(Float32,length(x.ds)) .+ x.reward
            append!(I₊, rw)
            append!(ds, x.ds)
            policy_matrix = zeros(Float32, length(x.ds), 150)
            matrix_indexes = map(i->i[2][2] * length(x.ds) - 1 + (i[1] - 1), enumerate(x.proof))
            @assert length(matrix_indexes) == length(x.ds)
            policy_matrix[matrix_indexes] .= 1
            # for i in 1:size(policy_matrix, 1)
            #     policy_matrix[i, matrix_indexes[i]] = 1
            # end
            push!(Pₚ, policy_matrix)
        end
        Pₚ = vcat(Pₚ...)
        new_ds = MyModule.no_reduce_multiple_fast_ex2mill(ds)
        push!(processed_training_data, (new_ds, I₊, Pₚ))
    end

    @show t_total
    for i in 1:inner_epochs
        total_loss = 0
        t_training = @elapsed for (i,r,pt) in processed_training_data
            # values_loss = (value_model.() - final_reward)
            # policy_loss = (all_choose_probs * log(the_rest))
            sa, grad = Flux.Zygote.withgradient(value_model, policy_model) do vm, pm
                o = vec(MyModule.heuristic(vm,i))
                o1 = pt * log.(MyModule.heuristic(pm, i))
                sum((o - r).^2) + sum(sqnorm, Flux.params(vm)) + sum(sqnorm, Flux.params(pm)) - sum(o1)
            end
            # @show sa
            total_loss += sa
            # @show typeof(grad)
            # @show length(grad)
            Optimisers.update!(value_opt_state, value_model, grad[1])
            Optimisers.update!(policy_opt_state, policy_model, grad[2])
        end
        @show total_loss
        @show t_training
    end
    # println("Epoch $epoch: Value Loss = $total_loss")
    serialize("models/trained_value_model_$(experiment_name)_ep$(epoch)_hidden$(hidden_size).bin", value_model)
    serialize("models/trained_policy_model_$(experiment_name)_ep$(epoch)_hidden$(hidden_size).bin", policy_model)
end
