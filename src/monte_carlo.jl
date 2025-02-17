using MyModule
using MyModule.Mill
using MyModule.Flux
using MyModule.DataStructures
using Optimisers
using Serialization
using BenchmarkTools
using ProfileCanvas

using MyModule: exp_size, initialize_tree_search, all_expand, intern!, Node, build_tree, expand_node!, NodeID, TreePolicyModel, Node, push_to_tree!, extract_smallest_node

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
function rollout(ex, model; max_expansions=10)
    soltree = Dict{UInt64, Node}()
    current = Node(ex, (), hash(ex), 0)
    soltree[current.node_id] = current
    smallest_node = current
    smallest_node_size = exp_size(ex)
    for step in 1:max_expansions
        ex = current.ex
        new_ex, _ = all_expand(ex, theory)
        new_nodes = map(x->Node(x, (), current.node_id, current.depth + 1), new_ex)
        filtered_nodes = filter(x->push_to_tree!(soltree, x), new_nodes)
        isempty(filtered_nodes) && break
        # @show length(filtered_nodes)
        o = map(x->model(x.ex), filtered_nodes)
        min_index = argmax(o)
        current = filtered_nodes[min_index]
        current_size = exp_size(current.ex)
        if smallest_node_size > current_size
            smallest_node_size = current_size
            smallest_node = current
        end
    end
    return soltree, smallest_node, []
end


function backpropagate_stats!(n, nodes_stats, soltree)
    if n.parent == n.node_id
        return
    end
    # nodes_stats[n.parent].visited_count += 1
    nodes_stats[n.parent][1] += 1
    nodes_stats[n.parent][2] += nodes_stats[n.node_id][2] + nodes_stats[n.node_id][3]
    backpropagate_stats!(soltree[n.parent], nodes_stats, soltree)
end


function monte_carlo_expand!(parent, soltree, open_list, model)
    new_ex, rules_applied = all_expand(parent.ex, theory)
    new_nodes = map(x->Node(x, (), parent.node_id, parent.depth + 1), new_ex)
    for x in new_nodes
        res = push_to_tree!(soltree, x)
        if res
            enqueue!(open_list, x, 0)
            push!(parent.children, x.node_id)
            return 
        end
    end
    n,v = dequeue_pair!(open_list)
    return
end


function monte_carlo_search!(value_model, policy_model, open_list, rolled_out, nodes_stats, soltree, close_list; max_expansions=10)
    root_node, _ = first(open_list)
    initial_size = exp_size(root_node.ex)
    for step in 1:max_expansions
        # game_state, _ = dequeue_pair!(open_list)
        game_state, _ = first(open_list)
        # @show game_state.ex
        monte_carlo_expand!(game_state, soltree, open_list, value_model)
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
            # trees = [rollout(gs.ex, policy_model, max_expansions=30) for gs in filtered_game_states]
            trees = [initialize_tree_search(gs.ex, policy_model, max_expansions=20) for gs in filtered_game_states]
            
            for (rt, (st, sm, cl)) in zip(filtered_game_states, trees)
                reward = initial_size - exp_size(sm.ex)
                # nodes_stats[rt.node_id] = (;visited_count=1,value_sum=value_model(sm.ex),reward=reward)
                nodes_stats[rt.node_id] = [1f0, only(value_model(sm.ex)), reward]
                backpropagate_stats!(rt, nodes_stats, soltree)
            end
            for g_state in next_game_states
                v = nodes_stats[g_state.node_id][3]
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
    root = Node(game_state, (), hash(game_state), 0)
    soltree[root.node_id] = root
    # n_visited[root.node_id] = 1
    # values_nodes[root.node_id] = value_model(root.ex)
    # nodes_stats[root.node_id] = (;visited_count=0,reward=0,value_sum=0)
    nodes_stats[root.node_id] = [0f0,0f0,0f0]
    push!(close_list, root.node_id)
    open_list[root] = 0
    rolled_out = Set{UInt64}()
    push!(rolled_out, root.node_id)
    monte_carlo_search!(value_model, policy_model, open_list, rolled_out, nodes_stats, soltree, close_list, max_expansions=max_expansions)
    smallest_node = MyModule.extract_smallest_node(soltree)
    nodes_in_proof, _ = MyModule.extract_proof(smallest_node, soltree)
    return vcat(root.ex, map(x->x.ex, nodes_in_proof))
end


# @assert 0 == 1
max_epochs = 10
inner_epochs = 10
optimizer=ADAM()
opt_state = Flux.setup(optimizer, value_model)
sqnorm(x) = sum(abs2, x)
game_data = [(;ds=nothing, reward=0, initial_expr=i) for i in data]
batch_size = 100
@elapsed for epoch in 1:max_epochs
    @show epoch
    empty!(MyModule.memoize_cache(MyModule.exp_size))
    empty!(MyModule.memoize_cache(MyModule.general_cached_inference))
    empty!(MyModule.memoize_cache(MyModule.all_expand))
    t_total = @elapsed game_data = map(enumerate(game_data)) do (ind, i)
        ex = i.initial_expr
        @show ex
        nodes_in_proof = init_monte_carlo(intern!(ex), value_model, value_model, max_expansions=50)
        new_reward = exp_size(nodes_in_proof[end])
        if i.reward < new_reward || isnothing(i.ds)
            all_expr = map(x->MyModule.expr(MyModule.nc,x), nodes_in_proof)
            # return(;ds=MyModule.no_reduce_multiple_fast_ex2mill(all_expr), reward=new_reward, initial_expr=ex)
            return(;ds=all_expr, reward=new_reward, initial_expr=ex)
        else
            return(i)
        end
    end
    processed_training_data = []

    for i in 1:batch_size:length(game_data)
        game_data_batch = game_data[i: (i+batch_size < length(game_data) ? i+batch_size-1 : length(game_data))]
        ds = []
        I₊ = Float32[]
        for x in game_data_batch
            rw = zeros(Float32,length(x.ds)) .+ x.reward
            append!(I₊, rw)
            append!(ds, x.ds)    
        end
        new_ds = MyModule.no_reduce_multiple_fast_ex2mill(ds)
        push!(processed_training_data, (new_ds,I₊))
    end
    @show t_total
    for i in 1:inner_epochs
        total_loss = 0
        t_training = @elapsed for (i,r) in processed_training_data
            # values_loss = (value_model.() - final_reward)
            # policy_loss = (all_choose_probs * log(the_rest))
            sa, grad = Flux.Zygote.withgradient(value_model) do m
                o = vec(MyModule.heuristic(m,i))
                sum((o - r).^2) + sum(sqnorm, Flux.params(m))
            end
            # @show sa
            total_loss += sa
            Optimisers.update!(opt_state, value_model, only(grad))
        end
        @show total_loss
        @show t_training
    end
    # println("Epoch $epoch: Value Loss = $total_loss")
    serialize("models/trained_value_model_$(experiment_name)_ep$(epoch)_hidden$(hidden_size).bin", value_model)
end
# r1 = theory[end-25]
# r2 = theory[30]
# s = 414646
# b = MyModule.nc.nodemap[MyModule.nc.nodes[s]]
# tmp = :(!((((v0 * 67) / v1) * v1 + 17 * v1) + 1007 < 116))
# tmp = :(!(1021 + (v0 * 66 + (v2 * 2 + v1 * 33)) < v0 * 66 + (min(v2 * 2, 31) + v1 * 33) && 1021 + (v0 * 66 + (v2 * 2 + v1 * 33)) < 129))