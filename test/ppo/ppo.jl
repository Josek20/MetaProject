using MyModule
using MyModule.Flux
using MyModule.Mill
using MyModule: intern!, exp_size, OnlyNode, NodeID, Node, all_expand
using Serialization
using Optimisers


experiment_name = "ppo"
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


function forward(ex, policy_model; max_expansions=50)
    soltree = Dict{UInt64, Node}()
    current = Node(ex, (), hash(ex), 0)
    soltree[current.node_id] = current
    smallest_node = current
    smallest_node_size = exp_size(ex)
    rolled_applied_rule = []
    rewrite_sequence = [ex]
    for step in 1:max_expansions
        ex = current.ex
        new_ex, rules_applied = all_expand(ex, theory)
        new_nodes = map(x->Node(x[1], x[2], current.node_id, current.depth + 1), zip(new_ex, rules_applied))
        new_nodes = filter(x->MyModule.push_to_tree!(soltree, x), new_nodes)
        isempty(new_nodes) && break
        o = map(new_nodes) do nn
            policy_model(nn.ex)[nn.rule_index[2]]
        end
        min_index = argmax(o)
        next_current = new_nodes[min_index]
        current_size = exp_size(next_current.ex)
        current = next_current
        push!(rolled_applied_rule, current.rule_index[2])
        push!(rewrite_sequence, current.ex)
        if smallest_node_size > current_size
            smallest_node_size = current_size
            smallest_node = current
        end
    end
    return smallest_node, rewrite_sequence, rolled_applied_rule
end


function advatages_estimation(all_values, rolled_out, rewards, t=1; γ = 0.9, λ = 0.3)
    t + 1 >= length(rolled_out) && return 0
    current_state, next_state = rolled_out[t:t + 1]
    # δₜ = rewards[t] + γ * only(value_model(next_state)) - only(value_model(current_state))
    δₜ = rewards[t] + γ * all_values[next_state] - all_values[current_state]
    Aₜ = (γ * λ) ^ (t - 1) * δₜ + advatages_estimation(all_values, rolled_out, rewards, t + 1)
    return Aₜ
end

 
function policy_loss(old_policy_model, new_policy_model, value_model, rewrite_sequence, training_data, rewards; ϵ = 0.2)
    # new_policy = MyModule.heuristic(old_policy_model, training_data)
    # old_policy = MyModule.heuristic(new_policy_model, training_data)
    # all_values = MyModule.heuristic(value_model, training_data)
    new_policy = old_policy_model(training_data)
    old_policy = new_policy_model(training_data)
    all_values = value_model.heuristic(training_data)
    ϵ = 0.2
    loss = 0
    for t in 1:length(rewrite_sequence) - 1
        r0 = new_policy[rewrite_sequence[t], t] ÷ old_policy[rewrite_sequence[t], t]
        @show r0
        Aₜ = advatages_estimation(all_values, collect(1:length(rewrite_sequence)), rewards, t)
        loss += min(r0 * Aₜ, min(r0, 1 - ϵ, 1 + ϵ) * Aₜ)
    end
    loss /= length(rewrite_sequence) - 1
end


function value_loss(value_model, rolled_out, rewards)
    tmp = value_model(rolled_out)[1:end-1] - rewards
    sum((tmp) .^ 2) / length(rewards)
end


epochs = 5

optimizer = ADAM()
value_opt_state = Flux.setup(optimizer, value_model.heuristic)
policy_opt_state = Flux.setup(optimizer, policy_model.heuristic)
# ex = :(min(v0 * 68 + (min(v2 * 4, 30) + v1 * 34), 131) - (v0 * 68 + (v2 * 4 + v1 * 34)) <= 1018)
ex = :(100 - 12 <= 1018)
previous_policy_models = [deepcopy(policy_model.heuristic)]
for ep in 1:epochs
    all_smallest = []
    @show ep
    t = @elapsed for ex in data[1:1]
        ex = intern!(ex)
        smallest_node, rewrite_sequence, rules_applied = forward(ex, policy_model)
        push!(all_smallest, smallest_node)
        rewards = [exp_size(rewrite_sequence[i]) - exp_size(rewrite_sequence[i + 1]) for i in 1:length(rewrite_sequence) - 1]
        tmp = MyModule.no_reduce_multiple_fast_ex2mill([MyModule.expr(MyModule.nc, i) for i in rewrite_sequence])
        
        # all_policies = policy_model(tmp)
        tmp = value_model(tmp)
        old_values = policy_model(ex)
        if ep == 1
            old_pm = previous_policy_models[end]
        elseif ep >= 2
            push!(previous_policy_models, deepcopy(policy_model.heuristic))
            old_pm = previous_policy_models[end-1]
        end
        sap, pol_grad = Flux.Zygote.withgradient(policy_model.heuristic) do pm
            policy_loss(old_pm, pm, value_model, rules_applied, tmp, rewards)
        end
        sav, val_grad = Flux.Zygote.withgradient(value_model.heuristic) do vm
            value_loss(vm, tmp, rewards)
        end
        @show sav
        @show sap
        Optimisers.update!(value_opt_state, value_model.heuristic, val_grad)
        Optimisers.update!(policy_opt_state, policy_model.heuristic, pol_grad)
        new_values = policy_model(ex)
        @assert old_values != new_values
        @assert sum(old_values - new_values) != 0f0
    end
    serialize("models/trained_policy_model_ep$(ep)_$(experiment_name).bin", policy_model)
    serialize("models/trained_value_model_ep$(ep)_$(experiment_name).bin", value_model)
    @show t
    aes = [exp_size(intern!(ex)) - exp_size(sm.ex) for (ex, sm) in zip(data, all_smallest)]
    @show sum(aes) / length(aes)
end