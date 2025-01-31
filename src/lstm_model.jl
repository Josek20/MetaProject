using MyModule
using MyModule.Flux
using MyModule.LRUCache
using Optimisers
using Serialization

batch_size = 1

function prepare_dataset(n=typemax(Int))
    samples = deserialize("data/training_data/size_heuristic_training_samples1.bin")
    samples = vcat(samples...)
    samples = sort(samples, by=x->MyModule.exp_size(x.initial_expr))
    # samples = sort(samples, by=x->length(x.proof))
    last(samples, min(length(samples), n))
end


struct MyTreeRNN{WX,WH,B}
    Wₓ::WX
    Wₕ::WH
    b::B
end


function MyTreeRNN(input_size, hidden_size)
    WX = randn(Float32, hidden_size, input_size + 2)
    WH = randn(Float32, hidden_size, hidden_size)
    b = zeros(Float32, hidden_size)
    return MyTreeRNN(WX, WH, b)
end


MyModule.@my_cache LRU(maxsize=10_000) function cached_lstm_inference(node::MyModule.NodeID, m::MyTreeRNN, act=gelu, sym_enc=sym_enc)
    node = MyModule.nc[node]
    x = node.encoding
    if !(node.iscall) && node.head ∉ [:&&, :||]
        x = size(x, 1) == length(sym_enc) ? vcat(x, [0, 0]) : x
        res = act(m.Wₓ * x + m.b)
        return res
    end
    left = cached_lstm_inference(node.left, m)
    right = node.right != MyModule.nullid ? cached_lstm_inference(node.right, m) : zeros(Float32, size(left,1))
    y = size(x, 1) == length(sym_enc) ? vcat(x, [0, 0]) : x
    res = act(m.Wₓ * y + (m.Wₕ * left + m.Wₕ * right) + m.b)
    return res
end


function (m::MyTreeRNN)(node::MyModule.NodeID, act=gelu, sym_enc=sym_enc)
    node = MyModule.nc[node]
    x = node.encoding
    if !(node.iscall) && node.head ∉ [:&&, :||]
        x = size(x, 1) == length(sym_enc) ? vcat(x, [0, 0]) : x
        res = act(m.Wₓ * x + m.b)
        return res
    end
    left = m(node.left)
    right = node.right != MyModule.nullid ? m(node.right) : zeros(Float32, size(left,1))
    y = size(x, 1) == length(sym_enc) ? vcat(x, [0, 0]) : x
    res = act(m.Wₓ * y + (m.Wₕ * left + m.Wₕ * right) + m.b)
    return res
end



function hardloss1(m, ds, I₊, I₋)
    o = vcat(m.(ds)...)
    sum(MyModule.Heaviside.(o[I₊] - o[I₋]))
end


ex = :(v1 + v0)
ex = :(100 - 10)
inex = MyModule.intern!(ex)
m = Chain(MyTreeRNN(length(sym_enc), 64), Dense(64,1))
@show m(inex)
ex = :(v0 + v1)
ex = :(10 - 100 <= !v0)
inex = MyModule.intern!(ex)

@show m(inex)
ex = :(100 - 10 <= !v0)
inex = MyModule.intern!(ex)
@show m(inex)

training_samples = prepare_dataset();
samples = map(training_samples) do sample
    ds, hp, hn, tr = MyModule.get_training_data_from_proof(sample.proof, sample.initial_expr)
    (; ds = MyModule.deduplicate(ds),
       hp,
       hn,
       initial_expr = sample.initial_expr,
       goal_size = MyModule.exp_size(sample.expression),
       tr = [MyModule.intern!(ex) for ex in tr]
    )
end

optimizer=ADAM()
opt_state = Flux.setup(optimizer, m)

for (ind,i) in enumerate(samples[end-300: end-100])
    @show ind
    a, hp, hn = i.tr, i.hp, i.hn
    sa, grad = Flux.Zygote.withgradient(m) do h
        o = vcat(h.(a)...)
        sum(softplus.(o[hp] - o[hn]))
    end
    Optimisers.update!(opt_state, m, grad[1])
end

violations = [hardloss1(m, s.tr, s.hp, s.hn) for s in samples[end-300: end-100]]