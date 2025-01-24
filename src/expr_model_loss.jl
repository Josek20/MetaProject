logistic(x) = log(1 + exp(x))
Heaviside(x) = x > 0 ? 1 : 0
hinge0(x::T) where {T} = x > 0 ? x : zero(T)
hinge1(x) = hinge0(x + 1)


function loss(heuristic, big_vector::ProductNode, hp, hn, surrogate::Function = softplus, agg::Function = mean)
    o = vec(heuristic(big_vector))
    return agg(surrogate.(o[hp] - o[hn]))
end


function hardloss(m, (ds, I₊, I₋))
    o = vec(MyModule.heuristic(m, ds))
    sum(Heaviside.(o[I₊] - o[I₋]))
end


function lossfun(m, (ds, I₊, I₋))
    o = vec(MyModule.heuristic(m, ds))
    mean(softplus.(o[I₊] - o[I₋]))
end