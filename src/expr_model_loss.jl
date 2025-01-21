logistic(x) = log(1 + exp(x))
hinge(x) = min(0, max(1, x + 1))
# loss01(x) = x + 1 > 0 ? x + 1 : 0

function loss(heuristic, big_vector::ProductNode, hp, hn, surrogate::Function = softplus, agg::Function = mean)
    o = vec(heuristic(big_vector))
    return agg(surrogate.(o[hp] - o[hn]))
end
