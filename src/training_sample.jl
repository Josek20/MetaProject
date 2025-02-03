mutable struct TrainingSample{D, S, E, P, HP, HN, IE}
    training_data::D
    saturated::S
    expression::E
    proof::P
    hp::HP
    hn::HN
    initial_expr::IE
end


# function TrainingSample(ex::Expr)
#     return TrainingSample{Union{ProductNode, Nothing}, Bool, ExprWithHash, Vector, Matrix, Matrix, Expr}(nothing, false, ExprWithHash(ex), [], [], [], ex)
# end


# function TrainingSample(list_of_ex::Vector)
#     return map(ex->TrainingSample{Union{ProductNode, Nothing}, Bool, ExprWithHash, Vector, Matrix, Matrix, Expr}(nothing, false, ExprWithHash(ex), [], zeros(0,0), zeros(0,0), ex), list_of_ex)
# end


function isbetter(a::TrainingSample, b::TrainingSample)
    size_a = exp_size(a.expression) 
    size_b = exp_size(b.expression)
    if size_a > size_b
        return true
    elseif size_a == size_b && length(a.proof) > length(b.proof)
        return true
    else
        return false
    end
end

