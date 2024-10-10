mutable struct ExprWithHash
    ex::Union{Expr, Symbol, Number}
	head::Union{Symbol, Number}
	args::Vector
	hash::UInt
end

function ExprWithHash(ex::Expr)
	args = ExprWithHash.(ex.args)
	head = ex.head#SymbolWithHash(ex)
	h = hash(hash(head), hash(args))
	ExprWithHash(ex, head, args, h) 
end

function ExprWithHash(ex::Symbol)
	args = []
	head = ex
	h = hash(ex)
	ExprWithHash(ex, head, args, h) 
end


function ExprWithHash(ex::Number)
	args = []
	head = ex
	h = hash(ex)
	ExprWithHash(ex, head, args, h) 
end


Base.hash(e::ExprWithHash, b::UInt) = hash(e.hash, b)
function old_traverse_expr!(ex::ExprWithHash, matchers::Vector{AbstractRule}, tree_ind::Int, trav_indexs::Vector{Int}, tmp::Vector{Tuple{Vector{Int}, Int}}, caching::LRU{ExprWithHash, Vector})
    if !isa(ex.ex, Expr)
        return
    end
    if haskey(caching, ex)
        b = copy(trav_indexs)
        append!(tmp, [(b, i) for i in caching[ex]])
        for (ind, i) in enumerate(ex.args)
            # Update the traversal index path with the current index
            push!(trav_indexs, ind)

            # Recursively traverse the sub-expression
            old_traverse_expr!(i, matchers, tree_ind, trav_indexs, tmp, caching)

            # After recursion, pop the last index to backtrack to the correct level
            pop!(trav_indexs)
        end
    end
    get!(caching, ex) do
        a = filter(em->!isnothing(em[2]), collect(enumerate(rt(ex.ex) for rt in matchers)))
        if !isempty(a)
            b = copy(trav_indexs)
            append!(tmp, [(b, i[1]) for i in a])
        end

        # Traverse sub-expressions
        for (ind, i) in enumerate(ex.args)
            # Update the traversal index path with the current index
            push!(trav_indexs, ind)

            # Recursively traverse the sub-expression
            old_traverse_expr!(i, matchers, tree_ind, trav_indexs, tmp, caching)

            # After recursion, pop the last index to backtrack to the correct level
            pop!(trav_indexs)
        end
        return isempty(a) ? [] : [i[1] for i in a]
    end
end

cache = LRU{ExprWithHash, Vector}(maxsize=10000)
tmp = Tuple{Vector{Int}, Int}[]
myex = :( (v0 + v1) + 119 <= min((v0 + v1) + 120, v2) && ((((v0 + v1) - v2) + 127) / (8 / 8) + v2) - 1 <= min(((((v0 + v1) - v2) + 134) / 16) * 16 + v2, (v0 + v1) + 119))
a = ExprWithHash(myex) 
@benchmark old_traverse_expr!(a, theory, 1, Int[], tmp, cache)