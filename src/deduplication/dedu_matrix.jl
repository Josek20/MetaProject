struct DeduplicatedMatrix{T} <:AbstractMatrix{T}
	x::Matrix{T}
	ii::Vector{Int}
end

function DeduplicatedMatrix(x::Matrix)
	mask, ii = find_duplicates(x)
	DeduplicatedMatrix(x[:,mask], ii)
end

DeduplicatedMatrix(x::DeduplicatedMatrix) = DeduplicatedMatrix(x.x, x.ii)
DeduplicatedMatrix(x::DeduplicatedMatrix, outer_ii::Vector{<:Integer}) = DeduplicatedMatrix(x.x, x.ii[outer_ii])

Base.size(x::DeduplicatedMatrix) = (size(x.x, 1), length(x.ii))
Base.getindex(x::DeduplicatedMatrix, i::Int, j::Int) = x.x[i, x.ii[j]]
Base.Matrix(x::DeduplicatedMatrix) = x.x[:,x.ii]

(m::Flux.Chain)(x::DeduplicatedMatrix) = DeduplicatedMatrix(m(x.x), x.ii)
(m::Flux.Dense)(x::DeduplicatedMatrix) = DeduplicatedMatrix(m(x.x), x.ii)

function scatter_cols!(c::AbstractMatrix, a::AbstractMatrix, ii, α, β)
	size(a,1) == size(c,1) || error("c and a should have the same number of rows")
	size(a,2) ≤ maximum(ii) || error("a has to has to have at least $(maximum(ii)) columns")
	size(c,2) ≤ length(ii) || error("c has to has to have at least $(length(ii)) columns")
	@inbounds for (i, j) in enumerate(ii)
		for k in 1:size(a,1)
			c[k,i] = β*c[k,i] + α*a[k, j]
		end
	end
end

function gather_cols!(c::AbstractMatrix, a::AbstractMatrix, ii, α, β)
	size(a,1) == size(c,1) || error("c and a should have the same number of rows")
	size(c,2) ≤ maximum(ii) || error("c has to has to have at least $(maximum(ii)) columns")
	size(a,2) ≤ length(ii) || error("a has to has to have at least $(length(ii)) columns")
	@inbounds for (i, j) in enumerate(ii)
		for k in 1:size(a,1)
			c[k,j] = β*c[k,j] + α*a[k, i]
		end
	end
end

function ChainRulesCore.rrule(::Type{DeduplicatedMatrix}, a, ii)
    function dedu_pullback(ȳ)
    	Ȳ = unthunk(ȳ)
    	δx = zeros(eltype(a), size(a))
    	gather_cols!(δx, ȳ, ii, true,true)
    	NoTangent(), δx, NoTangent()
    end

    function dedu_pullback(ȳ::Tangent{Any, @NamedTuple{x::Matrix{Float32}, ii::ZeroTangent}})
    	Ȳ = unthunk(ȳ)
    	# @show typeof(Ȳ)
    	δx = Ȳ.x
    	NoTangent(), δx, NoTangent()
    end
    return DeduplicatedMatrix(a, ii), dedu_pullback
end
