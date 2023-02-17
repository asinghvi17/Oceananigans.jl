using GilbertCurves

function generate_methods(size)
    list = GilbertCurves.gilbertindices(size)
    for (idx, l) in enumerate(list)
        @eval begin
            @inline encoded_index(::Val{$l.I}) = $idx
            @inline decoded_index(::Val{$idx}) = $l.I
        end
    end
end

struct GilbertArray{T} <: AbstractArray{T, 3}
    matrix_data :: AbstractArray{T, 2}
    size        :: Tuple
end

using Base: @propagate_inbounds

# Encoded and decoded indices
@inline jk2h(j, k) = encoded_index(Val((j, k)))
@inline h2jk(h)    = decoded_index(Val(h))

@propagate_inbounds Base.getindex(h::GilbertArray, i, j, k)       =  getindex(h.matrix_data,      i, jk2h(j, k)...)
@propagate_inbounds Base.setindex!(h::GilbertArray, val, i, j, k) = setindex!(h.matrix_data, val, i, jk2h(j, k)...)
@propagate_inbounds Base.lastindex(h::GilbertArray)               = lastindex(prod(h.size))
@propagate_inbounds Base.lastindex(h::GilbertArray, dim)          = lastindex(size(h, dim))

Base.size(h::GilbertArray)      = h.size
Base.size(h::GilbertArray, dim) = h.size[dim]
