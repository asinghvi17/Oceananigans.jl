"""
    struct MortonArray{T, N} <: AbstractArray{T, 3}

A three dimensional array organized in memory as Morton curve. 
The array is indexed in with `[i, j, k] <= prod.(nested_sizes)`
The `underlying_data` is a N-dimensional array with `N = length(cumulative_sizes)` and is
indexed in with `[h₁, h₂,... hₙ] <= grid_sizes` where `h` are all `gilbertindices`
"""
struct MortonArray{T} <: AbstractArray{T, 3}
    underlying_data   :: AbstractArray{T, 1}
    min_axis          :: Int
    array_size        :: Tuple
end

using Base: @propagate_inbounds

@propagate_inbounds Base.getindex(h::MortonArray, i, j, k)       =  getindex(h.underlying_data,      morton_encode3d(i, j, k, h.min_axis)...)
@propagate_inbounds Base.setindex!(h::MortonArray, val, i, j, k) = setindex!(h.underlying_data, val, morton_encode3d(i, j, k, h.min_axis)...)
@propagate_inbounds Base.lastindex(h::MortonArray)               = lastindex(size(h))
@propagate_inbounds Base.lastindex(h::MortonArray, dim)          = lastindex(size(h, dim))

Base.size(h::MortonArray)      = h.array_size
Base.size(h::MortonArray, dim) = h.array_size[dim]

function MortonArray(FT, arch, underlying_size)
    Nx, Ny, Nz = underlying_size

    underlying_data   = zeros(FT, arch, Nx * Ny * Nz)

    Nx2 = Base.nextpow(2, Nx)
    Ny2 = Base.nextpow(2, Ny)
    Nz2 = Base.nextpow(2, Nz)
    min_axis = min(ndigits(Nx2, base = 2), ndigits(Ny2, base = 2), ndigits(Nz2, base = 2))

    return MortonArray(underlying_data, min_axis, underlying_size)
end

Adapt.adapt_structure

"""
diagnostic to investigate the memory layout of a `MortonArray`.
returns the order of indices in x, y and z and the three-dimensionally
ordered `(c_ord_x, c_ord_y, c_ord_z)`
"""
function memory_layout(h::MortonArray)
    nx, ny, nz = size(h)
    index = []
    list  = []
    c_ord_x = Int[]
    c_ord_y = Int[]
    c_ord_z = Int[]
    for i in 1:nx, j in 1:ny, k in 1:nz
        push!(list, (i, j, k))
        push!(c_ord_x, i)
        push!(c_ord_y, j)
        push!(c_ord_z, k)
        push!(index, morton_encode3d(i, j, k, h.min_axis))
    end

    perm = sortperm(index)
    list = list[perm]

    c_x = Int[]
    c_y = Int[]
    c_z = Int[]

    for l in list
        push!(c_x, l[1])
        push!(c_y, l[2])
        push!(c_z, l[3])
    end

    return c_x, c_y, c_z, c_ord_x, c_ord_y, c_ord_z
end 

"""
    function morton_encode3d(i, j, k, min_axis)

returns the encoded morton index corresponding to 3D index (i, j, k).
`min_axis` is the minimum between the number of bits of the smallest powers
of 2 larger than the data size
"""
@inline function morton_encode3d(i, j, k, min_axis)
	i = i - 1
	j = j - 1

    d = 0
    pos = 0
    for t in 0:min_axis-1
        a = i & 0x1
        b = j & 0x1
		c = k & 0x1
        i = i >> 1
        j = j >> 1
		k = k >> 1
        temp = ((c<<2)|(b<<1)|a)
        d |= temp << pos
        pos = pos + 3
	end
    d |= (i | j | k) << pos

    return d + 1
end