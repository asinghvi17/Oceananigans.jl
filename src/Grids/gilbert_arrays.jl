using GilbertCurves, Primes

"""
    function generate_encoder_methods(size)

this function leverages the `gilbertindices` function in the `GilbertCurves` package
to generate methods for nested encoding functions that from an input `(i, j, k)`
return the associated gilbert index `h`
"""
function generate_encoder_methods(size)
    # Find optimal configuration for size and curves
    Nx, Ny, Nz = find_factorizations(size)
    curve = 0
    for (nx, ny, nz) in zip(Nx, Ny, Nz)
        encoder_func = Symbol(:encoded_index, curve)

        list = GilbertCurves.gilbertindices((nx, ny, nz))
        for (idx, l) in enumerate(list)
            @eval begin
                @inline $encoder_func(::Val{$l.I}) = $idx
                export $encoder_func
            end
        end
        curve+=1
    end
end

"""
    function find_factorization(size)

this function leverages the `factor` function in the `Primes` package to find a decomposition of 
`size` into suitable smaller chunks that fill up the space `size` 
"""
function find_factorizations(size)
    fact = factor.(Vector, [size...])
    while any(length.(fact) .!= length(fact[1]))
        ml   = min(length.(fact)...)
        fact = [length(f) > ml ? [f[1]*f[2], f[3:end]...] : f for f in fact]
    end
    return Tuple(reverse(tuple(fact[i]...)) for i in 1:3)
end

"""
    struct GilbertArray{T, N} <: AbstractArray{T, 3}

A three dimensional array organized in memory as a series of 
nested Gilbert curves. The array is indexed in with `[i, j, k] <= prod.(nested_sizes)`
The `underlying_data` is a N-dimensional array with `N = length(cumulative_sizes)` and is
indexed in with `[h₁, h₂,... hₙ] <= grid_sizes` where `h` are all `gilbertindices`
"""
struct GilbertArray{T, N} <: AbstractArray{T, 3}
    underlying_data   :: AbstractArray{T, N}
    nested_sizes      :: Tuple
    previous_sizes    :: Tuple
    cumulative_sizes  :: Tuple
    encoder_functions :: Tuple
end

using Base: @propagate_inbounds

# Encoded and decoded indices
@inline ijk2h(i, j, k, previous_sizes, sizes, encoder) = 
        unroll_encoders.(i, j, k, previous_sizes, sizes, encoder)

@propagate_inbounds Base.getindex(h::GilbertArray, i, j, k)       =  getindex(h.underlying_data,      ijk2h(i, j, k, h.previous_sizes, h.nested_sizes, h.encoder_functions)...)
@propagate_inbounds Base.setindex!(h::GilbertArray, val, i, j, k) = setindex!(h.underlying_data, val, ijk2h(i, j, k, h.previous_sizes, h.nested_sizes, h.encoder_functions)...)
@propagate_inbounds Base.lastindex(h::GilbertArray)               = lastindex(size(h))
@propagate_inbounds Base.lastindex(h::GilbertArray, dim)          = lastindex(size(h, dim))

Base.size(h::GilbertArray)      = h.cumulative_sizes[end]
Base.size(h::GilbertArray, dim) = h.cumulative_sizes[end][dim]

@inline function unroll_encoders(i, j, k, previous_size, size, encoder)
    #find index first grid to Nth grid
    ijk = (i, j, k)
    ijk  = div.(ijk, previous_size) 
    hijk = mod.(ijk, size) .+ 0x1
    return encoder(Val((hijk)))
end

function GilbertArray(FT, arch, sz)
    nested_sizes  = find_factorizations(sz)  
    nested_sizes  = Tuple((nx, ny, nz) for (nx, ny, nz) in zip(nested_sizes...))
    grid_sizes    = prod.(nested_sizes)
    cumulative_sizes  = [(1, 1, 1)]
    i = 1
    for n in nested_sizes
        push!(cumulative_sizes, cumulative_sizes[i] .* n)
        i+=1
    end
    underlying_data   = zeros(FT, arch, grid_sizes...)
    encoder_functions = Tuple(eval(Symbol(:encoded_index, curve-1)) for curve in 1:length(grid_sizes))
    return GilbertArray(underlying_data, nested_sizes, tuple(cumulative_sizes[1:end-1]...), tuple(cumulative_sizes[2:end]...), encoder_functions)
end
