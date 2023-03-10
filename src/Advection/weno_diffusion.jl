using Polynomials
using Oceananigans.Operators
using Oceananigans.Advection: left_stencil_x, 
                              left_biased_weno_weights,
                              coeff_left_p,
                              right_stencil_x, 
                              right_biased_weno_weights,
                              coeff_right_p,
                              VelocityStencil,
                              AbstractSmoothnessStencil,
                              AbstractUpwindBiasedAdvectionScheme

using KernelAbstractions.Extras.LoopInfo: @unroll

for (dir, val, cT) in zip((:x, :y, :z), (1, 2, 3), (:XT, :YT, :ZT))
    left_stencil_coeff  = Symbol(:left_stencil_coeff_, dir)
    right_stencil_coeff = Symbol(:right_stencil_coeff_, dir)
    left_stencil  = Symbol(:left_stencil_, dir)
    right_stencil = Symbol(:right_stencil_, dir)

    @eval begin
        @inline function $left_stencil_coeff(i, j, k, grid, scheme::WENO{N, FT, XT, YT, ZT}, 
                                     ψ, idx, loc, args...) where {N, FT, XT, YT, ZT}
            ψₜ = $left_stencil(i, j, k, scheme, ψ, grid, args...)
            weights = left_biased_weno_weights(ψₜ, scheme, Val($val), Nothing, args...)
            w = zeros(N^2) 
            st = 1
            @unroll for stencil in 1:N
                coeff = coeff_left_p(scheme, Val(stencil-1), $cT, Val($val), idx, loc)
                @unroll for val in 1:N
                    w[st] = weights[stencil] * coeff[N - val + 1]
                    st += 1
                end
            end

            return reverse((0.0, full_weno_stencil(w, scheme)...))
        end    

        @inline function $right_stencil_coeff(i, j, k, grid, scheme::WENO{N, FT, XT, YT, ZT}, 
                                      ψ, idx, loc, args...) where {N, FT, XT, YT, ZT}
            ψₜ = $right_stencil(i, j, k, scheme, c, grid)
            weights = right_biased_weno_weights(ψ, scheme, Val($val), Nothing, args...)
            w = zeros(N^2) 
            st = 1
            @unroll for stencil in 1:N
                coeff = coeff_right_p(scheme, Val(stencil-1), $cT, Val($val), idx, loc)
                @unroll for val in 1:N
                    w[st] = weights[stencil] * coeff[N - val + 1]
                    st += 1
                end
            end

            return reverse((full_weno_stencil(w, scheme)..., 0.0))
        end    

        @inline function $left_stencil_coeff(i, j, k, grid, scheme::WENO{N, FT, XT, YT, ZT}, 
                                     ψ, idx, loc, VI::VelocityStencil, args...) where {N, FT, XT, YT, ZT}
            weights = left_biased_weno_weights((i, j, k), scheme, Val($val), VI, args...)
            w = zeros(N^2) 
            st = 1
            @unroll for stencil in 1:N
                coeff = coeff_left_p(scheme, Val(stencil-1), $cT, Val($val), idx, loc)
                @unroll for val in 1:N
                    w[st] = weights[stencil] * coeff[N - val + 1]
                    st += 1
                end
            end

            return reverse((0.0, full_weno_stencil(w, scheme)...))
        end    

        @inline function $right_stencil_coeff(i, j, k, grid, scheme::WENO{N, FT, XT, YT, ZT}, 
                                      ψ, idx, loc, VI::VelocityStencil, args...) where {N, FT, XT, YT, ZT}
            weights = right_biased_weno_weights((i, j, k), scheme, Val($val), VI, args...)
            w = zeros(N^2) 
            st = 1
            @unroll for stencil in 1:N
                coeff = coeff_right_p(scheme, Val(stencil-1), $cT, Val($val), idx, loc)
                @unroll for val in 1:N
                    w[st] = weights[stencil] * coeff[N - val + 1]
                    st += 1
                end
            end

            return reverse((full_weno_stencil(w, scheme)..., 0.0))
        end    
    end
end

@inline full_weno_stencil(w, scheme::WENO{2}) = (w[1], 
                                                 w[2]+w[3], 
                                                 w[4])

@inline full_weno_stencil(w, scheme::WENO{3}) = (w[1], 
                                                 w[2]+w[3], 
                                                 w[4]+w[5]+w[6], 
                                                 w[7]+w[8], 
                                                 w[9])

@inline full_weno_stencil(w, scheme::WENO{4}) = (w[1], 
                                                 w[2]+w[3], 
                                                 w[4]+w[5]+w[6], 
                                                 w[7]+w[8]+w[9]+w[10], 
                                                 w[11]+w[12]+w[13], 
                                                 w[14]+w[15], 
                                                 w[16])

@inline full_weno_stencil(w, scheme::WENO{5}) = (w[1], 
                                                 w[2]+w[3], 
                                                 w[4]+w[5]+w[6], 
                                                 w[7]+w[8]+w[9]+w[10], 
                                                 w[11]+w[12]+w[13]+w[14]+w[15], 
                                                 w[16]+w[17]+w[18]+w[19], 
                                                 w[20]+w[21]+w[22], 
                                                 w[23]+w[24],
                                                 w[25])

@inline full_weno_stencil(w, scheme::WENO{6}) = (w[1], 
                                                 w[2]+w[3], 
                                                 w[4]+w[5]+w[6], 
                                                 w[7]+w[8]+w[9]+w[10], 
                                                 w[11]+w[12]+w[13]+w[14]+w[15], 
                                                 w[16]+w[17]+w[18]+w[19]+w[20]+w[21], 
                                                 w[22]+w[23]+w[24]+w[25]+w[26],
                                                 w[27]+w[28]+w[29]+w[30], 
                                                 w[31]+w[32]+w[33], 
                                                 w[34]+w[35], 
                                                 w[36])

const AUS{N} = AbstractUpwindBiasedAdvectionScheme{N} where N

@inline stencil_x(ψ, i, j, k, N, args...) = ψ[i-N:i+N, j, k]
@inline stencil_y(ψ, i, j, k, N, args...) = ψ[i, j-N:j+N, k]
@inline stencil_z(ψ, i, j, k, N, args...) = ψ[i, j, k-N:k+N]

@inline stencil_x(ψ::Function, i, j, k, N, args...) = ψ(i-N:i+N, j, k, args...)
@inline stencil_y(ψ::Function, i, j, k, N, args...) = ψ(i, j-N:j+N, k, args...)
@inline stencil_z(ψ::Function, i, j, k, N, args...) = ψ(i, j, k-N:k+N, args...)
                                                 
for (side, dir, 
     iᵟ⁺, jᵟ⁺, kᵟ⁺, 
     iᵟ⁻, jᵟ⁻, kᵟ⁻) in zip((:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ, :xᶜᵃᵃ, :yᵃᶜᵃ,   :zᵃᵃᶜ),
                           (:x,    :y,    :z,    :x,    :y,      :z),
                           (0,  0, 0, 1, 0, 0), (0,  0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1),
                           (-1, 0, 0, 0, 0, 0), (0, -1, 0, 0, 0, 0), (0, 0, 0, 0, 0, -1))

    upwind_diffusion = Symbol(:upwind_diffusion_, side)
    left_stencil_coeff = Symbol(:left_stencil_coeff_, dir)
    right_stencil_coeff = Symbol(:right_stencil_coeff_, dir)

    stencil = Symbol(:stencil_, dir)

    @eval begin
        @inline function $upwind_diffusion(i, j, k, grid, scheme::AUS{N}, u⁺, u⁻, ψ, args...) where N

            sᴸ⁻ =  $left_stencil_coeff(i+$iᵟ⁻, j+$jᵟ⁻, k+$kᵟ⁻, grid, scheme, ψ, 1, 1, args...)
            sᴿ⁻ = $right_stencil_coeff(i+$iᵟ⁻, j+$jᵟ⁻, k+$kᵟ⁻, grid, scheme, ψ, 1, 1, args...)
        
            sᴸ⁺ =  $left_stencil_coeff(i+$iᵟ⁺, j+$jᵟ⁺, k+$kᵟ⁺, grid, scheme, ψ, 1, 1, args...)
            sᴿ⁺ = $right_stencil_coeff(i+$iᵟ⁺, j+$jᵟ⁺, k+$kᵟ⁺, grid, scheme, ψ, 1, 1, args...)
        
            op = zeros(length(sᴸ⁻)+1)
        
            @unroll for t in eachindex(sᴸ⁻)
                op[t+1] += u⁺ * ifelse(u⁺ > 0, sᴸ⁺[t], sᴿ⁺[t])
                op[t]   -= u⁻ * ifelse(u⁻ > 0, sᴸ⁻[t], sᴿ⁻[t])
            end

            xs = (-N:N)
            p  = fit(xs, op)
        
            # pacoeffs = Tuple(ifelse(iseven(i), p.coeffs[i], 0) for i in 1:2N+1)
            pscoeffs = Tuple(ifelse(iseven(i), 0, p.coeffs[i]) for i in 1:2N+1)
        
            ps = Polynomial(pscoeffs)
        
            ψt = $stencil(ψ, i, j, k, N, grid, args...)

            return sum(ps.(xs) .* ψt)
        end
    end
end

@inline function tracer_weno_diffusion_x(i, j, k, grid, scheme, u, c)
    u⁺ = u[i+1, j, k]
    u⁻ = u[i, j, k]

    return upwind_diffusion_xᶜᵃᵃ(i, j, k, grid, scheme, u⁺, u⁻, c) * Axᶠᶜᶜ(i, j, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
end

@inline function tracer_weno_diffusion_y(i, j, k, grid, scheme, u, c)
    u⁺ = u[i, j+1, k]
    u⁻ = u[i, j,   k]

    return upwind_diffusion_yᵃᶜᵃ(i, j, k, grid, scheme, u⁺, u⁻, c) * Ayᶜᶠᶜ(i, j, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
end

@inline function tracer_weno_diffusion_z(i, j, k, grid, scheme, u, c)
    u⁺ = u[i, j, k+1]
    u⁻ = u[i, j, k]

    return upwind_diffusion_zᵃᵃᶜ(i, j, k, grid, scheme, u⁺, u⁻, c) * Ayᶜᶠᶜ(i, j, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
end

@inline function vector_invariant_vorticity_diffusion_U(i, j, k, grid, scheme::VectorInvariant{N}, u, v) where N
    Sζ = scheme.vorticity_stencil

    @inbounds v̂ = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, v) / Δxᶠᶜᶜ(i, j, k, grid) 

    sᴸ =  left_stencil_coeff_y(i, j+1, k, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, 1, 1, Sζ, u, v)
    sᴿ = right_stencil_coeff_y(i, j+1, k, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, 1, 1, Sζ, u, v)

    op = v̂ .* ifelse(v̂ > 0, sᴸ, sᴿ)

    xs = (-N+1:N)
    p  = fit(xs, op)

    pscoeffs = Tuple(ifelse(iseven(i), 0, p.coeffs[i]) for i in 1:length(p.coeffs))

    ps = Polynomial(pscoeffs)

    ψt = zeros(length(xs))

    st = 1
    for t in -N+1+j:j+N
        ψt[st] = ζ₃ᶠᶠᶜ(i, t, k, grid, u, v)
        st +=1
    end

    return sum(ps.(xs) .* ψt)
end

@inline function vector_invariant_vorticity_diffusion_V(i, j, k, grid, scheme::VectorInvariant{N}, u, v) where N
    Sζ = scheme.vorticity_stencil

    @inbounds û = ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, u) / Δyᶜᶠᶜ(i, j, k, grid)

    sᴸ =  left_stencil_coeff_x(i+1, j, k, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, 1, 1, Sζ, u, v)
    sᴿ = right_stencil_coeff_x(i+1, j, k, scheme.vorticity_scheme, ζ₃ᶠᶠᶜ, 1, 1, Sζ, u, v)

    op = û .* ifelse(û > 0, sᴸ, sᴿ)

    xs = (-N+1:N)
    p  = fit(xs, op)

    pscoeffs = Tuple(ifelse(iseven(i), 0, p.coeffs[i]) for i in 1:length(p.coeffs))

    ps = Polynomial(pscoeffs)

    ψt = zeros(length(xs))

    st = 1
    for t in -N+1+i:i+N
        ψt[st] = ζ₃ᶠᶠᶜ(t, j, k, grid, u, v)
        st +=1
    end

    return sum(ps.(xs) .* ψt)
end

# using Oceananigans.Fields: ConstantField

# function calc_tendency!(c, G)

#     u = ConstantField(-1.0)

#     for i in 1:size(c, 1)
#         G[i, 1, 1] = - tracer_weno_diffusion_x(i, 1, 1, grid, WENO(order = 5), u, c)
#     end

# end

# function iterate!(c, G, Δt)
#     fill_halo_regions!(c)
#     calc_tendency!(c, G)
#     c .+= Δt .* G
# end

