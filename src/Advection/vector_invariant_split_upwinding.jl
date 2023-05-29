const VectorInvariantSplitVerticalUpwinding  = VectorInvariant{<:Any, <:Any, <:Any, <:Any, <:AbstractUpwindBiasedAdvectionScheme, <:SplitUpwinding}

#####
##### Split upwinding is a Partial Upwinding where the upwind choice occurrs _inside_
##### the difference operator instead of outside. _MOST_ stable formulation, low kinetic energy
##### 

##### 
##### Split Upwinding of Divergence flux
#####

@inline function Auᶜᶜᶜ(i, j, k, grid, scheme, u) 
    û = ℑxᶜᵃᵃ(i, j, k, grid, u)

    Uᴸ =  _left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, Ax_qᶠᶜᶜ, u)
    Uᴿ = _right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, Ax_qᶠᶜᶜ, u)

    return ifelse(û > 0, Uᴸ, Uᴿ)
end

@inline function Avᶜᶜᶜ(i, j, k, grid, scheme, v) 
    v̂ = ℑyᵃᶜᵃ(i, j, k, grid, v)

    Vᴸ =  _left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, Ay_qᶜᶠᶜ, v)
    Vᴿ = _right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, Ay_qᶜᶠᶜ, v)

    return ifelse(v̂ > 0, Vᴸ, Vᴿ)
end

@inline Auᶠᶠᶜ(i, j, k, grid, scheme, u) = 
     _symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, Ax_qᶠᶜᶜ, u)

@inline Avᶠᶠᶜ(i, j, k, grid, scheme, v) = 
     _symmetric_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, Ay_qᶜᶠᶜ, v)

@inline function upwind_divergence_flux_Uᶠᶜᶜ(i, j, k, grid, scheme::VectorInvariantSplitVerticalUpwinding, u, v) 
    @inbounds û = u[i, j, k] 
    
    δu = δxᶠᵃᵃ(i, j, k, grid, Auᶜᶜᶜ, scheme, u) 
    δv = δyᵃᶜᵃ(i, j, k, grid, Avᶠᶠᶜ, scheme, v)

    return û * (δu + δv)
end

@inline function upwind_divergence_flux_Vᶜᶠᶜ(i, j, k, grid, scheme::VectorInvariantSplitVerticalUpwinding, u, v) 
    @inbounds v̂ = v[i, j, k] 

    δu = δxᶜᵃᵃ(i, j, k, grid, Auᶠᶠᶜ, scheme, u) 
    δv = δyᵃᶠᵃ(i, j, k, grid, Avᶜᶜᶜ, scheme, v)

    return v̂ * (δu + δv)
end

##### 
##### Split Upwinding of Kinetic Energy gradient
#####

@inline function uᵁ²ᶜᶜᶜ(i, j, k, grid, scheme, u) 
    û = ℑxᶜᵃᵃ(i, j, k, grid, u)

    Uᴸ =  _left_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, half_ϕ², u)
    Uᴿ = _right_biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, half_ϕ², u)

    return ifelse(û > 0, Uᴸ, Uᴿ)
end

@inline function vᵁ²ᶜᶜᶜ(i, j, k, grid, scheme, v) 
    v̂ = ℑyᵃᶜᵃ(i, j, k, grid, v)

    Vᴸ =  _left_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, half_ϕ², v)
    Vᴿ = _right_biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, half_ϕ², v)

    return ifelse(v̂ > 0, Vᴸ, Vᴿ)
end

@inline uˢ²ᶜᶜᶜ(i, j, k, grid, scheme, u) =
     _symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, half_ϕ², u)

@inline vˢ²ᶜᶜᶜ(i, j, k, grid, scheme, v) = 
     _symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, scheme.vertical_scheme, half_ϕ², v)

@inline function bernoulli_head_U(i, j, k, grid, scheme::VectorInvariantSplitVerticalUpwinding, u, v)

    δKu = δxᶠᵃᵃ(i, j, k, grid, uᵁ²ᶜᶜᶜ, scheme, u)
    δKv = δxᶠᵃᵃ(i, j, k, grid, vˢ²ᶜᶜᶜ, scheme, v)

    return (δKu + δKv) / Δxᶠᶜᶜ(i, j, k, grid)
end

@inline function bernoulli_head_V(i, j, k, grid, scheme::VectorInvariantSplitVerticalUpwinding, u, v)

    δKu = δyᵃᶠᵃ(i, j, k, grid, uˢ²ᶜᶜᶜ, scheme, u)
    δKv = δyᵃᶠᵃ(i, j, k, grid, vᵁ²ᶜᶜᶜ, scheme, v)

    return (δKu + δKv) / Δyᶜᶠᶜ(i, j, k, grid)
end