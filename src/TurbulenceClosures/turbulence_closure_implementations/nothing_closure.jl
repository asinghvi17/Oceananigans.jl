@inline ∇_dot_qᶜ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, args...) where FT = zero(FT)
@inline ∂ⱼ_τ₁ⱼ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, args...) where FT = zero(FT)
@inline ∂ⱼ_τ₂ⱼ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, args...) where FT = zero(FT)
@inline ∂ⱼ_τ₃ⱼ(i, j, k, grid::AbstractGrid{FT}, ::Nothing, args...) where FT = zero(FT)

calculate_diffusivities!(K, closure::Nothing, args...) = nothing
