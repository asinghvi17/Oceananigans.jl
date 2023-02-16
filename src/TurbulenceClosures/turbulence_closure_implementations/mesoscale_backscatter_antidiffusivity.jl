using Oceananigans.Operators
using Oceananigans.Advection: AbstractAdvectionScheme

struct MesoscaleBackscatterAntiDiffusivity{B, DA, CA, DC} <: AbstractScalarDiffusivity{ExplicitTimeDiscretization, HorizontalFormulation}
    backscattered_fraction :: B
    diff_adv :: DA
    cons_adv :: CA
    dissipation_closure :: DC
end

const  MBAD = MesoscaleBackscatterAntiDiffusivity
const IMBAD = MesoscaleBackscatterAntiDiffusivity{<:Any, <:AbstractAdvectionScheme}
const EMBAD = MesoscaleBackscatterAntiDiffusivity{<:Any, Nothing}

DiffusivityFields(grid, tracer_names, bcs, ::MBAD) =  (; νₑ = CenterField(grid))

function calculate_diffusivities!(diffusivity_fields, closure::MBAD, model)
    arch     = model.architecture
    grid     = model.grid
    tracers  = model.tracers

    event = launch!(arch, grid, :xyz,
                    calculate_viscosities!,
                    diffusivity_fields, grid, closure, tracers,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

@inline viscosity(::MesoscaleBackscatterAntiDiffusivity, K) = K.νₑ

@inline two_pass_filter_ccc(i, j, k, grid, f, args...) = ℑxyᶜᶜᵃ(i, j, k, grid, ℑxyᶠᶠᵃ, f, args...)
@inline two_pass_filter_ffc(i, j, k, grid, f, args...) = ℑxyᶠᶠᵃ(i, j, k, grid, ℑxyᶜᶜᵃ, f, args...)

@inline two_pass_filter_fcc(i, j, k, grid, f, args...) = ℑxyᶠᶜᵃ(i, j, k, grid, ℑxyᶜᶠᵃ, f, args...)
@inline two_pass_filter_cfc(i, j, k, grid, f, args...) = ℑxyᶜᶠᵃ(i, j, k, grid, ℑxyᶠᶜᵃ, f, args...)

@inline viscous_flux_ux(i, j, k, grid, clo::MBAD, K, clk, fields, b) = - two_pass_filter_ccc(i, j, k, grid, ν_σᶜᶜᶜ, clo, K, clk, fields, ∂xᶜᶜᶜ, fields.u)
@inline viscous_flux_uy(i, j, k, grid, clo::MBAD, K, clk, fields, b) = - two_pass_filter_ffc(i, j, k, grid, ν_σᶠᶠᶜ, clo, K, clk, fields, ∂yᶠᶠᶜ, fields.u)
@inline viscous_flux_vx(i, j, k, grid, clo::MBAD, K, clk, fields, b) = - two_pass_filter_ffc(i, j, k, grid, ν_σᶠᶠᶜ, clo, K, clk, fields, ∂xᶠᶠᶜ, fields.v)
@inline viscous_flux_vy(i, j, k, grid, clo::MBAD, K, clk, fields, b) = - two_pass_filter_ccc(i, j, k, grid, ν_σᶜᶜᶜ, clo, K, clk, fields, ∂yᶜᶜᶜ, fields.v)

@inline diffusive_flux_x(i, j, k, grid, cl::MBAD, args...) = zero(grid)
@inline diffusive_flux_y(i, j, k, grid, cl::MBAD, args...) = zero(grid)
@inline diffusive_flux_z(i, j, k, grid, cl::MBAD, args...) = zero(grid)

@inline backscattered_energy(i, j, k, grid, args...) = zero(grid)

@inline backscattered_energy(i, j, k, grid, cl::MBAD, args...) = - (ℑxᶜᵃᵃ(i, j, k, grid, explicit_backscatter_U, cl, args...) +
                                                                    ℑyᵃᶜᵃ(i, j, k, grid, explicit_backscatter_V, cl, args...))

@inline U_times_∂τ(i, j, k, grid, U, ∂τ, args...) = U[i, j, k] * ∂τ(i, j, k, grid, args...)

@inline explicit_backscatter_U(i, j, k, grid, cl, K, clock, U, b) = two_pass_filter_fcc(i, j, k, grid, U_times_∂τ, U.u, ∂ⱼ_τ₁ⱼ, cl, K, clock, U, b)
@inline explicit_backscatter_V(i, j, k, grid, cl, K, clock, U, b) = two_pass_filter_cfc(i, j, k, grid, U_times_∂τ, U.v, ∂ⱼ_τ₂ⱼ, cl, K, clock, U, b)

@kernel function calculate_viscosities!(diffusivity, grid, cl, tracers)
    i, j, k = @index(Global, NTuple)

    visc = sqrt(Azᶜᶜᶜ(i, j, k, grid) * max(tracers.E[i, j, k], 0.0))
    diffusivity.νₑ[i, j, k] = - cl.backscattered_fraction * visc
end

using Oceananigans.Advection: div_Uc, U_dot_∇u, U_dot_∇v

<<<<<<< HEAD
# rewrite fluxes: (u⋅∇)F = ∇(u⋅F) + (F⋅∇)u
# dissipation is the second term on the RHS (where F is ν∇u)
=======
@inline implicit_dissipation_U(i, j, k, grid, cl, U) = U.u[i, j, k] * (U_dot_∇u(i, j, k, grid, cl.diffusing_advection, U) - 
                                                                       U_dot_∇u(i, j, k, grid, cl.conserving_advection, U))

@inline implicit_dissipation_V(i, j, k, grid, cl, U) = U.v[i, j, k] * (U_dot_∇v(i, j, k, grid, cl.diffusing_advection, U) - 
                                                                       U_dot_∇v(i, j, k, grid, cl.conserving_advection, U))

@inline implicit_dissipation(i, j, k, grid, cl::MBAD, U) = ℑxᶜᵃᵃ(i, j, k, grid, implicit_dissipation_U, cl, U) +
                                                           ℑyᵃᶜᵃ(i, j, k, grid, implicit_dissipation_V, cl, U)
>>>>>>> 3e6807c97a9f3c0be324bd25e06305cc1f271129

@inline ϕ²(i, j, k, grid, f, args...) = f(i, j, k, grid, args...)^2

@inline explicit_dissipation_U(i, j, k, grid, cl, K, clk, U, b) = ν_σᶜᶜᶜ(i, j, k, grid, cl.dissipation_closure, K, clk, U, ϕ², ∂xᶜᶜᶜ, U.u) +
                                                                  ℑxyᶜᶜᵃ(i, j, k, grid, ν_σᶠᶠᶜ, cl.dissipation_closure, K, clk, U, ϕ², ∂yᶠᶠᶜ, U.u)       
@inline explicit_dissipation_V(i, j, k, grid, cl, K, clk, U, b) = ν_σᶜᶜᶜ(i, j, k, grid, cl.dissipation_closure, K, clk, U, ϕ², ∂yᶜᶜᶜ, U.v) +
                                                                  ℑxyᶜᶜᵃ(i, j, k, grid, ν_σᶠᶠᶜ, cl.dissipation_closure, K, clk, U, ϕ², ∂xᶠᶠᶜ, U.v) 

@inline adv_diff_U(i, j, k, grid, adv1, adv2, U) = 2 * U.u[i, j, k] * (U_dot_∇u(i, j, k, grid, adv1, U) - U_dot_∇u(i, j, k, grid, adv2, U))
@inline adv_diff_V(i, j, k, grid, adv1, adv2, U) = 2 * U.v[i, j, k] * (U_dot_∇v(i, j, k, grid, adv1, U) - U_dot_∇v(i, j, k, grid, adv2, U))

@inline implicit_dissipation_U(i, j, k, grid, cl, U) = two_pass_filter_fcc(i, j, k, grid, adv_diff_U, cl.diff_adv, cl.cons_adv, U)
@inline implicit_dissipation_V(i, j, k, grid, cl, U) = two_pass_filter_cfc(i, j, k, grid, adv_diff_V, cl.diff_adv, cl.cons_adv, U)

@inline implicit_dissipation(i, j, k, grid, args...) = zero(grid)
@inline explicit_dissipation(i, j, k, grid, args...) = zero(grid)

@inline implicit_dissipation(i, j, k, grid, cl::IMBAD, args...) = ℑxᶜᵃᵃ(i, j, k, grid, implicit_dissipation_U, cl, args...) +
                                                                  ℑyᵃᶜᵃ(i, j, k, grid, implicit_dissipation_V, cl, args...)

@inline explicit_dissipation(i, j, k, grid, cl::EMBAD, args...) = explicit_dissipation_U(i, j, k, grid, cl, args...) +
                                                                  explicit_dissipation_V(i, j, k, grid, cl, args...)

@inline function hydrostatic_subgrid_kinetic_energy_tendency(i, j, k, grid,
                                                             val_tracer_index::Val{tracer_index},
                                                             advection,
                                                             closure,
                                                             e_immersed_bc,
                                                             buoyancy,
                                                             backgound_fields,
                                                             velocities,
                                                             tracers,
                                                             auxiliary_fields,
                                                             diffusivities,
                                                             forcing,
                                                             clock) where tracer_index
         
    @inbounds E = tracers[tracer_index]

    model_fields = merge(velocities, tracers, auxiliary_fields)

    return (- div_Uc(i, j, k, grid, advection, velocities, E)
            + implicit_dissipation(i, j, k, grid, closure, model_fields)
            + explicit_dissipation(i, j, k, grid, closure, diffusivities, clock, model_fields, buoyancy)
            - backscattered_energy(i, j, k, grid, closure, diffusivities, clock, model_fields, buoyancy))
end      


            
for dissipation in [:implicit_dissipation, :explicit_dissipation]
      @eval begin
            @inline $dissipation(i, j, k, grid, closures::Tuple{<:Any}, args...) =
                  $dissipation(i, j, k, grid, closures[1], args...)

            @inline $dissipation(i, j, k, grid, closures::Tuple{<:Any, <:Any}, args...) = (
                  $dissipation(i, j, k, grid, closures[1], args...)
                  + $dissipation(i, j, k, grid, closures[2], args...))
                  
            @inline $dissipation(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any}, args...) = (
                  $dissipation(i, j, k, grid, closures[1], args...)
                  + $dissipation(i, j, k, grid, closures[2], args...) 
                  + $dissipation(i, j, k, grid, closures[3], args...))
                  
            @inline $dissipation(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any, <:Any}, args...) = (
                  $dissipation(i, j, k, grid, closures[1], args...)
                  + $dissipation(i, j, k, grid, closures[2], args...) 
                  + $dissipation(i, j, k, grid, closures[3], args...) 
                  + $dissipation(i, j, k, grid, closures[4], args...))
                        
            @inline $dissipation(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any, <:Any, <:Any}, args...) = (
                  $dissipation(i, j, k, grid, closures[1], args...)
                  + $dissipation(i, j, k, grid, closures[2], args...) 
                  + $dissipation(i, j, k, grid, closures[3], args...) 
                  + $dissipation(i, j, k, grid, closures[4], args...)
                  + $dissipation(i, j, k, grid, closures[5], args...))
                  
            @inline $dissipation(i, j, k, grid, closures::Tuple, args...) = (
                  $dissipation(i, j, k, grid, closures[1], args...)
                  + $dissipation(i, j, k, grid, closures[2:end], args...))
      end
end
@inline backscattered_energy(i, j, k, grid, closures::Tuple{<:Any}, Ks, args...) =
      backscattered_energy(i, j, k, grid, closures[1], Ks[1], args...)

@inline backscattered_energy(i, j, k, grid, closures::Tuple{<:Any, <:Any}, Ks, args...) = (
      backscattered_energy(i, j, k, grid, closures[1], Ks[1], args...)
    + backscattered_energy(i, j, k, grid, closures[2], Ks[2], args...))

@inline backscattered_energy(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any}, Ks, args...) = (
      backscattered_energy(i, j, k, grid, closures[1], Ks[1], args...)
    + backscattered_energy(i, j, k, grid, closures[2], Ks[2], args...) 
    + backscattered_energy(i, j, k, grid, closures[3], Ks[3], args...))

@inline backscattered_energy(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any, <:Any}, Ks, args...) = (
      backscattered_energy(i, j, k, grid, closures[1], Ks[1], args...)
    + backscattered_energy(i, j, k, grid, closures[2], Ks[2], args...) 
    + backscattered_energy(i, j, k, grid, closures[3], Ks[3], args...) 
    + backscattered_energy(i, j, k, grid, closures[4], Ks[4], args...))

@inline backscattered_energy(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any, <:Any, <:Any}, Ks, args...) = (
      backscattered_energy(i, j, k, grid, closures[1], Ks[1], args...)
    + backscattered_energy(i, j, k, grid, closures[2], Ks[2], args...) 
    + backscattered_energy(i, j, k, grid, closures[3], Ks[3], args...) 
    + backscattered_energy(i, j, k, grid, closures[4], Ks[4], args...)
    + backscattered_energy(i, j, k, grid, closures[5], Ks[5], args...))

@inline backscattered_energy(i, j, k, grid, closures::Tuple, Ks, args...) = (
      backscattered_energy(i, j, k, grid, closures[1], Ks[1], args...)
    + backscattered_energy(i, j, k, grid, closures[2:end], Ks[2:end], args...))