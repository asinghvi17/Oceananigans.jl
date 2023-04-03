using Oceananigans
using Statistics
using Printf

#####
##### Define operations correpsonding to the u-, v-, and w-tendencies
#####

using Oceananigans.Models.NonhydrostaticModels: u_velocity_tendency, v_velocity_tendency, w_velocity_tendency
 
tendency_arguments(model) = (model.advection,
                             model.coriolis,
                             model.stokes_drift,
                             model.closure,
                             nothing, # immersed boundary condition goes here
                             model.buoyancy,
                             model.background_fields,
                             model.velocities,
                             model.tracers,
                             model.auxiliary_fields,
                             model.diffusivity_fields,
                             model.forcing)

@inline function Guᶠᶜᶜ(i, j, k, grid,
                       advection, coriolis, stokes_drift, closure, immersed_bc, buoyancy, background_fields,
                       velocities, tracers, auxiliary_fields, diffusivity_fields, forcing, pHY′, clock)

    return u_velocity_tendency(i, j, k, grid,
                               advection, coriolis, stokes_drift, closure, immersed_bc, buoyancy, background_fields,
                               velocities, tracers, auxiliary_fields, diffusivity_fields, forcing, pHY′, clock)
end

@inline function Gvᶜᶠᶜ(i, j, k, grid,
                       advection, coriolis, stokes_drift, closure, immersed_bc, buoyancy, background_fields,
                       velocities, tracers, auxiliary_fields, diffusivity_fields, forcing, pHY′, clock)

    return v_velocity_tendency(i, j, k, grid,
                               advection, coriolis, stokes_drift, closure, immersed_bc, buoyancy, background_fields,
                               velocities, tracers, auxiliary_fields, diffusivity_fields, forcing, pHY′, clock)
end

@inline function Gwᶜᶜᶠ(i, j, k, grid,
                       advection, coriolis, stokes_drift, closure, immersed_bc, buoyancy, background_fields,
                       velocities, tracers, auxiliary_fields, diffusivity_fields, forcing, clock)
                                       
    return w_velocity_tendency(i, j, k, grid,
                               advection, coriolis, stokes_drift, closure, immersed_bc, buoyancy, background_fields,
                               velocities, tracers, auxiliary_fields, diffusivity_fields, forcing, clock)
end

u_tendency_operation(model) = KernelFunctionOperation{Face, Center, Center}(Guᶠᶜᶜ, model.grid, tendency_arguments(model)..., model.pressures.pHY′, model.clock)
v_tendency_operation(model) = KernelFunctionOperation{Center, Face, Center}(Gvᶜᶠᶜ, model.grid, tendency_arguments(model)..., model.pressures.pHY′, model.clock)
w_tendency_operation(model) = KernelFunctionOperation{Center, Center, Face}(Gwᶜᶜᶠ, model.grid, tendency_arguments(model)..., model.clock)

grid = RectilinearGrid(CPU(), size=(128, 128, 1), x=(0, 2π), y=(0, 2π), z=(0, 1), topology=(Periodic, Periodic, Bounded))

function turbulent_decay(grid; Δt=1e-2)

    model = NonhydrostaticModel(; grid, advection=WENO())
    ϵ(x, y, z) = 2rand() - 1
    set!(model, u=ϵ, v=ϵ)

    simulation = Simulation(model; Δt, stop_time=1)

    Gu = u_tendency_operation(model)
    Gv = v_tendency_operation(model)
    Gw = w_tendency_operation(model)

    u, v, w = model.velocities

    # Previous velocities
    u⁻ = XFaceField(grid)
    v⁻ = YFaceField(grid)
    w⁻ = ZFaceField(grid)

    parent(u⁻) .= parent(u)
    parent(v⁻) .= parent(v)
    parent(w⁻) .= parent(w)

    #Gk = @at (Center, Center, Center) u * Gu + v * Gv + w * Gw
    # k = @at (Center, Center, Center) 1/2 * (u^2 + v^2 + w^2)

    #Gk = @at (Center, Center, Center) 1/2 * ((u + u⁻) * Gu + (v + v⁻) * Gv)
    Gk = @at (Center, Center, Center) u * Gu + v * Gv
     k = @at (Center, Center, Center) 1/2 * (u^2 + v^2)

    GK_t = []
    diagnosed_K_t  = Float64[]
    diagnosed_t    = Float64[]
    integrated_K_t = Float64[]
    integrated_t   = Float64[]

    # Add t = 0
    push!(integrated_K_t, mean(k))
    push!(integrated_t, time(simulation))

    function accumulate_timeseries(sim)
        push!(GK_t, mean(Gk))

        push!(diagnosed_K_t, mean(k))
        push!(diagnosed_t, time(sim))

        if length(integrated_K_t) > 0
            integrated_K = integrated_K_t[end] + sim.Δt * GK_t[end]
        else # first time step
            integrated_K = diagnosed_K_t[1] + sim.Δt * GK_t[1]
        end

        push!(integrated_K_t, integrated_K)
        push!(integrated_t, time(sim) + sim.Δt)

        return nothing
    end

    simulation.callbacks[:accumulate] = Callback(accumulate_timeseries)

    # Update velocities _after_ time-series is accumulated
    function update_previous_velocities(sim)
        u, v, w = sim.model.velocities

        if iteration(sim) > 0
            @assert u⁻ != u
            @assert v⁻ != v
            #@assert w⁻ != w
        end

        parent(u⁻) .= parent(u)
        parent(v⁻) .= parent(v)
        #parent(w⁻) .= parent(w)

        return nothing
    end

    # simulation.callbacks[:update] = Callback(update_previous_velocities)

    # Shenanigan for forcing an Euler step
    function force_euler_step(sim)
        sim.model.timestepper.previous_Δt = 0
    end

    simulation.callbacks[:force] = Callback(force_euler_step)

    function progress(sim)
        if iteration(sim) > 0
            @info @sprintf("Iter: %d, time: %.2f, K diagnosed: %.4e, K integrated: %.4e",
                           iteration(sim), time(sim), diagnosed_K_t[end], integrated_K_t[end])
        end
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))

    run!(simulation)

    return diagnosed_K_t, diagnosed_t, integrated_K_t, integrated_t
end

using GLMakie

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Time", ylabel="Volume-integrated kinetic energy, K")

dK, dt, iK, it = turbulent_decay(grid, Δt=2e-2)
lines!(ax, it, iK, label="Numerically integrated K, Δt=2e-2", linestyle=:dash)

dK, dt, iK, it = turbulent_decay(grid, Δt=1e-2)
lines!(ax, it, iK, label="Numerically integrated K, Δt=1e-2", linestyle=:dash)

dK, dt, iK, it = turbulent_decay(grid, Δt=5e-3)
lines!(ax, it, iK, label="Numerically integrated K, Δt=5e-3", linestyle=:dash)

dK, dt, iK, it = turbulent_decay(grid, Δt=2e-3)
lines!(ax, it, iK, label="Numerically integrated K, Δt=2e-3", linestyle=:dash)
lines!(ax, dt, dK, label="Diagnosed K, Δt=2e-3", linewidth=4, color=(:black, 0.6))

axislegend(ax)
display(fig)

