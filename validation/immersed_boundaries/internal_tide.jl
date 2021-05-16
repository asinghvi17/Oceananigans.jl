using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Plots

grid = RegularRectilinearGrid(size=(1024, 1024), x=(-10, 10), z=(0, 5), topology=(Periodic, Flat, Bounded))

# Gaussian bump of width "1"
bump(x, y, z) = z < exp(-x^2)

grid_with_bump = ImmersedBoundaryGrid(grid, GridFittedBoundary(bump))

# Tidal forcing
tidal_forcing(x, y, z, t) = 1e-4 * cos(t)

model = HydrostaticFreeSurfaceModel(architecture = GPU(),
                                    grid = grid_with_bump,
                                    momentum_advection = CenteredSecondOrder(),
                                    free_surface = ExplicitFreeSurface(gravitational_acceleration=10),
                                    closure = IsotropicDiffusivity(ν=1e-4, κ=1e-4),
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = FPlane(f=sqrt(0.5)),
                                    forcing = (u = tidal_forcing,))

# Linear stratification
set!(model, b = (x, y, z) -> 10 * z)

progress(s) = @info @sprintf("Progress: %.2f \%, max|w|: %.2e",
                             s.model.clock.time / s.stop_time, maximum(abs, model.velocities.w))

gravity_wave_speed = sqrt(model.free_surface.gravitational_acceleration * grid.Lz)
Δt = 0.2 * grid.Δx / gravity_wave_speed
              
simulation = Simulation(model, Δt = Δt, stop_time = 10, progress = progress, iteration_interval = 100)

serialize_grid(file, model) = file["serialized/grid"] = model.grid.grid

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                      schedule = TimeInterval(0.02),
                                                      prefix = "internal_tide",
                                                      init = serialize_grid,
                                                      force = true)
                        
run!(simulation)

using JLD2

function nice_divergent_levels(c, clim; nlevels=20)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
    return (-clim, clim), levels
end

function nan_solid(x, z, u, bump)
    Nx, Nz = size(u)
    x2 = reshape(x, Nx, 1)
    z2 = reshape(z, 1, Nz)
    u[bump.(x2, 0, z2)] .= NaN
    return nothing
end

function visualize_internal_tide(prefix)

    filename = prefix * ".jld2"
    file = jldopen(filename)

    grid = file["serialized/grid"]

    bump(x, y, z) = z < exp(-x^2)

    xu, yu, zu = nodes((Face, Center, Center), grid)
    xw, yw, zw = nodes((Center, Center, Face), grid)

    iterations = parse.(Int, keys(file["timeseries/t"]))    

    anim = @animate for (i, iter) in enumerate(iterations)

        u = file["timeseries/u/$iter"][:, 1, :]
        w = file["timeseries/w/$iter"][:, 1, :]

        wlims, wlevels = nice_divergent_levels(w, 1e-3)
        ulims, ulevels = nice_divergent_levels(u, 1e-3)
        
        nan_solid(xu, zu, u, bump) 
        nan_solid(xw, zw, w, bump) 

        u_plot = contourf(xu, zu, u'; title = "x velocity", color = :balance, linewidth = 0, levels = ulevels, clims = (-ulim, ulim))
        w_plot = contourf(xw, zw, w'; title = "z velocity", color = :balance, linewidth = 0, levels = wlevels, clims = (-wlim, wlim))

        plot(u_plot, w_plot, layout = (2, 1))
    end

    mp4(anim, "internal_tide.mp4", fps = 8)

    close(file)
end
