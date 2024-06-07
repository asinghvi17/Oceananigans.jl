using Oceananigans
using Statistics: std
using Printf
using GLMakie

grid = RectilinearGrid(topology = (Bounded, Periodic, Bounded),
                       size = (16, 20, 4), extent = (800, 1000, 100))

@inline east_wall(x, y, z) = x > 400
grid = ImmersedBoundaryGrid(grid, GridFittedBoundary(east_wall))
    
model = NonhydrostaticModel(; grid, timestepper=:RungeKutta3, buoyancy = BuoyancyTracer(), tracers = :b)

N² = 6e-6
b∞(x, y, z) = N² * z
set!(model, b=b∞)
    
#simulation = Simulation(model, Δt=π, stop_time=1e3)
simulation = Simulation(model, Δt=25, stop_time=1e4)

previous_time = Ref(time(simulation))

function progress_message(sim)
    u, v, w = sim.model.velocities
    
    @printf("Iter: % 4d, time: % 24s, Δt: %s, iteration × Δt: %s, std(pNHS) = %.2e",
            iteration(sim),
            time(sim), time(sim) - previous_time[],
            iteration(sim) * sim.Δt,
            std(model.pressures.pNHS))

    @printf(", max|u|: (%.3e, %.3e, %.3e) ",
            maximum(abs, u),
            maximum(abs, v),
            maximum(abs, w))

    println()


    previous_time[] = time(sim)
    return nothing
end

add_callback!(simulation, progress_message, IterationInterval(1))

pstds = []
capture_pressure(sim) = push!(pstds, std(model.pressures.pNHS))
add_callback!(simulation, capture_pressure, TimeInterval(100))

run!(simulation)

u, v, w = model.velocities
uxzn = interior(u, :, 1, :)
vxzn = interior(u, :, 1, :)
wxzn = interior(u, :, 1, :)

fig = Figure()
axuxz = Axis(fig[1, 1])
axwxz = Axis(fig[1, 2])

heatmap!(axuxz, uxzn)
heatmap!(axwxz, wxzn)

display(fig)


#=
using Oceananigans

Ns = 400    # number of time saves
T = 8e2*π/7 # simulation stop time (s)
Δt = 16/15  # timestep (s)
Nt = T/Δt

grid = RectilinearGrid(size = (), topology=(Flat, Flat, Flat))
model = NonhydrostaticModel(; grid, timestepper=:RungeKutta3)
simulation = Simulation(model; Δt, stop_time=T)

captured_times = []
function capture_time(sim)
    @info iteration(sim)
    push!(captured_times, time(sim))
end

callback = Callback(capture_time, TimeInterval(T/Ns))
add_callback!(simulation, callback)

run!(simulation)

@show time(simulation) iteration(simulation)
@show length(captured_times)
@show time(simulation) == T
=#
