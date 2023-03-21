using GLMakie
using Oceananigans
using Oceananigans.Units
using Printf

using Oceananigans.TurbulenceClosures:
    RiBasedVerticalDiffusivity,
    CATKEVerticalDiffusivity,
    ConvectiveAdjustmentVerticalDiffusivity,
    ExplicitTimeDiscretization

#####
##### Setup simulation
#####

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=0.1, convective_νz=0.01)

grid = RectilinearGrid(size=64, z=(-256, 0), topology=(Flat, Flat, Bounded))
grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -250))

coriolis = FPlane(f=1e-4)

N² = 1e-5
Qᵇ = +1e-8
Qᵘ = -2e-4 #

b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))

closure = CATKEVerticalDiffusivity()

model = HydrostaticFreeSurfaceModel(; grid, closure, coriolis,
                                    tracers = (:b, :e),
                                    buoyancy = BuoyancyTracer(),
                                    boundary_conditions = (; b=b_bcs, u=u_bcs))
                                    
bᵢ(x, y, z) = N² * z
set!(model, b=bᵢ, e=1e-6)

simulation = Simulation(model, Δt=10minute, stop_time=5days)

diffusivities = (κᶜ = model.diffusivity_fields.κᶜ,
                 κᵉ = model.diffusivity_fields.κᵉ,
                 κᵘ = model.diffusivity_fields.κᵘ)

outputs = merge(model.velocities, model.tracers, diffusivities)

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(10minutes),
                                                      filename = "catke_windy_convection",
                                                      overwrite_existing = true)

progress(sim) = @info string("Iter: ", iteration(sim), " t: ", prettytime(sim))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

@info "Running a simulation of $model..."

run!(simulation)

#####
##### Visualize
#####

filepath = "catke_windy_convection.jld2"

bt = FieldTimeSeries(filepath, "b")
ut = FieldTimeSeries(filepath, "u")
vt = FieldTimeSeries(filepath, "v")
et = FieldTimeSeries(filepath, "e")
κᶜt = FieldTimeSeries(filepath, "κᶜ")
κᵉt = FieldTimeSeries(filepath, "κᵉ")
κᵘt = FieldTimeSeries(filepath, "κᵘ")

z = znodes(bt)
Nt = length(bt.times)

fig = Figure(resolution=(1600, 800))

slider = Slider(fig[2, 1:4], range=1:Nt, startvalue=1)
n = slider.value

buoyancy_label = @lift "Buoyancy at t = " * prettytime(bt.times[$n])
velocities_label = @lift "Velocities at t = " * prettytime(bt.times[$n])
TKE_label = @lift "Turbulent kinetic \n energy at t = " * prettytime(bt.times[$n])
diffusivities_label = @lift "Vertical eddy \n diffusivities at t = " * prettytime(bt.times[$n])

ax_b = Axis(fig[1, 1], xlabel=buoyancy_label, ylabel="z (m)")
ax_u = Axis(fig[1, 2], xlabel=velocities_label, ylabel="z (m)")
ax_e = Axis(fig[1, 3], xlabel=TKE_label, ylabel="z (m)")
ax_κ = Axis(fig[1, 4], xlabel=diffusivities_label, ylabel="z (m)")

xlims!(ax_b, -grid.Lz * N², 0)
xlims!(ax_u, -0.1, 0.1)
xlims!(ax_e, -1e-4, 2e-3)
xlims!(ax_κ, -0.01, 6.0)

colors = [:black, :blue, :red, :orange]

bn = @lift interior(bt[$n], 1, 1, :)
un = @lift interior(ut[$n], 1, 1, :)
vn = @lift interior(vt[$n], 1, 1, :)
en = @lift interior(et[$n], 1, 1, :)
κᶜn = @lift interior(κᶜt[$n], 1, 1, :)
κᵉn = @lift interior(κᵉt[$n], 1, 1, :)
κᵘn = @lift interior(κᵘt[$n], 1, 1, :)

lines!(ax_b, bn, z)
lines!(ax_u, un, z, label="u")
lines!(ax_u, vn, z, label="v", linestyle=:dash)
lines!(ax_e, en, z)

lines!(ax_κ, κᶜn, z, label="κᶜ")
lines!(ax_κ, κᵉn, z, label="κᵉ")
lines!(ax_κ, κᵘn, z, label="κᵘ")

axislegend(ax_u, position=:rb)
axislegend(ax_κ, position=:rb)

display(fig)

# record(fig, "windy_convection.mp4", 1:Nt, framerate=24) do nn
#     n[] = nn
# end

