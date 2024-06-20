using Oceananigans.TurbulenceClosures: AnisotropicMinimumDissipation
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.DistributedComputations: cpu_architecture, partition_global_array

function run_ocean_large_eddy_simulation_regression_test(arch, grid_type, closure)
    name = "ocean_large_eddy_simulation_" * string(typeof(first(closure)).name.wrapper)

    spinup_steps = 10000
      test_steps = 10
              Δt = 2.0

    # Parameters
      Qᵀ = 5e-5     # Temperature flux at surface
      Qᵘ = -2e-5    # Velocity flux at surface
    ∂T∂z = 0.005    # Initial vertical temperature gradient

    # Grid
    N = L = 16
    if grid_type == :regular
        grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L), halo=(2, 2, 2))
    elseif grid_type == :vertically_unstretched
        zF = range(-L, 0, length=N+1)
        grid = RectilinearGrid(arch, size=(N, N, N), x=(0, L), y=(0, L), z=zF, halo=(2, 2, 2))
    end

    # Boundary conditions
    u_bcs = FieldBoundaryConditions(top = BoundaryCondition(Flux, Qᵘ))
    T_bcs = FieldBoundaryConditions(top = BoundaryCondition(Flux, Qᵀ), bottom = BoundaryCondition(Gradient, ∂T∂z))
    S_bcs = FieldBoundaryConditions(top = BoundaryCondition(Flux, 5e-8))

    equation_of_state = LinearEquationOfState(thermal_expansion=2e-4, haline_contraction=8e-4)

    # Model instantiation
    model = NonhydrostaticModel(; grid, closure,
                                coriolis = FPlane(f=1e-4),
                                buoyancy = SeawaterBuoyancy(; equation_of_state),
                                tracers = (:T, :S),
                                hydrostatic_pressure_anomaly = CenterField(grid),
                                boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs))

    # The type of the underlying data, not the offset array.
    ArrayType = typeof(model.velocities.u.data.parent)
    nx, ny, nz = size(model.tracers.T)

    u, v, w = model.velocities
    T, S = model.tracers

    ####
    #### Uncomment the block below to generate regression data.
    ####

    #=
    @warn "Generating new data for the ocean LES regression test."

    # Initialize model: random noise damped at top and bottom
    Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise
    T₀(x, y, z) = 20 + ∂T∂z * z + ∂T∂z * model.grid.Lz * 1e-2 * Ξ(z)
    u₀(x, y, z) = sqrt(abs(Qᵘ)) * 1e-3 * Ξ(z)
    set!(model, u=u₀, w=u₀, T=T₀, S=35)

    simulation.stop_iteration = spinup_steps-test_steps
    run!(simulation)

    checkpointer = Checkpointer(model, schedule = IterationInterval(test_steps), prefix = name,
                                dir = joinpath(dirname(@__FILE__), "data"))

    simulation.output_writers[:checkpointer] = checkpointer

    simulation.stop_iteration += 2test_steps
    run!(simulation)
    pop!(simulation.output_writers, :checkpointer)
    =#

    ####
    #### Regression test
    ####

    datadep_path = "regression_test_data/" * name * "_iteration$spinup_steps.jld2"
    initial_filename = @datadep_str datadep_path

    solution₀, Gⁿ₀, G⁻₀ = get_fields_from_checkpoint(initial_filename)

    Nz = grid.Nz

    solution_indices   = [2:nx+3, 2:ny+3, 2:nz+3]
    w_solution_indices = [2:nx+3, 2:ny+3, 2:nz+4]

    cpu_arch = cpu_architecture(architecture(grid))

    u₀ = partition_global_array(cpu_arch, ArrayType(solution₀.u), size(u))
    v₀ = partition_global_array(cpu_arch, ArrayType(solution₀.v), size(v))
    w₀ = partition_global_array(cpu_arch, ArrayType(solution₀.w), size(w))
    T₀ = partition_global_array(cpu_arch, ArrayType(solution₀.T), size(T))
    S₀ = partition_global_array(cpu_arch, ArrayType(solution₀.S), size(S))

    Gⁿu₀ = partition_global_array(cpu_arch, ArrayType(Gⁿ₀.u), size(u))
    Gⁿv₀ = partition_global_array(cpu_arch, ArrayType(Gⁿ₀.v), size(v))
    Gⁿw₀ = partition_global_array(cpu_arch, ArrayType(Gⁿ₀.w), size(w))
    GⁿT₀ = partition_global_array(cpu_arch, ArrayType(Gⁿ₀.T), size(T))
    GⁿS₀ = partition_global_array(cpu_arch, ArrayType(Gⁿ₀.S), size(S))

    G⁻u₀ = partition_global_array(cpu_arch, ArrayType(G⁻₀.u), size(u))
    G⁻v₀ = partition_global_array(cpu_arch, ArrayType(G⁻₀.v), size(v))
    G⁻w₀ = partition_global_array(cpu_arch, ArrayType(G⁻₀.w), size(w))
    G⁻T₀ = partition_global_array(cpu_arch, ArrayType(G⁻₀.T), size(T))
    G⁻S₀ = partition_global_array(cpu_arch, ArrayType(G⁻₀.S), size(S))

    interior(model.velocities.u) .= u₀
    interior(model.velocities.v) .= v₀
    interior(model.velocities.w) .= w₀
    interior(model.tracers.T)    .= T₀
    interior(model.tracers.S)    .= S₀

    interior(model.timestepper.Gⁿ.u) .= Gⁿu₀
    interior(model.timestepper.Gⁿ.v) .= Gⁿv₀
    interior(model.timestepper.Gⁿ.w) .= Gⁿw₀
    interior(model.timestepper.Gⁿ.T) .= GⁿT₀
    interior(model.timestepper.Gⁿ.S) .= GⁿS₀

    interior(model.timestepper.G⁻.u) .= G⁻u₀
    interior(model.timestepper.G⁻.v) .= G⁻v₀
    interior(model.timestepper.G⁻.w) .= G⁻w₀
    interior(model.timestepper.G⁻.T) .= G⁻T₀
    interior(model.timestepper.G⁻.S) .= G⁻S₀

    model.clock.time = spinup_steps * Δt
    model.clock.iteration = spinup_steps

    update_state!(model; compute_tendencies = true)
    model.clock.last_Δt = Δt

    for n in 1:test_steps
        time_step!(model, Δt, euler=false)
    end

    datadep_path = "regression_test_data/" * name * "_iteration$(spinup_steps+test_steps).jld2"
    final_filename = @datadep_str datadep_path

    solution₁, Gⁿ₁, G⁻₁ = get_fields_from_checkpoint(final_filename)

    test_fields = CUDA.@allowscalar (u = Array(interior(model.velocities.u)),
                                     v = Array(interior(model.velocities.v)),
                                     w = Array(interior(model.velocities.w)[:, :, 1:nz]),
                                     T = Array(interior(model.tracers.T)),
                                     S = Array(interior(model.tracers.S)))

    u₁ = partition_global_array(cpu_arch, Array(solution₁.u), size(u))
    v₁ = partition_global_array(cpu_arch, Array(solution₁.v), size(v))
    w₁ = partition_global_array(cpu_arch, Array(solution₁.w), size(test_fields.w))
    T₁ = partition_global_array(cpu_arch, Array(solution₁.T), size(T))
    S₁ = partition_global_array(cpu_arch, Array(solution₁.S), size(S))

    correct_fields = (u = u₁[2:nx+1, 2:ny+1, 2:nz+1],
                      v = v₁[2:nx+1, 2:ny+1, 2:nz+1],
                      w = w₁[2:nx+1, 2:ny+1, 2:nz+1],
                      T = T₁[2:nx+1, 2:ny+1, 2:nz+1],
                      S = S₁[2:nx+1, 2:ny+1, 2:nz+1])

    summarize_regression_test(test_fields, correct_fields)

    @test all(test_fields.u .≈ correct_fields.u)
    @test all(test_fields.v .≈ correct_fields.v)
    @test all(test_fields.w .≈ correct_fields.w)
    @test all(test_fields.T .≈ correct_fields.T)
    @test all(test_fields.S .≈ correct_fields.S)

    return nothing
end
