
using Oceananigans
using Oceananigans.Grids: min_Δx
using Oceananigans.Advection: WENO5S
using JLD2
using Plots
using BenchmarkTools

""" cosine grid """

function test_L2norm_solution(exact, approx)
    
end

function multiple_steps!(model)
    for i = 1:20
        time_step!(model, 1e-6)
    end
    return nothing
end

Nx = 20

if @isdefined wenoU
    close(wenoU)
end
if @isdefined wenoS
    close(wenoS)
end

arch         = GPU()
stretched    = true

if stretched

    Δx = zeros(Nx)

    Δx    .= 100
    Δx[4]  = 65
    Δx[5]  = 12
    Δx[6]  = 48
    Δx[7]  = 208
    Δx[8]  = 15

    xF = zeros(Float64, Nx+1)
    xF[1] = 0
    for i in 2:Nx+1
        xF[i] = xF[i-1] + Δx[i-1]
    end
    xF = xF ./ xF[end]
else    
    xF = range(0,1,length = Nx+1)
end

grid  = RectilinearGrid(size = (Nx,), x = xF, halo = (4,), topology = (Periodic, Flat, Flat), architecture = arch)    

Δ_min = min_Δx(grid)

# Time-scale for advection propagation across the smallest grid cell
CFL    = 0.75
Δt_max = CFL * Δ_min

wback(x, y, z, t) = 1.0

U = BackgroundField(wback)
c₀(x, y, z) = 10*exp(-((x-0.5)/0.2)^2)
advection    = [WENO5S(grid = grid), WENO5()]

for weno in advection
    model = NonhydrostaticModel(architecture = arch,
                                        grid = grid,
                                   advection = weno,
                                     tracers = (:c,),
                                 timestepper = :RungeKutta3,
                           background_fields = (u=U,),
                                    buoyancy = nothing)

    set!(model, c=c₀, u=0, v=0, w=0)

    c = model.tracers.c

    function progress(s)
        c = s.model.tracers.c
        @info "Maximum(c) = $(maximum(c)), time = $(s.model.clock.time) / $(s.stop_time)"
        return nothing
    end

    simulation = Simulation(model,
                            Δt = Δt_max,
                            stop_time = 4000*Δt_max)

                                                            
    output_fields = model.tracers

    output_prefix = "test_weno_$weno"

    simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                          schedule = TimeInterval(5*Δt_max),
                                                          prefix = output_prefix,
                                                          force = true)

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

    run!(simulation, pickup=false)

end

for weno in advection
    model = NonhydrostaticModel(architecture = arch,
                                        grid = grid,
                                   advection = weno,
                                     tracers = (:c,),
                           background_fields = (u=U,),
                                    buoyancy = nothing)


    c₀(x, y, z) = 10*exp(-((x-0.5)/0.2)^2)

    set!(model, c=c₀)

    for i=1:10
         time_step!(model, 1e-6) # warmup
    end
    @btime multiple_steps!($model)
end
        
wenoU = jldopen("test_weno_$(advection[1]).jld2")
wenoS = jldopen("test_weno_$(advection[2]).jld2")

global cu = ()
global cs = ()

for (i, key) in enumerate(keys(wenoU["timeseries/c"]))
    if i > 1
        global cu = (cu..., wenoU["timeseries/c/$key"])
        global cs = (cs..., wenoS["timeseries/c/$key"])
    end
end

x = grid.xᶜᵃᵃ[1:grid.Nx]

global t = 0
anim = @animate for i ∈ 1:length(cu)
    plot( x, cu[i][:], lw = 3, label = ["uniform weno scheme"])
    plot!(x, cs[i][:], lw = 3, label = ["stretched weno scheme"])
    plot!(x, c₀.(mod.((x .- t),1), 0, 0), seriestype = :scatter,  label = ["analytical"])
    global t += 5*Δt_max
end

stretched ? video = "str" : video = "reg"

run(`ffmpeg -r 5 -f image2 -s 1920x1080 -i $(anim.dir)/%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p $(video).mp4`)

@info "Finished plots"
