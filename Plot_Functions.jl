using Plots
include("Multivariate_Gaussian.jl")

function get_p_state(p)
    ρ = zeros(2,2)
    ρ[1,1] = p
    ρ[2,2] = 1 - p
    ρ[1,2] = sqrt(p * (1 - p))
    ρ[2,1] = sqrt(p * (1 - p))
    return ρ
end

function plot_evolution_2level(a,b,g₂,bloch)
    t_track = 0.0:10: 1200
    g_track = 1e-5:1e-5:1e-3
    trace_square_track = [tr(get_mat(a,b,g₂,t)^2) for t in t_track]
    trace_track = [tr(get_mat(a,b,g₂,t)) for t in t_track]
    if !bloch
        ev_11 = [a * int_total(0,0,g₂,t) for t in t_track]
        ev_12 = [b * int_total(0,1,g₂,t) for t in t_track]
        ev_22 = [(1-a) * int_total(1,1,g₂,t) for t in t_track]
        ev_33 = [conj(b) * int_total(1,0,g₂,t) for t in t_track]
        plt = plot(legend = :outertopleft)
        title!("Integral evolution, g=1e-10")
        xlabel!("Time")
        ylabel!("Integral value")	
        plot!(t_track, round.([real.(ev_11) real.(ev_22)],digits=10), label=["00" "11"], xlabel="Time", ylabel="Real part of the integral")
        display(plt)
    else
        x = [real(2 * conj(b) * int_total(1,0,g₂,t)) for t in t_track]
        y = [imag(2 * conj(b) * int_total(1,0,g₂,t)) for t in t_track]
        plt = plot(legend = :outertopleft)
        plot!(t_track,x)
        plot!(t_track,y)
        # plot!(t_track,real.(trace_square_track), label="Trace square", xlabel="Time", ylabel="Trace square")
        # plot!(t_track,real.(trace_track), label="Trace", xlabel="Time", ylabel="Trace")
        title!("Bloch sphere evolution, g=1e-2")
        display(plt)
    end
end


function print_QFI_2_levels(a,b,g₂,t)
    t_track = 0.:1.: 1200.
    QFI_Liu_track = [QFI_Liu(a,b,g₂,t) for t in t_track]
    # QFI_Liu_track_highg = [QFI_Liu(a,b,0.1,t) for t in t_track]
    ρ₀ = get_mat(a,b,g₂,0)
    # QFI_general_track = [QFI_general(ρ₀,g₂,t) for t in t_track]
    QFI_general_track_highg = [QFI_general(ρ₀,0.1,t) for t in t_track]
    plt = plot(legend = :outertopleft)
    title!("QFI evolution")
    xlabel!("Time")
    ylabel!("QFI value")
    # plot!(t_track, real.(QFI_general_track_highg), label="QFI General High g", xlabel="Time", ylabel="QFI value")
    # plot!(t_track, real.(QFI_Liu_track_highg), label="QFI Liu High g", xlabel="Time", ylabel="QFI value")
    # plot!(t_track, real.(QFI_general_track), label="QFI General", xlabel="Time", ylabel="QFI value")
    plot!(t_track, real.(QFI_Liu_track), label="QFI Liu", xlabel="Time", ylabel="QFI value")
    display(plt)
end

function plot_eigen_2(a,b,g₂,t)
    t_track = 0.0:1.:5000.0
    eigenvalues_1 = [eigvals(get_mat(a,b,g₂,t))[1] for t in t_track]
    eigenvalues_2 = [eigvals(get_mat(a,b,g₂,t))[2] for t in t_track]
    plt = plot(legend = :outertopleft)
    title!("eval evolution")
    xlabel!("Time")
    ylabel!("QFI value")
    plot!(t_track, real.(eigenvalues_1), label="1st eval")
    plot!(t_track, real.(eigenvalues_2), label="2nd eval")
    display(plt)
end


function plot_general_QFI(ρ₀)
    track = 1:1e-2:200
    g_track = 1e-5:1e-5:1e-2
    qtrack = [QFI_general(ρ₀,g_track[5],t) for t in t_track]
    track = [QFI_general(ρ₀,g,t) for g in g_track]
    plt = plot(legend = :outertopleft)
    title!("QFI evolution")
    xlabel!("Time")
    ylabel!("QFI value")
    plot!(t_track, real.(qtrack), label="QFI 1e-2", xlabel="Time", ylabel="QFI value")
    plot!(g_track, real.(track), label="QFI General", xlabel="g", ylabel="QFI value")
    display(plt)
end



# print_QFI_2_levels(0.5,0.3,1e-2,10.0)
plot_eigen_2(0.5,0.3,1e-2,10.0)

# plot_evolution_2level(0.5,0.3,1e-2,true)
# plot_general_QFI(ρ₀)
# QFI_general(ρ₀,1e-2,10.0)