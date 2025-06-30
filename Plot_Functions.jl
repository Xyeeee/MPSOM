using Plots
using FFTW
using SciPy
using LaTeXStrings
include("Multivariate_Gaussian.jl")


function FIG_1()
    dt = 1e-3
    t_track = 0.0:dt:2 * pi/(ω(1,g₂)-ω(0,g₂)) *  2
    seg = Int(floor(length(t_track)/2))
    x = [real(2 * conj(0.4) * int_total(1,0,g₂,t,N_th)) for t in t_track]
    y = [imag(2 * conj(0.4) * int_total(1,0,g₂,t,N_th)) for t in t_track]
    scaling = x.^2 .+ y.^2
    x = x .* scaling
    y = y .* scaling

    x_1 = [real(2 * conj(0.3) * int_total(1,0,g₂,t,N_th)) for t in t_track]
    y_1 = [imag(2 * conj(0.3) * int_total(1,0,g₂,t,N_th)) for t in t_track]
    scaling_1 = x_1.^2 .+ y_1.^2
    x_1 = x_1 .* scaling_1
    y_1 = y_1 .* scaling_1
    plt = plot(legend= :outertopright) 
    plot!(x[1:seg],y[1:seg], label="s=0.2 I",thickness_scaling=1.5, line=(1,:solid,:blue), xlabel="rₓ/r²", ylabel="rᵧ/r²")
    plot!(x_1[1:seg],y_1[1:seg], label="s=0.1 I",thickness_scaling=1.5, line=(1,:solid,:red))
    plot!(x[seg+1:end],y[seg+1:end], label="s=0.2 II",thickness_scaling=1.5, line=(1,:dot,:blue))
    plot!(x_1[seg+1:end],y_1[seg+1:end], label="s=0.1 II",thickness_scaling=1.5, line=(1,:dot,:red))
    
    savefig("bloch_sphere_evolution.pdf")
end


function FIG_2()
    dt = 0.1
    t_track = 0.0:dt:2 * pi/(ω(1,g₂)-ω(0,g₂)) *  100
    num_track = t_track./(2 * pi/(ω(1,g₂)-ω(0,g₂)))
    x = [real(2 * conj(0.4) * int_total(1,0,g₂,t,N_th)) for t in t_track]
    plt = plot(legend=false)
    plot!(num_track,x, xlabel="t × δ/2π", thickness_scaling = 1.5, line=(0.5,:blue), ylabel="rₓ")
    savefig("x_evo.pdf")
end


function FIG_3()
    dt = 1.
    t_track = 0.0:dt:2 * pi/(Ω(1,g₂)-Ω(0,g₂)) * 4
    num_track = t_track./(2 * pi/(Ω(1,g₂)-Ω(0,g₂)))
    trace_square_track = [tr(get_mat(0.2,0.4,g₂,t,N_th)^2) for t in t_track]
    ts_2 = [tr(get_mat(0.5,0.4,g₂,t,N_th)^2) for t in t_track]
    plt = plot(legend = :topright)
    plot!(num_track, round.(real.(trace_square_track),digits=10), label="pure", thickness_scaling=1.5, line=(2,:solid,:blue),xlabel="t × Δ/2π", ylabel=L"\mathcal{P}")
    plot!(num_track, round.(real.(ts_2),digits=10), label="mixed", thickness_scaling=1.5, line=(2,:dot,:red))
    savefig("purity.pdf")   
end

function FIG_4()
    dt = 1.0
    t_track = 0.:dt:(2 * pi/(Ω(1,g₂)-Ω(0,g₂))) * 4
    num_track = t_track./(2 * pi/(Ω(1,g₂)-Ω(0,g₂)))
    ρ₀ = get_mat(0.2,0.4,g₂,0,N_th)
    Q_track = [QFI_general(ρ₀,g₂,t,N_th) for t in t_track]
    plt = plot(legend= :topleft)
    plot!(num_track, real.(Q_track), label="pure", xlabel="t × Δ/2π", ylabel="QFI", thickness_scaling=1.5, line=(2,:solid,:blue))
    savefig("QFI_pure.pdf")
end


function FIG_5()
    dt = 1e-3
    # t_track = 0.:dt:(2 * pi/(Ω(1,g₂)-Ω(0,g₂))) * 4
    t_track = (2 * pi/(Ω(1,g₂)-Ω(0,g₂))) - 10: dt: (2 * pi/(Ω(1,g₂)-Ω(0,g₂))) + 10
    num_track = t_track./(2 * pi/(Ω(1,g₂)-Ω(0,g₂)))
    Q_track = [QFI_Liu(0.5,0.4,g₂,t,N_th) for t in t_track]
    plt = plot(legend= :topleft)
    plot!(num_track, real.(Q_track), label="mixed", xlabel="t × Δ/2π", ylabel="QFI", thickness_scaling=1.5, line=(2,:solid,:red))
    savefig("QFI_mixed_0.2_zoomed.pdf")
end


function FIG_6()
    dt = 100.
    t_track = 0.0:dt:2 * pi/(Ω(1,g₂)-Ω(0,g₂)) * 1e3
    trace_square_track = [tr(get_mat(0.2,0.4,g₂,t,N_th)^2) for t in t_track]
    F = SciPy.fft.fft(trace_square_track)[1:length(t_track) ÷ 2]
    idxs = partialsortperm(abs.(F), 1:4, rev=true)
    freqs = fftfreq(length(t_track), 1/dt)[1:length(t_track) ÷ 2]
    println("Top 4 frequencies: ", freqs[idxs])
    plt = plot(legend = :topright)
    plot!(freqs.* 2*pi/(Ω(1,g₂)-Ω(0,g₂)), abs.(F),label="spectral peaks", yscale=:log10, xlabel="f × 2π/Δ", ylabel=L"|\tilde{\mathcal{P}}|", thickness_scaling=1.5, line=(2,:solid,:blue))
    savefig("spectral_purity.pdf")  
end


function FIG_7()
    # dt = 1.0
    # t_track = 0.0:dt:2 * pi/(ω(1,g₂)-ω(0,g₂)) *  10000

    dt = 100
    t_track = 0.0:dt:2 * pi/(Ω(1,g₂)-Ω(0,g₂)) *  1000
    x = [real(2 * conj(0.4) * int_total(1,0,g₂,t,N_th)) for t in t_track]
            
    F = SciPy.fft.fft(x)[1:length(t_track) ÷ 2]
    freqs = fftfreq(length(t_track), 1/dt)[1:length(t_track) ÷ 2]
    idxs = partialsortperm(abs.(F), 1:4, rev=true)
    println("Top 4 frequencies: ", freqs[idxs])
    plt = plot(legend= :false)  
    # plot!(freqs[Int(floor(length(freqs)/3.15)):Int(floor(length(freqs)/2.845))].* 2*pi/(ω(1,g₂)-ω(0,g₂)), abs.(F)[Int(floor(length(freqs)/3.15)):Int(floor(length(freqs)/2.845))],label="spectral peaks", yscale=:log10,xlabel="f × 2π/δ", ylabel=ylabel=L"|\tilde{r}_x|", thickness_scaling=1.5, line=(2,:solid,:red))
    plot!(freqs.* 2*pi/(Ω(1,g₂)-Ω(0,g₂)), abs.(F), yscale=:log10,xlabel="f × 2π/Δ", ylabel=ylabel=L"|\tilde{r}_x|", thickness_scaling=1.5, line=(2,:solid,:blue))
    savefig("spectral_x_long.pdf")
end

function FIG_8()
    dt = 1e-2
    t_track = (2 * pi/(Ω(1,g₂)-Ω(0,g₂))) - 10.:dt:(2 * pi/(Ω(1,g₂)-Ω(0,g₂))) + 10
    num_track = t_track./(2 * pi/(Ω(1,g₂)-Ω(0,g₂)))
    Q_track = [QFI_Liu(0.5,0.3,g₂,t,N_th) for t in t_track]
    C_track = [CFI(0.5,0.3,g₂,t,0.0,N_th) for t in t_track]
    plt = plot(legend= :topleft)
    plot!(num_track, real.(Q_track), label="QFI", xlabel="t × Δ/2π", ylabel="Fisher Information", thickness_scaling=1.5, line=(2,:solid,:red))
    plot!(num_track, real.(C_track), label="CFI", xlabel="t × Δ/2π", thickness_scaling=1.5, line=(2,:dash,:black))
    savefig("Q_and_C.pdf")
end

function FIG_9()
    t = 4 * pi/(Ω(1,g₂)-Ω(0,g₂)) + 1e-9
    phi_track = -pi:1e-2:pi
    QFI_track = QFI_Liu(0.5,0.3,g₂,t,N_th) * g₂^2 *ones(length(phi_track))
    CFI_track = [CFI(0.5,0.3,g₂,t,ϕ,N_th) for ϕ in phi_track] .* g₂^2
    plt = plot(legend= :topleft)
    plot!(phi_track./pi, real.(CFI_track), label="CFI", xlabel="ϕ/π", ylabel="Fisher Information", thickness_scaling=1.5, line=(2,:solid,:black))
    plot!(phi_track./pi, real.(QFI_track), label="QFI", xlabel="ϕ/π", ylabel="Fisher Information", thickness_scaling=1.5, line=(2,:dash,:red))
    savefig("phi_evo.pdf")
end

function FIG_10()
    log_T_track = -7:1e-1:8
    T_track = exp.(log_T_track)
    N_track = [1/(exp(h_bar * Ω₀/(k_b*temp))-1) for temp in T_track]
    t_1 = 2* 100 * pi*(1/(Ω(1,g₂)-Ω(0,g₂)) + 1/(ω(1,g₂)-ω(0,g₂)))
    t_2 = 1000 * t_1
    t_3 = 10000 * t_1

    Q_track = [QFI_Liu(0.5,0.3,g₂,t_1,N) for N in N_track].*g₂^2
    Q_2 = [QFI_Liu(0.5,0.3,g₂,t_2,N) for N in N_track].*g₂^2
    Q_3 = [QFI_Liu(0.5,0.3,g₂,t_3,N) for N in N_track].*g₂^2
    plt = plot(legend=:topright)
    plot!(
        plt, 
        T_track, 
        real.(Q_track), 
        label="N=100",
        xscale=:log10,
        yscale=:log10,
        xlabel="T",
        xticks=(10.0 .^ (-7:7)),
        ylabel="QFI",
        thickness_scaling=1.5, line=(:solid,:blue,2))
    plot!(plt, T_track, real.(Q_2), label="N=1000", xlabel="T", ylabel="QFI", thickness_scaling=1.5, line=(:dash,:blue,2))
    plot!(plt, T_track, real.(Q_3), label="N=100000", xlabel="T", ylabel="QFI",thickness_scaling=1.5, line=(:dot,:blue,2))
    savefig("QFI_versus_T.pdf")
end

function FIG_11()
    dt = 1e-3
    t_track = 0.0:dt:2 * pi/(ω(1,g₂)-ω(0,g₂)) *  2
    x = [real(2 * conj(0.4) * int_total(1,0,g₂,t,N_th)) for t in t_track]
    y = [imag(2 * conj(0.4) * int_total(1,0,g₂,t,N_th)) for t in t_track]
    z = 0.6*ones(length(t_track))
    x_1 = x
    y_1 = y
    z_1 = zeros(length(t_track))
    
    theta_1 = -pi:0.1:pi
    theta_2 = -pi:0.1:pi
    x_sphere = [cos(t_1) * cos(t_2) for t_1 in theta_1 for t_2 in theta_2]
    y_sphere = [cos(t_1) * sin(t_2) for t_1 in theta_1 for t_2 in theta_2]
    z_sphere = [sin(t_1)+t_2-t_2 for t_1 in theta_1 for t_2 in theta_2]
    plotlyjs()
    plt = plot(legend=:topright)
    plot!(x_sphere, y_sphere, z_sphere,colorbar = :none, label=:false, line=(0.1,:gray))
    plot!(x,y,z, label="s=0.2 pure",thickness_scaling=1.5, line=(1,:solid,:blue))
    plot!(x_1,y_1,z_1, label="s=0.2 mixed",thickness_scaling=1.5, line=(1,:solid,:red))

    # plot!(cos.(-pi:0.01:pi), sin.(-pi:0.01:pi), zeros(length(-pi:0.01:pi)), line=(1,:dash,:black))
    # plot!(cos.(-pi:0.01:pi), zeros(length(-pi:0.01:pi)), sin.(-pi:0.01:pi),  line=(1,:solid,:black))
    # arrows
    plot!([0.0,0.0],[0.0,0.0],[0.0,1.0],label=:false,arrow=true,line=(3,:solid,:black),zlabel="x")
    plot!([0.0,0.0],[0.0,1.0],[0.0,0.0],label=:false,arrow=true,yflip=true,line=(3,:solid,:black),ylabel="y")
    plot!([1.0,0.0],[0.0,0.0],[0.0,0.0],label=:false,arrow=true,line=(3,:solid,:black),xlabel="z")
    savefig("bloch_3D.pdf")
end

function plot_evolution_2level(a,b,bloch)
    if !bloch
        dt = 100.
        # t_track = 0.0:dt:2 * pi/(Ω(1,g₂)-Ω(0,g₂)) * 1e3
        # t_track = 2 * pi/(Ω(1,g₂)-Ω(0,g₂))- 1e-5: dt: 2 * pi/(Ω(1,g₂)-Ω(0,g₂)) + 1e-5
        t_track = 0.0:dt:2 * pi/(Ω(1,g₂)-Ω(0,g₂)) * 4
        num_track = t_track./(2 * pi/(Ω(1,g₂)-Ω(0,g₂)))
        trace_square_track = [tr(get_mat(a,b,g₂,t,N_th)^2) for t in t_track]
        # trace_track = [tr(get_mat(a,b,g₂,t)) for t in t_track]
        # t_1 = [tr(get_mat(0.8,0.4,g₂,t)) for t in t_track]
        ts_1 = [tr(get_mat(0.8,0.4,g₂,t,N_th)^2) for t in t_track]
        # t_2 = [tr(get_mat(0.5,0.3,g₂,t)) for t in t_track]
        ts_2 = [tr(get_mat(0.5,0.3,g₂,t,N_th)^2) for t in t_track]
        # ev_11 = [a * int_total(0,0,g₂,t) for t in t_track]
        # ev_12 = [b * int_total(0,1,g₂,t) for t in t_track]
        # ev_22 = [(1-a) * int_total(1,1,g₂,t) for t in t_track]
        # ev_33 = [conj(b) * int_total(1,0,g₂,t) for t in t_track]
        # A_part = [1/sqrt(det(get_A(0,1,g₂,t))) for t in t_track]
        # phi_part = [exp(-im * (ψ(0,g₂,t)-ψ(1,g₂,t))) for t in t_track]
        # F = sqrt.(SciPy.fft.fft(trace_square_track))[1:length(t_track) ÷ 2]
        # idxs = partialsortperm(abs.(F), 1:4, rev=true)
        # freqs = fftfreq(length(t_track), 1/dt)[1:length(t_track) ÷ 2]
        # println("Top 4 frequencies: ", freqs[idxs])
        # max_freq = maximum(freqs)
        # freqs_adj = [
        #     if freq < 0
        #         freq + 2*max_freq
        #     else
        #         freq
        #     end for freq in freqs
        # ]
        
        # println(freqs)
        # B_part = [exp(1/2* transpose(get_B(0,1,g₂,t)) * get_A(0,1,g₂,t)^-1 * get_B(0,1,g₂,t)) for t in t_track]
        # C_part = [exp(C(0,1,g₂,t)) for t in t_track]
        plt = plot(legend = :topright)
        # title!("Fourier transform of system dynamics")
        # xlabel!("Time")
        # ylabel!("Integral value")	
        # plot!(t_track, round.(real.(phi_part),digits=10), label=["phi"], xlabel="Time", ylabel="Real part of the integral", thickness_scale=0.5, alpha=0.5)
        # title!("Purity of state")
        # xlabel!("Frequency")
        # ylabel!("Relative amplitude")
        # plot!(freqs.* 2*pi/(Ω(1,g₂)-Ω(0,g₂)), abs.(F),label="spectral peaks", xlabel="f × 2π/Δ", ylabel="Relative Amplitude", thickness_scaling=1.5, line=(2,:solid,:blue))
        plot!(num_track, round.(real.(trace_square_track),digits=10), label="s=0.2", thickness_scaling=1.5, line=(2,:solid),xlabel="t × Δ/2π", ylabel="Purity")
        # plot!(t_track, round.(real.(trace_track),digits=10), label="Trace", xlabel="Time", ylabel="Trace")
        plot!(num_track, round.(real.(ts_1),digits=10), label="s=0.8",thickness_scaling=1.5, line=(2,:dash))
        # plot!(t_track, round.(real.(trace_track),digits=10), label="Trace", xlabel="Time", ylabel="Trace")
        plot!(num_track, round.(real.(ts_2),digits=10), label="mixed", thickness_scaling=1.5, line=(2,:dot))
        # plot!(t_track, round.(real.(trace_track),digits=10), label="Trace", xlabel="Time", ylabel="Trace")
        
        # display(plt)
        # return argmax(real.(trace_square_track[2:100]))
        # savefig("spectral_purity.pdf")
        savefig("purity.pdf")
        # idxs = partialsortperm(abs.(F), 1:8, rev=true)
        # println("Top 8 periods: ", 1 ./ freqs[idxs])
    else
        dt = 0.1
        t_track = 0.0:dt:2 * pi/(ω(1,g₂)-ω(0,g₂)) *  100
        num_track = t_track./(2 * pi/(ω(1,g₂)-ω(0,g₂)))
        # seg = Int(floor(length(t_track)/2))
        x = [real(2 * conj(b) * int_total(1,0,g₂,t,N_th)) for t in t_track]
        y = [imag(2 * conj(b) * int_total(1,0,g₂,t,N_th)) for t in t_track]
        scaling = x.^2 .+ y.^2
        x = x .* scaling.^2
        y = y .* scaling.^2


        x_1 = [real(2 * conj(0.3) * int_total(1,0,g₂,t,N_th)) for t in t_track]
        y_1 = [imag(2 * conj(0.3) * int_total(1,0,g₂,t,N_th)) for t in t_track]
        scaling_1 = x_1.^2 .+ y_1.^2
        x_1 = x_1 .* scaling_1.^2
        y_1 = y_1 .* scaling_1.^2

        
        #cos_list= [(1+cos(2*Ω(1,g₂) * t))  for t in t_track]
        
        plt = plot(legend= :topright)  
        
        # F = sqrt.(SciPy.fft.fft(x))[1:length(t_track) ÷ 2]
        # freqs = fftfreq(length(t_track), 1/dt)[1:length(t_track) ÷ 2]
        # idxs = partialsortperm(abs.(F), 1:4, rev=true)
        # println("Top 4 frequencies: ", freqs[idxs])
        # plot!(freqs[Int(floor(length(freqs)/3.5)):Int(floor(length(freqs)/2.5))], abs.(F)[Int(floor(length(freqs)/3.5)):Int(floor(length(freqs)/2.5))],label="spectral peaks", xlabel="Frequency", ylabel="Relative Amplitude", thickness_scaling=1.5, line=(2,:solid,:red))
        # title!("Fourier transform of Bloch sphere coordinates")
        # xlabel!("Frequency")
        # ylabel!("Amplitude")
        # plot!(freqs[Int(floor(length(freqs)/3.5)):Int(floor(length(freqs)/2.5))].* 2*pi/(ω(1,g₂)-ω(0,g₂)), abs.(F)[Int(floor(length(freqs)/3.5)):Int(floor(length(freqs)/2.5))],label="spectral peaks", xlabel="f × 2π/(ω₁-ω₀)", ylabel="Relative Amplitude", thickness_scaling=1.5, line=(2,:solid,:red))
        
        # display(plt)
        plot!(num_track,x,label="rₓ", xlabel="t × δ/2π", thickness_scaling = 1.5, line=(0.5,:blue), ylabel="rₓ")
        # plot!(t_track,cos_list,label="trial", xlabel="Time", ylabel="σₓ")
        
        # plot!(t_track,y,label="rᵧ", xlabel="Time", ylabel="rᵧ")
        # plot!(x,y)
        # display(plt)
        # plot!(x[1:seg],y[1:seg], label="s=0.2 -1",thickness_scaling=1.5, line=(1,:solid,:blue), xlabel="rₓ/r²", ylabel="rᵧ/r²")
        # plot!(x_1[1:seg],y_1[1:seg], label="s=0.8 -1",thickness_scaling=1.5, line=(1,:solid,:red))
        # plot!(x_2[1:seg],y_2[1:seg], label="mixed -1",thickness_scaling=1.5, line=(1,:solid,:green))
        # plot!(x[seg+1:end],y[seg+1:end], label="s=0.2 -2",thickness_scaling=1.5, line=(1,:dot,:blue))
        # plot!(x_1[seg+1:end],y_1[seg+1:end], label="s=0.8 -2",thickness_scaling=1.5, line=(1,:dot,:red))
        # plot!(x_2[seg+1:end],y_2[seg+1:end], label="mixed -2",thickness_scaling=1.5, line=(1,:dot,:green))
        # title!("Bloch sphere evolution, g=1e-2")
        savefig("x_evo.pdf")	
        # savefig("bloch_sphere_evolution.pdf")
    end
end


function print_QFI_2_levels(a,b)
    dt = 1e-5
    # t_track = 0.:dt:(2 * pi/(Ω(1,g₂)-Ω(0,g₂))) * 8
    # t_track = (2 * pi/(Ω(1,g₂)-Ω(0,g₂)))-1e-3: dt:(2 * pi/(Ω(1,g₂)-Ω(0,g₂))) + 1e-3
    # num_track = t_track./(2 * pi/(Ω(1,g₂)-Ω(0,g₂)))
    # QFI_Liu_track = [QFI_Liu(a,b,g₂,t,N_th) for t in t_track].* g₂^2
    # track_1 = [QFI_Liu(0.8,0.4,g₂,t) for t in t_track]
    # track_2 = [QFI_Liu(0.5,0.3,g₂,t) for t in t_track]
    # QFI_Liu_track_highg = [QFI_Liu(a,b,0.1,t) for t in t_track]
    # ρ₀ = get_mat(a,b,g₂,0)
    # rho_1 = get_mat(0.8,0.4,g₂,0)
    x_track = -pi :0.02:pi
    t = 4 * pi/(Ω(1,g₂)-Ω(0,g₂)) +1e-9
    t_1 = 4 * pi/(Ω(1,g₂)-Ω(0,g₂))
    QFI_track = ones(length(x_track)) .* QFI_Liu(a,b,g₂,t,N_th)* g₂^2
    CFI_track = [CFI(a,b,g₂,t,x,N_th) for x in x_track].* g₂^2
    Q_1 = ones(length(x_track)) .* QFI_Liu(a,b,g₂,t_1,N_th) * g₂^2
    C_1 = [CFI(a,b,g₂,t_1,x,N_th) for x in x_track].* g₂^2
    # QFI_general_track = [QFI_general(ρ₀,g₂,t) for t in t_track]
    # Q_8 = [QFI_general(rho_1,g₂,t) for t in t_track]
    # QFI_general_track_highg = [QFI_general(ρ₀,0.1,t) for t in t_track]
    plt = plot()
    # title!("Fisher Information evolution")
    # xlabel!("Time")
    # ylabel!("FI values")
    # plot!(t_track, real.(QFI_general_track_highg), label="QFI General High g", xlabel="Time", ylabel="QFI value")
    # plot!(t_track, real.(QFI_Liu_track_highg), label="QFI Liu High g", xlabel="Time", ylabel="QFI value")
    # plot!(num_track, real.(QFI_general_track), label="s=0.2", xlabel="t × (Ω₁-Ω₀)/2π", ylabel="QFI value", thickness_scaling=1.5, line=(2,:solid,:blue))
    # plot!(num_track, real.(Q_8), label="s=0.8",thickness_scaling=1.5, line=(2,:dash,:red))
    # plot!(num_track, real.(QFI_Liu_track), label="QFI", xlabel="t × (Ω₁-Ω₀)/2π", ylabel="Fisher Information", thickness_scaling=1.5, line=(2,:solid,:purple))
    # plot!(num_track, real.(CFI_track), label="CFI",line=(2,:dash,:yellow), thickness_scaling=1.5)
    plot!(x_track./pi, real.(CFI_track), label="CFI", xlabel="ϕ/π", ylabel="Fisher Information", thickness_scaling=1.5, line=(2,:solid,:green))
    plot!(x_track./pi, real.(QFI_track), label="QFI", xlabel="ϕ/π", ylabel="Fisher Information", thickness_scaling=1.5, line=(2,:dash,:orange))
    plot!(x_track./pi, real.(C_1), label="CFI T+1e-9", xlabel="ϕ/π", ylabel="Fisher Information", thickness_scaling=1.5, line=(2,:solid,:red))
    plot!(x_track./pi, real.(Q_1), label="QFI T+1e-9", xlabel="ϕ/π", ylabel="Fisher Information", thickness_scaling=1.5, line=(2,:dash,:blue))
    # plot!(t_track, real.(track_1), label="s=0.8",thickness_scaling=1.5, line=(5,:dash))
    # plot!(t_track, real.(track_2), label="mixed",thickness_scaling=1.5, line=(5,:dot))
    # display(plt)

    savefig("phi_diff.pdf")
    # savefig("Q_and_C_real.pdf")
end


function plot_fft_QFI(a,b)
    dt = 50
    t_track = 0.: dt:500000.
    QFI_track = [QFI_Liu(a,b,g₂,t,N_th) for t in t_track]
    adj_QFI_track = (QFI_track) .^(1)  # Adjusting for the square root
    F = fft(adj_QFI_track)[length(t_track) ÷ 20:length(t_track) ÷ 2]
    freqs = fftfreq(length(t_track), 1/dt)[length(t_track) ÷ 20:length(t_track) ÷ 2]
    plt = plot()
    title!("Fourier transform of QFI")
    xlabel!("Frequency")
    ylabel!("Relative amplitude")
    plot!(freqs, abs.(F))
    display(plt)
    idxs = partialsortperm(abs.(F), 1:8, rev=true)
    println("Top 8 frequencies: ", 1 ./ freqs[idxs])
end




function plot_eigen_2(a,b)
    t_track = 0.0:100:100000
    eigenvalues_1 = [eigvals(get_dev_mat(a,b,g₂,t,N_th))[1] for t in t_track]
    eigenvalues_2 = [eigvals(get_dev_mat(a,b,g₂,t,N_th))[2] for t in t_track]
    plt = plot()
    title!("Eigenvalue evolution")
    xlabel!("Time")
    ylabel!("eigenvalues")
    plot!(t_track, real.(eigenvalues_1), label="1st eval")
    plot!(t_track, real.(eigenvalues_2), label="2nd eval")
    display(plt)
end
plot_eigen_2(0.5,0.3)

function plot_general_QFI(ρ₀)
    track = 1:1e-2:200
    g_track = 1e-5:1e-5:1e-2
    qtrack = [QFI_general(ρ₀,g_track[5],t,N_th) for t in t_track]
    track = [QFI_general(ρ₀,g,t,N_th) for g in g_track]
    plt = plot(legend = :outertopleft)
    title!("QFI evolution")
    xlabel!("Time")
    ylabel!("QFI value")
    plot!(t_track, real.(qtrack), label="QFI 1e-2", xlabel="Time", ylabel="QFI value")
    plot!(g_track, real.(track), label="QFI General", xlabel="g", ylabel="QFI value")
    display(plt)
end



# print_QFI_2_levels(0.5,0.3)
# plot_eigen_2(0.5,0.3,g₂,10.0)
# plot_evolution_2level(0.2,0.4,true)
# plot_general_QFI(ρ₀)
# QFI_general(ρ₀,1e-2,10.0)
# plot_fft_QFI(0.5,0.3,1e-2,10.0)

