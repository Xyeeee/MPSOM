using Plots

mass = 1
Ω₀ = 1
σ = 1
ω₀ = 1

# n = 1
# m = 1
# t = 1

Ω(n) = sqrt(Ω₀^2 + 2 * g₂*n/mass)
g(n) = n * g₁/sqrt(2 * mass * Ω(n))
ω(n) = ω₀ * n + Ω(n)/2

ψ(n,t) = ω(n) * t - g(n)^2/Ω(n)^2 * (Ω(n) * t - sin(Ω(n) * t))
β(n,t) = g(n) / Ω(n) * (exp(-im * Ω(n) * t) - 1)


μ(n) = 1/2 * sqrt(Ω₀/Ω(n)) * (1 + Ω₀/Ω(n))
ν(n) = 1/2 * sqrt(Ω₀/Ω(n)) * (1 - Ω₀/Ω(n))

A_σ(n,m) = 2 + 1/σ^2 - ν(n)/μ(n) - ν(m)/μ(m)

Δ(n,m) = ν(n)/μ(n) - ν(m)/μ(m)

B_σ(n,m) = 1/σ^2 + 2 + (ν(n)/μ(n) + ν(m)/μ(m)) + Δ(n,m)^2/(2 * A_σ(n,m))

A(n,m) = 2 - ν(n)/μ(n) - ν(m)/μ(m)

B(n,m) = 2 + (ν(n)/μ(n) + ν(m)/μ(m)) + Δ(n,m)^2/(2 * A(n,m))

Δ′₋(n,m) = -Δ(n,m)/A(n,m) -1
Δ′₊(n,m) = -Δ(n,m)/A(n,m) +1

P(n,m) = ν(n)/μ(n) - (-Δ′₋(n,m)^2/(4 * B(n,m)) + 1/(4 * A(n,m))) * (2/μ(n))^2

Q(n,m) = ν(m)/μ(m) - (-Δ′₊(n,m)^2/(4 * B(n,m)) + 1/(4 * A(n,m))) * (2/μ(m))^2

P′(n,m) = ν(n)/μ(n) - (-Δ′₋(n,m)^2/(4 * B_σ(n,m)) + 1/(4 * A_σ(n,m))) * (2/μ(n))^2

Q′(n,m) = ν(m)/μ(m) - (-Δ′₊(n,m)^2/(4 * B_σ(n,m)) + 1/(4 * A_σ(n,m))) * (2/μ(m))^2

βᵣ(n,t) = real(β(n,t))*cos(Ω(n)*t) - imag(β(n,t))*sin(Ω(n)*t)
βᵢ(n,t) = real(β(n,t))*sin(Ω(n)*t) + imag(β(n,t))*cos(Ω(n)*t)

C(n,m) = 2 * (Δ′₋(n,m) *Δ′₊(n,m)/(4 * B(n,m)) - 1/(4 * A(n,m))) * (2/μ(n)) * (2/μ(m))

C′(n,m) = 2 * (Δ′₋(n,m) *Δ′₊(n,m)/(4 * B_σ(n,m)) - 1/(4 * A_σ(n,m))) * (2/μ(n)) * (2/μ(m))

D(n,m,t) = 2 - P′(n,m) - P(n,m)* exp(-2 * im * Ω(n) * t)

Φ(n,m,t) = 2 * (βᵣ(n,t) - P(n,m) * β(n,t) * exp(-im * Ω(n) * t))+ C(n,m) * conj(β(m,t)) * exp(-im * Ω(m) * t)

E(n,m,t) = 2 + P′(n,m) + P(n,m)* exp(-2 * im * Ω(n) * t) + (P′(n,m)-P(n,m)* exp(-2 * im * Ω(n) * t))^2/(4 * D(n,m,t))

Δₚ(n,m,t) = P′(n,m) - P(n,m)* exp(-2 * im * Ω(n) * t)

Ψ(n,m,t) = 2 * βᵢ(n,t) - 2 * im * P(n,m) * β(n,t) * exp(-im * Ω(n) * t) +  im * C(n,m) * conj(β(m,t)) * exp(-im * Ω(m) * t) - im * Δₚ(n,m,t)/D(n,m,t) * Φ(n,m,t)

Fᵣ(n,m,t) = -im * (C′(n,m) - C(n,m) * exp(im * (Ω(m) - Ω(n)) * t)) - im * Δₚ(n,m,t)/D(n,m,t) * (C′(n,m) + C(n,m) *  exp(im * (Ω(m) - Ω(n)) * t))

Fᵢ(n,m,t) = C′(n,m) * (1 + Δₚ(n,m,t)/D(n,m,t)) + C(n,m) * exp(im * (Ω(m) - Ω(n)) * t) * (1 - Δₚ(n,m,t)/D(n,m,t))

H(n,m,t) = 2 - Q′(n,m) - Q(n,m) * exp(-2 * im * Ω(m) * t) - Fᵣ(n,m,t)^2/(4 * E(n,m,t)) - (C′(n,m) + C(n,m) * exp(im * (Ω(m) - Ω(n)) * t))^2/(4 * D(n,m,t))

Z(n,m,t) = 2 * βᵣ(m,t) -2 * Φ(n,m,t)* (C′(n,m) - C(n,m) * exp(im * (Ω(m) - Ω(n)) * t))/(4 * D(n,m,t)) + C(n,m) * β(m,t) * exp(im * Ω(m) * t) -2 * Q(n,m) * conj(β(m,t)) * exp(im * Ω(m) * t) - 2 * Fᵣ(n,m,t) * Ψ(n,m,t)/(4 * E(n,m,t))

G(n,m,t) = -Fᵢ(n,m,t)^2/(4 * E(n,m,t)) - (C′(n,m) - C(n,m) * exp(im * (Ω(m) - Ω(n)) * t))^2/(4 * D(n,m,t)) + 2 + Q′(n,m) + Q(n,m) * exp(2 * im * Ω(m) * t) - 2*(im * ((C′(n,m)^2 - C(n,m)^2 * exp(2 * im * (Ω(m) - Ω(n)) * t))/(4 * D(n,m,t)) + (Q′(n,m) - Q(n,m) * exp(2 * im * Ω(m) * t))) - Fᵣ(n,m,t) * Fᵢ(n,m,t)/(4 * E(n,m,t)))^2/(4 * H(n,m,t))

ϕ_final(n,m,t) =(im * C(n,m) * β(n,t) * exp(im * Ω(m) * t) + 2 * im * Q(n,m) * exp(im * Ω(m)* t)* conj(β(n,t)) - 2 * Fᵢ(n,m,t) * Ψ(n,m,t)/(4 * E(n,m,t)) - 2 * Φ(n,m,t) * im * (C′(n,m) - C(n,m) * exp(im * (Ω(m) - Ω(n)) * t)) - Z(n,m,t)/H(n,m,t) * (im * (C′(n,m)^2 - C(n,m)^2 * exp(2 * im * (Ω(m) - Ω(n)) * t))/(4 * D(n,m,t)) + (Q′(n,m)- Q(n,m) * exp(2 * im * Ω(m) * t)) - Fᵣ(n,m,t) * Fᵢ(n,m,t)/(4 * E(n,m,t))) + 2* βᵢ(m,t))

time_evol(n,m,t) = 
    1/(μ(n)*μ(m)) * 
    2* exp(im * (ψ(n,t)-ψ(m,t))) /
    (
        (pi^4) * 
        sqrt((1/2)^8 * A(n,m) * B(n,m) * A_σ(n,m) * B_σ(n,m) * D(n,m,t) * E(n,m,t) * H(n,m,t) * G(n,m,t))
    ) * 
    exp(1/2 * ϕ_final(n,m,t)^2 / (4 * H(n,m,t))) * 
    exp(
        -1/2 * C(n,m) * β(n,t) * conj(β(m,t)) 
        - Ψ(n,m,t)^2/(4 * E(n,m,t)) 
        - Φ(n,m,t)^2/(4 * D(n,m,t)) 
        - 1/2 * (abs(β(n,t))^2 + abs(β(m,t))^2) 
        - 1/2 * P(n,m) * β(n,t)^2 -1/2 * Q(n,m) * β(m,t)^2
    )

println(time_evol(1,1,1))

# plot time_evol(n,m,t) for n = m in [0,1,2] and t between [0,1] with 0.01 step
ts = -pi:0.01:pi
plot()
for n in 1:1
    m = 1
    y = [βᵢ(n,t) for t in ts]
    plot!(t, [real(y), imag(y)], label="βᵢ(n,t),n=m=1")
end
xlabel!("Time (t)")
ylabel!("Φ(n,m,t)")
title!("Time Evolution for n=m=[0,1,2]")
