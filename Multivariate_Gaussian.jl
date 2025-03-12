using LinearAlgebra
# Parameters involved in the model

mass = 1.0
Ω₀ = 1.0
σ = 1.0
ω₀ = 1.0
g₁ = 1.0
g₂ = 1e-3
m=1.
n=1.
t=1.


Ω(n) = sqrt(Ω₀^2 + 2 * g₂*n/mass)
g(n) = n * g₁/sqrt(2 * mass * Ω(n))
ω(n) = ω₀ * n + Ω(n)/2

ψ(n,t) = ω(n) * t - g(n)^2/Ω(n)^2 * (Ω(n) * t - sin(Ω(n) * t))
β(n,t) = g(n) / Ω(n) * (exp(-im * Ω(n) * t) - 1)


μ(n) = 1/2 * sqrt(Ω₀/Ω(n)) * (1 + Ω₀/Ω(n))
ν(n) = 1/2 * sqrt(Ω₀/Ω(n)) * (1 - Ω₀/Ω(n))

# Integral matrix definitions
# 1. η real 2. η imag 3. γ real 4. γ imag 5. ϵ real 6. ϵ imag 7. δ real 8. δ imag
function get_A(n,m,t)

    A = zeros(Complex{Float64}, 8, 8)
    A[1,1] = 2 + 1/σ^2 + ν(n)/μ(n) + ν(m)/μ(m)
    A[1,2] = im * (ν(n)/μ(n) - ν(m)/μ(m))
    A[2,1] = im * (ν(n)/μ(n) - ν(m)/μ(m))
    A[1,3] = -1/μ(n)
    A[4,1] = -1/μ(n)
    A[1,4] = im/μ(n)
    A[4,1] = im/μ(n)
    A[1,5] = 0
    A[5,1] = 0
    A[1,6] = 0
    A[6,1] = 0
    A[1,7] = -1/μ(m)
    A[7,1] = -1/μ(m)
    A[1,8] = -im/μ(m)
    A[8,1] = -im/μ(m)

    A[2,2] = 2 + 1/σ^2 - ν(n)/μ(n) - ν(m)/μ(m)
    A[2,3] = -im/μ(n)
    A[3,2] = -im/μ(n)
    A[2,4] = -1/μ(n)
    A[4,2] = -1/μ(n)
    A[2,5] = 0
    A[5,2] = 0
    A[2,6] = 0
    A[6,2] = 0
    A[2,7] = im/μ(m)
    A[7,2] = im/μ(m)
    A[2,8] = -1/μ(m)
    A[8,2] = -1/μ(m)

    A[3,3] = 2 - ν(n)/μ(n) * (1 + exp(-2 * im * Ω(n) * t))
    A[3,4] = -ν(n)/μ(n) * im * exp(-2 * im * Ω(n) * t)
    A[4,3] = -ν(n)/μ(n) * im * exp(-2 * im * Ω(n) * t)
    A[3,5] = -1/μ(n) * exp(-im * Ω(n) * t)
    A[5,3] = -1/μ(n) * exp(-im * Ω(n) * t)
    A[3,6] = -im/μ(n) * exp(-im * Ω(n) * t)
    A[6,3] = -im/μ(n) * exp(-im * Ω(n) * t)
    A[3,7] = 0
    A[7,3] = 0
    A[3,8] = 0
    A[8,3] = 0

    A[4,4] = 2 + ν(n)/μ(n) * (1 + exp(-2 * im * Ω(n) * t))
    A[4,5] = -im/μ(n) * exp(-im * Ω(n) * t)
    A[5,4] = -im/μ(n) * exp(-im * Ω(n) * t)
    A[4,6] = -1/μ(n) * exp(-im * Ω(n) * t)
    A[6,4] = -1/μ(n) * exp(-im * Ω(n) * t)
    A[4,7] = 0
    A[7,4] = 0
    A[4,8] = 0
    A[8,4] = 0

    A[5,5] = 2 + ν(m)/μ(m) + ν(n)/μ(n) 
    A[5,6] = im * (ν(m)/μ(m) - ν(n)/μ(n))
    A[6,5] = im * (ν(m)/μ(m) - ν(n)/μ(n))
    A[5,7] = -1/μ(m) * exp(im * Ω(m) * t)
    A[7,5] = -1/μ(m) * exp(im * Ω(m) * t)
    A[5,8] = im/μ(m) * exp(im * Ω(m) * t)
    A[8,5] = im/μ(m) * exp(im * Ω(m) * t)

    A[6,6] = 2 - ν(m)/μ(m) - ν(n)/μ(n)
    A[6,7] = -im/μ(m) * exp(im * Ω(m) * t)
    A[7,6] = -im/μ(m) * exp(im * Ω(m) * t)
    A[6,8] = -1/μ(m) * exp(im * Ω(m) * t)
    A[8,6] = -1/μ(m) * exp(im * Ω(m) * t)

    A[7,7] = 2 - ν(m)/μ(m) * (1 + exp(2 * im * Ω(m) * t))
    A[7,8] = ν(m)/μ(m) * im * exp(2 * im * Ω(m) * t)
    A[8,7] = ν(m)/μ(m) * im * exp(2 * im * Ω(m) * t)

    A[8,8] = 2 + ν(m)/μ(m) * (1 + exp(2 * im * Ω(m) * t))
    return A
end

function get_B(n,m,t)
    B = zeros(Complex{Float64},8)
    B[1] = 0
    B[2] = 0
    B[3] = 2 * (real(β(n,t)) * cos(Ω(n) * t) - imag(β(n,t)) * sin(Ω(n) * t) - β(n,t) *  ν(n)/μ(n) *  exp(-im * Ω(n) * t))
    B[4] = 2 * (real(β(n,t)) * sin(Ω(n) * t) + imag(β(n,t)) * cos(Ω(n) * t) - im * β(n,t) *  ν(n)/μ(n) *  exp(-im * Ω(n) * t))
    B[5] = -2/μ(n) * β(n,t) - 2/μ(m) * conj(β(m,t))
    B[6] = 2 * im/μ(n) * β(n,t) -2* im/μ(m) * conj(β(m,t))
    B[7] = 2 * (real(β(m,t)) * cos(Ω(m) * t) - imag(β(m,t)) * sin(Ω(m) * t) - conj(β(m,t)) *  ν(m)/μ(m) *  exp(im * Ω(m) * t))
    B[8] = 2 * (real(β(m,t)) * sin(Ω(m) * t) + imag(β(m,t)) * cos(Ω(m) * t) + im * conj(β(m,t)) *  ν(m)/μ(m) *  exp(im * Ω(m) * t))

    return -1/2 * B
end

C(n,m,t) = -1/2 * abs(β(n,t))^2 * (1 - ν(n)/μ(n)) -1/2 * abs(β(m,t))^2 * (1 - ν(m)/μ(m))

int_total(n,m,t) = exp(-im * (ψ(n,t)-ψ(m,t)))/(μ(n) * μ(m)) * sqrt((2*π)^8/det(get_A(n,m,t))) * exp(C(n,m,t)) * exp(1/2 * get_B(n,m,t)' * get_A(n,m,t)^-1 * get_B(n,m,t))

int_total(1,1,1)
