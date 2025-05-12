using LinearAlgebra
using ForwardDiff
using QuadGK


# Parameters involved in the model

mass = 1.0
Ω₀ = 1.
σ = 1.0
ω₀ = 1.0
g₁ = 0.0
g₂ = 1e-2
N_th = 1.0

Ω(n,g₂) = sqrt(Ω₀^2 + 2 * g₂*n/mass)
dev_Ω(n,g₂) = 1/2 * 2 * n/(mass *Ω(n,g₂))
g(n,g₂) = n * g₁/sqrt(2 * mass * Ω(n,g₂))
dev_g(n,g₂) = -1/2 * g(n,g₂)/Ω(n,g₂) * dev_Ω(n,g₂)
ω(n,g₂) = ω₀ * n + Ω(n,g₂)/2
dev_ω(n,g₂) = 1/2 * dev_Ω(n,g₂)

ψ(n,g₂,t) = ω(n,g₂) * t - g(n,g₂)^2/Ω(n,g₂)^2 * (Ω(n,g₂) * t - sin(Ω(n,g₂) * t))
dev_ψ(n,g₂,t) = dev_ω(n,g₂) * t - 2 * (dev_g(n,g₂) * g(n,g₂)/Ω(n,g₂)^2 - g(n,g₂)^2/Ω(n,g₂)^3 * dev_Ω(n,g₂))* (Ω(n,g₂) * t - sin(Ω(n,g₂) * t)) - g(n,g₂)^2/Ω(n,g₂)^2 * (dev_Ω(n,g₂) * t - t * dev_Ω(n,g₂) * cos(Ω(n,g₂) * t))
β(n,g₂,t) = g(n,g₂) / Ω(n,g₂) * (exp(-im * Ω(n,g₂) * t) - 1)
dev_β(n,g₂,t) = (dev_g(n,g₂)/Ω(n,g₂) -g(n,g₂)/Ω(n,g₂)^2 * dev_Ω(n,g₂))  * (exp(-im * Ω(n,g₂) * t) - 1) -im * t * dev_Ω(n,g₂)* g(n,g₂) / Ω(n,g₂) * exp(-im * Ω(n,g₂) * t) 
μ(n,g₂) = 1/2 * sqrt(Ω(n,g₂)/Ω₀) * (1 + Ω₀/Ω(n,g₂))
dev_μ(n,g₂) = 1/2 * (μ(n,g₂)/Ω(n,g₂)* dev_Ω(n,g₂) - sqrt(Ω(n,g₂)/Ω₀) * Ω₀/Ω(n,g₂)^2 * dev_Ω(n,g₂))
ν(n,g₂) = 1/2 * sqrt(Ω(n,g₂)/Ω₀) * (1 - Ω₀/Ω(n,g₂))
dev_ν(n,g₂) = 1/2 * (ν(n,g₂)/Ω(n,g₂)* dev_Ω(n,g₂) + sqrt(Ω(n,g₂)/Ω₀) * Ω₀/Ω(n,g₂)^2 * dev_Ω(n,g₂))
# Integral matrix definitions
# 1. η real 2. η imag 3. γ real 4. γ imag 5. ϵ real 6. ϵ imag 7. δ real 8. δ imag
function get_A(n,m,g₂,t)
    A = zeros(Complex{Float64}, 8, 8)
    A[1,1] = 2 + 2/(N_th) + ν(n,g₂)/μ(n,g₂) + ν(m,g₂)/μ(m,g₂)
    A[1,2] = im * (ν(n,g₂)/μ(n,g₂) - ν(m,g₂)/μ(m,g₂))
    A[2,1] = im * (ν(n,g₂)/μ(n,g₂) - ν(m,g₂)/μ(m,g₂))
    A[1,3] = -1/μ(n,g₂)
    A[3,1] = -1/μ(n,g₂)
    A[1,4] = im/μ(n,g₂)
    A[4,1] = im/μ(n,g₂)
    A[1,5] = 0
    A[5,1] = 0
    A[1,6] = 0
    A[6,1] = 0
    A[1,7] = -1/μ(m,g₂)
    A[7,1] = -1/μ(m,g₂)
    A[1,8] = -im/μ(m,g₂)
    A[8,1] = -im/μ(m,g₂)

    A[2,2] = 2 + 2/(N_th) - ν(n,g₂)/μ(n,g₂) - ν(m,g₂)/μ(m,g₂)
    A[2,3] = -im/μ(n,g₂)
    A[3,2] = -im/μ(n,g₂)
    A[2,4] = -1/μ(n,g₂)
    A[4,2] = -1/μ(n,g₂)
    A[2,5] = 0
    A[5,2] = 0
    A[2,6] = 0
    A[6,2] = 0
    A[2,7] = im/μ(m,g₂)
    A[7,2] = im/μ(m,g₂)
    A[2,8] = -1/μ(m,g₂)
    A[8,2] = -1/μ(m,g₂)

    A[3,3] = 2 - ν(n,g₂)/μ(n,g₂) * (1 + exp(-2 * im * Ω(n,g₂) * t))
    A[3,4] = ν(n,g₂)/μ(n,g₂) * im * (1 - exp(-2 * im * Ω(n,g₂) * t))
    A[4,3] = ν(n,g₂)/μ(n,g₂) * im * (1 - exp(-2 * im * Ω(n,g₂) * t))
    A[3,5] = -1/μ(n,g₂) * exp(-im * Ω(n,g₂) * t)
    A[5,3] = -1/μ(n,g₂) * exp(-im * Ω(n,g₂) * t)
    A[3,6] = im/μ(n,g₂) * exp(-im * Ω(n,g₂) * t)
    A[6,3] = im/μ(n,g₂) * exp(-im * Ω(n,g₂) * t)
    A[3,7] = 0
    A[7,3] = 0
    A[3,8] = 0
    A[8,3] = 0

    A[4,4] = 2 + ν(n,g₂)/μ(n,g₂) * (1 + exp(-2 * im * Ω(n,g₂) * t))
    A[4,5] = -im/μ(n,g₂) * exp(-im * Ω(n,g₂) * t)
    A[5,4] = -im/μ(n,g₂) * exp(-im * Ω(n,g₂) * t)
    A[4,6] = -1/μ(n,g₂) * exp(-im * Ω(n,g₂) * t)
    A[6,4] = -1/μ(n,g₂) * exp(-im * Ω(n,g₂) * t)
    A[4,7] = 0
    A[7,4] = 0
    A[4,8] = 0
    A[8,4] = 0

    A[5,5] = 2 + ν(m,g₂)/μ(m,g₂) + ν(n,g₂)/μ(n,g₂) 
    A[5,6] = im * (ν(m,g₂)/μ(m,g₂) - ν(n,g₂)/μ(n,g₂))
    A[6,5] = im * (ν(m,g₂)/μ(m,g₂) - ν(n,g₂)/μ(n,g₂))
    A[5,7] = -1/μ(m,g₂) * exp(im * Ω(m,g₂) * t)
    A[7,5] = -1/μ(m,g₂) * exp(im * Ω(m,g₂) * t)
    A[5,8] = im/μ(m,g₂) * exp(im * Ω(m,g₂) * t)
    A[8,5] = im/μ(m,g₂) * exp(im * Ω(m,g₂) * t)

    A[6,6] = 2 - ν(m,g₂)/μ(m,g₂) - ν(n,g₂)/μ(n,g₂)
    A[6,7] = -im/μ(m,g₂) * exp(im * Ω(m,g₂) * t)
    A[7,6] = -im/μ(m,g₂) * exp(im * Ω(m,g₂) * t)
    A[6,8] = -1/μ(m,g₂) * exp(im * Ω(m,g₂) * t)
    A[8,6] = -1/μ(m,g₂) * exp(im * Ω(m,g₂) * t)

    A[7,7] = 2 - ν(m,g₂)/μ(m,g₂) * (1 + exp(2 * im * Ω(m,g₂) * t))
    A[7,8] = ν(m,g₂)/μ(m,g₂) * im * (exp(2 * im * Ω(m,g₂) * t)-1)
    A[8,7] = ν(m,g₂)/μ(m,g₂) * im * (exp(2 * im * Ω(m,g₂) * t)-1)

    A[8,8] = 2 + ν(m,g₂)/μ(m,g₂) * (1 + exp(2 * im * Ω(m,g₂) * t))
    return A
end

function get_B(n,m,g₂,t)
    B = zeros(Complex{Float64},8)
    B[1] = 0
    B[2] = 0
    B[3] = 2 * (real(β(n,g₂,t)) * cos(Ω(n,g₂) * t) - imag(β(n,g₂,t)) * sin(Ω(n,g₂) * t) - β(n,g₂,t) *  ν(n,g₂)/μ(n,g₂) *  exp(-im * Ω(n,g₂) * t))
    B[4] = 2 * (real(β(n,g₂,t)) * sin(Ω(n,g₂) * t) + imag(β(n,g₂,t)) * cos(Ω(n,g₂) * t) - im * β(n,g₂,t) *  ν(n,g₂)/μ(n,g₂) *  exp(-im * Ω(n,g₂) * t))
    B[5] = -2/μ(n,g₂) * β(n,g₂,t) - 2/μ(m,g₂) * conj(β(m,g₂,t))
    B[6] = 2 * im/μ(n,g₂) * β(n,g₂,t) -2* im/μ(m,g₂) * conj(β(m,g₂,t))
    B[7] = 2 * (real(β(m,g₂,t)) * cos(Ω(m,g₂) * t) - imag(β(m,g₂,t)) * sin(Ω(m,g₂) * t) - conj(β(m,g₂,t)) *  ν(m,g₂)/μ(m,g₂) *  exp(im * Ω(m,g₂) * t))
    B[8] = 2 * (real(β(m,g₂,t)) * sin(Ω(m,g₂) * t) + imag(β(m,g₂,t)) * cos(Ω(m,g₂) * t) + im * conj(β(m,g₂,t)) *  ν(m,g₂)/μ(m,g₂) *  exp(im * Ω(m,g₂) * t))
    
    B[3] += - β(n,g₂,t) * exp(im * Ω(n,g₂) * t) + conj(β(n,g₂,t)) * exp(-im * Ω(n,g₂) * t)
    B[4] += im * β(n,g₂,t) * exp(im * Ω(n,g₂) * t) + im * conj(β(n,g₂,t)) * exp(-im * Ω(n,g₂) * t)
    B[7] += - conj(β(m,g₂,t)) * exp(-im * Ω(m,g₂) * t) + β(m,g₂,t) * exp(im * Ω(m,g₂) * t)
    B[8] += - im * conj(β(m,g₂,t)) * exp(-im * Ω(m,g₂) * t) - im * β(m,g₂,t) * exp(im * Ω(m,g₂) * t)
    return -1/2 * B
end

C(n,m,g₂,t) = -1/2 * (abs(β(n,g₂,t))^2 + abs(β(m,g₂,t))^2 - ν(m,g₂)/μ(m,g₂) * conj(β(m,g₂,t)^2) - ν(n,g₂)/μ(n,g₂) * β(n,g₂,t)^2) 


int_total(n,m,g₂,t) = 1/(π^3 * π * N_th)*exp(-im * (ψ(n,g₂,t)-ψ(m,g₂,t)))/(μ(n,g₂) * μ(m,g₂)) * sqrt((2*π)^8/det(get_A(n,m,g₂,t))) * exp(C(n,m,g₂,t)) * exp(1/2* transpose(get_B(n,m,g₂,t)) * get_A(n,m,g₂,t)^-1 * get_B(n,m,g₂,t))

function get_dev_A(n,m,g₂,t)
    dev_A = zeros(Complex{Float64}, 8, 8)
    dev_A[1,1] =  dev_ν(n,g₂)/μ(n,g₂) - ν(n,g₂)/μ(n,g₂)^2 * dev_μ(n,g₂) + dev_ν(m,g₂)/μ(m,g₂) - ν(m,g₂)/μ(m,g₂)^2 * dev_μ(m,g₂)
    dev_A[1,2] = im * (dev_ν(n,g₂)/μ(n,g₂) - ν(n,g₂)/μ(n,g₂)^2 * dev_μ(n,g₂) - dev_ν(m,g₂)/μ(m,g₂) + ν(m,g₂)/μ(m,g₂)^2 * dev_μ(m,g₂))
    dev_A[2,1] = im * (dev_ν(n,g₂)/μ(n,g₂) - ν(n,g₂)/μ(n,g₂)^2 * dev_μ(n,g₂) - dev_ν(m,g₂)/μ(m,g₂) + ν(m,g₂)/μ(m,g₂)^2 * dev_μ(m,g₂))
    dev_A[1,3] = 1/μ(n,g₂)^2 * dev_μ(n,g₂) 
    dev_A[3,1] = 1/μ(n,g₂)^2 * dev_μ(n,g₂)
    dev_A[1,4] = -im/μ(n,g₂)^2 * dev_μ(n,g₂)
    dev_A[4,1] = -im/μ(n,g₂)^2 * dev_μ(n,g₂)
    dev_A[1,5] = 0
    dev_A[5,1] = 0
    dev_A[1,6] = 0
    dev_A[6,1] = 0
    dev_A[1,7] = 1/μ(m,g₂)^2 * dev_μ(m,g₂)
    dev_A[7,1] = 1/μ(m,g₂)^2 * dev_μ(m,g₂)
    dev_A[1,8] = im/μ(m,g₂)^2 * dev_μ(m,g₂)
    dev_A[8,1] = im/μ(m,g₂)^2 * dev_μ(m,g₂)

    dev_A[2,2] = -dev_ν(n,g₂)/μ(n,g₂) + ν(n,g₂)/μ(n,g₂)^2 * dev_μ(n,g₂) - dev_ν(m,g₂)/μ(m,g₂) + ν(m,g₂)/μ(m,g₂)^2 * dev_μ(m,g₂)
    dev_A[2,3] = im/μ(n,g₂)^2 * dev_μ(n,g₂)
    dev_A[3,2] = im/μ(n,g₂)^2 * dev_μ(n,g₂)
    dev_A[2,4] = 1/μ(n,g₂)^2 * dev_μ(n,g₂)
    dev_A[4,2] = 1/μ(n,g₂)^2 * dev_μ(n,g₂)
    dev_A[2,5] = 0
    dev_A[5,2] = 0
    dev_A[2,6] = 0
    dev_A[6,2] = 0
    dev_A[2,7] = -im/μ(m,g₂)^2 * dev_μ(m,g₂)
    dev_A[7,2] = -im/μ(m,g₂)^2 * dev_μ(m,g₂)
    dev_A[2,8] = 1/μ(m,g₂)^2 * dev_μ(m,g₂)
    dev_A[8,2] = 1/μ(m,g₂)^2 * dev_μ(m,g₂)
    
    dev_A[3,3] = (-dev_ν(n,g₂)/μ(n,g₂) + ν(n,g₂)/μ(n,g₂)^2 * dev_μ(n,g₂)) * (1 + exp(-2 * im * Ω(n,g₂) * t)) + 2 * ν(n,g₂)/μ(n,g₂) * dev_Ω(n,g₂) * t* im * exp(-2 * im * Ω(n,g₂) * t)
    dev_A[3,4] = im*(dev_ν(n,g₂)/μ(n,g₂) - ν(n,g₂)/μ(n,g₂)^2 * dev_μ(n,g₂))* (1 - exp(-2 * im * Ω(n,g₂) * t)) - 2 * ν(n,g₂)/μ(n,g₂) * dev_Ω(n,g₂) * t* exp(-2 * im * Ω(n,g₂) * t)
    dev_A[4,3] = im*(dev_ν(n,g₂)/μ(n,g₂) - ν(n,g₂)/μ(n,g₂)^2 * dev_μ(n,g₂))* (1 - exp(-2 * im * Ω(n,g₂) * t)) - 2 * ν(n,g₂)/μ(n,g₂) * dev_Ω(n,g₂) * t* exp(-2 * im * Ω(n,g₂) * t)
    dev_A[3,5] = 1/μ(n,g₂)^2 * dev_μ(n,g₂) * exp(-im * Ω(n,g₂) * t) + 1/μ(n,g₂) * dev_Ω(n,g₂) * t* im * exp(-im * Ω(n,g₂) * t)
    dev_A[5,3] = 1/μ(n,g₂)^2 * dev_μ(n,g₂) * exp(-im * Ω(n,g₂) * t) + 1/μ(n,g₂) * dev_Ω(n,g₂) * t* im * exp(-im * Ω(n,g₂) * t)
    dev_A[3,6] = -im/μ(n,g₂)^2 * dev_μ(n,g₂) * exp(-im * Ω(n,g₂) * t) + 1/μ(n,g₂) * dev_Ω(n,g₂) * t* exp(-im * Ω(n,g₂) * t)
    dev_A[6,3] = -im/μ(n,g₂)^2 * dev_μ(n,g₂) * exp(-im * Ω(n,g₂) * t) + 1/μ(n,g₂) * dev_Ω(n,g₂) * t* exp(-im * Ω(n,g₂) * t)
    dev_A[3,7] = 0
    dev_A[7,3] = 0
    dev_A[3,8] = 0
    dev_A[8,3] = 0

    dev_A[4,4] = (dev_ν(n,g₂)/μ(n,g₂)- ν(n,g₂)/μ(n,g₂)^2 * dev_μ(n,g₂)) * (1 + exp(-2 * im * Ω(n,g₂) * t)) - 2 * ν(n,g₂)/μ(n,g₂) * dev_Ω(n,g₂) * t * im * exp(-2 * im * Ω(n,g₂) * t)
    dev_A[4,5] = im/μ(n,g₂)^2 * dev_μ(n,g₂) * exp(-im * Ω(n,g₂) * t) - 1/μ(n,g₂) * dev_Ω(n,g₂) * t* exp(-im * Ω(n,g₂) * t)
    dev_A[5,4] = im/μ(n,g₂)^2 * dev_μ(n,g₂) * exp(-im * Ω(n,g₂) * t) - 1/μ(n,g₂) * dev_Ω(n,g₂) * t* exp(-im * Ω(n,g₂) * t)
    dev_A[4,6] = 1/μ(n,g₂)^2 * dev_μ(n,g₂) * exp(-im * Ω(n,g₂) * t) + im/μ(n,g₂) * dev_Ω(n,g₂) * t* exp(-im * Ω(n,g₂) * t)
    dev_A[6,4] = 1/μ(n,g₂)^2 * dev_μ(n,g₂) * exp(-im * Ω(n,g₂) * t) + im/μ(n,g₂) * dev_Ω(n,g₂) * t* exp(-im * Ω(n,g₂) * t)
    dev_A[4,7] = 0
    dev_A[7,4] = 0
    dev_A[4,8] = 0
    dev_A[8,4] = 0

    dev_A[5,5] = dev_ν(m,g₂)/μ(m,g₂) - ν(m,g₂)/μ(m,g₂)^2 * dev_μ(m,g₂) + dev_ν(n,g₂)/μ(n,g₂) - ν(n,g₂)/μ(n,g₂)^2 * dev_μ(n,g₂)
    dev_A[5,6] = im * (dev_ν(m,g₂)/μ(m,g₂) - ν(m,g₂)/μ(m,g₂)^2 * dev_μ(m,g₂) - dev_ν(n,g₂)/μ(n,g₂) + ν(n,g₂)/μ(n,g₂)^2 * dev_μ(n,g₂))
    dev_A[6,5] = im * (dev_ν(m,g₂)/μ(m,g₂) - ν(m,g₂)/μ(m,g₂)^2 * dev_μ(m,g₂) - dev_ν(n,g₂)/μ(n,g₂) + ν(n,g₂)/μ(n,g₂)^2 * dev_μ(n,g₂))
    dev_A[5,7] = 1/μ(m,g₂)^2 * dev_μ(m,g₂) * exp(im * Ω(m,g₂) * t) - 1/μ(m,g₂) * dev_Ω(m,g₂) * t* im * exp(im * Ω(m,g₂) * t)
    dev_A[7,5] = 1/μ(m,g₂)^2 * dev_μ(m,g₂) * exp(im * Ω(m,g₂) * t) - 1/μ(m,g₂) * dev_Ω(m,g₂) * t* im * exp(im * Ω(m,g₂) * t)
    dev_A[5,8] = -im/μ(m,g₂)^2 * dev_μ(m,g₂) * exp(im * Ω(m,g₂) * t) - 1/μ(m,g₂) * dev_Ω(m,g₂) * t* exp(im * Ω(m,g₂) * t)
    dev_A[8,5] = -im/μ(m,g₂)^2 * dev_μ(m,g₂) * exp(im * Ω(m,g₂) * t) - 1/μ(m,g₂) * dev_Ω(m,g₂) * t* exp(im * Ω(m,g₂) * t)

    dev_A[6,6] = -dev_ν(m,g₂)/μ(m,g₂) + ν(m,g₂)/μ(m,g₂)^2 * dev_μ(m,g₂) - dev_ν(n,g₂)/μ(n,g₂) + ν(n,g₂)/μ(n,g₂)^2 * dev_μ(n,g₂)
    dev_A[6,7] = im/μ(m,g₂)^2 * dev_μ(m,g₂) * exp(im * Ω(m,g₂) * t) + 1/μ(m,g₂) * dev_Ω(m,g₂) * t* exp(im * Ω(m,g₂) * t)
    dev_A[7,6] = im/μ(m,g₂)^2 * dev_μ(m,g₂) * exp(im * Ω(m,g₂) * t) + 1/μ(m,g₂) * dev_Ω(m,g₂) * t* exp(im * Ω(m,g₂) * t)
    dev_A[6,8] = 1/μ(m,g₂)^2 * dev_μ(m,g₂) * exp(im * Ω(m,g₂) * t) - im/μ(m,g₂) * dev_Ω(m,g₂) * t* exp(im * Ω(m,g₂) * t)
    dev_A[8,6] = 1/μ(m,g₂)^2 * dev_μ(m,g₂) * exp(im * Ω(m,g₂) * t) - im/μ(m,g₂) * dev_Ω(m,g₂) * t* exp(im * Ω(m,g₂) * t)

    dev_A[7,7] = (-dev_ν(m,g₂)/μ(m,g₂) + ν(m,g₂)/μ(m,g₂)^2 * dev_μ(m,g₂)) * (1 + exp(2 * im * Ω(m,g₂) * t)) - 2 * ν(m,g₂)/μ(m,g₂) * dev_Ω(m,g₂) * t* im * exp(2 * im * Ω(m,g₂) * t)
    dev_A[7,8] = im * (dev_ν(m,g₂)/μ(m,g₂) - ν(m,g₂)/μ(m,g₂)^2 * dev_μ(m,g₂)) * (exp(2 * im * Ω(m,g₂) * t)-1) - 2 * ν(m,g₂)/μ(m,g₂) * dev_Ω(m,g₂) * t* exp(2 * im * Ω(m,g₂) * t)
    dev_A[8,7] = im * (dev_ν(m,g₂)/μ(m,g₂) - ν(m,g₂)/μ(m,g₂)^2 * dev_μ(m,g₂)) * (exp(2 * im * Ω(m,g₂) * t)-1) - 2 * ν(m,g₂)/μ(m,g₂) * dev_Ω(m,g₂) * t* exp(2 * im * Ω(m,g₂) * t)
    dev_A[8,8] = (dev_ν(m,g₂)/μ(m,g₂)- ν(m,g₂)/μ(m,g₂)^2 * dev_μ(m,g₂)) * (1 + exp(2 * im * Ω(m,g₂) * t)) + 2 * ν(m,g₂)/μ(m,g₂) * dev_Ω(m,g₂) * t* im * exp(2 * im * Ω(m,g₂) * t)
    return dev_A
end

function get_dev_B(n,m,g₂,t)
    dev_B = zeros(Complex{Float64},8)
    dev_B[1] = 0
    dev_B[2] = 0
    dev_B[3] = 2 * (real(dev_β(n,g₂,t)) * cos(Ω(n,g₂) * t) - real(β(n,g₂,t)) * sin(Ω(n,g₂) * t) * dev_Ω(n,g₂) * t  - imag(dev_β(n,g₂,t)) * sin(Ω(n,g₂) * t) - imag(β(n,g₂,t)) * cos(Ω(n,g₂) * t) * t * dev_Ω(n,g₂) - (dev_β(n,g₂,t) *  ν(n,g₂)/μ(n,g₂) + β(n,g₂,t) *  (dev_ν(n,g₂)/μ(n,g₂) - ν(n,g₂)/μ(n,g₂)^2 * dev_μ(n,g₂)) - β(n,g₂,t) * ν(n,g₂)/μ(n,g₂) * im * dev_Ω(n,g₂) * t) *  exp(-im * Ω(n,g₂) * t))
                - (dev_β(n,g₂,t) + β(n,g₂,t) * im * t * dev_Ω(n,g₂))* exp(im * Ω(n,g₂) * t) 
                + (conj(dev_β(n,g₂,t)) - conj(β(n,g₂,t)) * im * t * dev_Ω(n,g₂)) * exp(-im * Ω(n,g₂) * t)
    dev_B[4] = 2 * (real(dev_β(n,g₂,t)) * sin(Ω(n,g₂) * t) + real(β(n,g₂,t)) * cos(Ω(n,g₂) * t) * dev_Ω(n,g₂) * t  + imag(dev_β(n,g₂,t)) * cos(Ω(n,g₂) * t) - imag(β(n,g₂,t)) * sin(Ω(n,g₂) * t) * t * dev_Ω(n,g₂) - im * (dev_β(n,g₂,t) *  ν(n,g₂)/μ(n,g₂) + β(n,g₂,t) *  (dev_ν(n,g₂)/μ(n,g₂) - ν(n,g₂)/μ(n,g₂)^2 * dev_μ(n,g₂)) - β(n,g₂,t) * ν(n,g₂)/μ(n,g₂) * im * dev_Ω(n,g₂) * t) *  exp(-im * Ω(n,g₂) * t))
                + (im * dev_β(n,g₂,t) - β(n,g₂,t) * t * dev_Ω(n,g₂))* exp(im * Ω(n,g₂) * t) 
                + (im * conj(dev_β(n,g₂,t)) + conj(β(n,g₂,t)) * t * dev_Ω(n,g₂)) * exp(-im * Ω(n,g₂) * t)
    dev_B[5] = 2/μ(n,g₂)^2 * dev_μ(n,g₂) * β(n,g₂,t) -2/μ(n,g₂) *dev_β(n,g₂,t)  +2/μ(m,g₂)^2 * conj(β(m,g₂,t)) * dev_μ(m,g₂) -2/μ(m,g₂) * conj(dev_β(m,g₂,t))
    dev_B[6] = -2 * im/μ(n,g₂)^2 * dev_μ(n,g₂) * β(n,g₂,t) + 2* im/μ(n,g₂) *dev_β(n,g₂,t) + 2 * im/μ(m,g₂)^2 * conj(β(m,g₂,t)) * dev_μ(m,g₂) -2* im/μ(m,g₂) * conj(dev_β(m,g₂,t))
    dev_B[7] = 2 * (real(dev_β(m,g₂,t)) * cos(Ω(m,g₂) * t) - real(β(m,g₂,t)) * sin(Ω(m,g₂) * t) * dev_Ω(m,g₂) * t  - imag(dev_β(m,g₂,t)) * sin(Ω(m,g₂) * t) - imag(β(m,g₂,t)) * cos(Ω(m,g₂) * t) * t * dev_Ω(m,g₂) - (conj(dev_β(m,g₂,t)) *  ν(m,g₂)/μ(m,g₂) + conj(β(m,g₂,t)) *  (dev_ν(m,g₂)/μ(m,g₂) - ν(m,g₂)/μ(m,g₂)^2 * dev_μ(m,g₂)) + conj(β(m,g₂,t)) * ν(m,g₂)/μ(m,g₂) * im * dev_Ω(m,g₂) * t) *  exp(im * Ω(m,g₂) * t))
                +(-conj(dev_β(m,g₂,t)) + conj(β(m,g₂,t)) * im* t* dev_Ω(m,g₂))* exp(-im* Ω(m,g₂)* t)
                + (dev_β(m,g₂,t) + β(m,g₂,t) * im* t* dev_Ω(m,g₂))* exp(im* Ω(m,g₂)* t)
    dev_B[8] = 2 * (real(dev_β(m,g₂,t)) * sin(Ω(m,g₂) * t) + real(β(m,g₂,t)) * cos(Ω(m,g₂) * t) * dev_Ω(m,g₂) * t  + imag(dev_β(m,g₂,t)) * cos(Ω(m,g₂) * t) - imag(β(m,g₂,t)) * sin(Ω(m,g₂) * t) * t * dev_Ω(m,g₂) + im * (conj(dev_β(m,g₂,t)) *  ν(m,g₂)/μ(m,g₂) + conj(β(m,g₂,t)) *  (dev_ν(m,g₂)/μ(m,g₂) - ν(m,g₂)/μ(m,g₂)^2 * dev_μ(m,g₂)) + conj(β(m,g₂,t)) * ν(m,g₂)/μ(m,g₂) * im * dev_Ω(m,g₂) * t) *  exp(im * Ω(m,g₂) * t))
                - (im* conj(dev_β(m,g₂,t)) + conj(β(m,g₂,t)) * t* dev_Ω(m,g₂))* exp(-im* Ω(m,g₂)* t)
                +(-im* dev_β(m,g₂,t) + β(m,g₂,t) * t* dev_Ω(m,g₂))* exp(im* Ω(m,g₂)* t)
    return -1/2 * dev_B
end

dev_C(n,m,g₂,t) = -(
    real(dev_β(n,g₂,t)) * real(β(n,g₂,t)) + 
    imag(dev_β(n,g₂,t)) * imag(β(n,g₂,t)) + 
    real(dev_β(m,g₂,t)) * real(β(m,g₂,t)) + 
    imag(dev_β(m,g₂,t)) * imag(β(m,g₂,t))
) + 1/2 * (
    (dev_ν(m,g₂)/μ(m,g₂) - ν(m,g₂)/μ(m,g₂)^2 * dev_μ(m,g₂)) * conj(β(m,g₂,t)^2) + 
    ν(m,g₂)/μ(m,g₂) * 2 * conj(β(m,g₂,t)) * conj(dev_β(m,g₂,t)) + 
    (dev_ν(n,g₂)/μ(n,g₂) - ν(n,g₂)/μ(n,g₂)^2 * dev_μ(n,g₂)) * β(n,g₂,t)^2 +
    ν(n,g₂)/μ(n,g₂) * (2 * dev_β(n,g₂,t) * β(n,g₂,t))
)

function dev_int_total(n,m,g₂,t)
    I = int_total(n,m,g₂,t)
    to_return = -im * (dev_ψ(n,g₂,t)-dev_ψ(m,g₂,t))
    cont_1 = -(dev_μ(n,g₂)/μ(n,g₂) + dev_μ(m,g₂)/μ(m,g₂))
    to_return += cont_1
    cont_2 =  -1/2 * tr(get_A(n,m,g₂,t)^(-1)*get_dev_A(n,m,g₂,t))
    to_return += cont_2
    cont_3 = dev_C(n,m,g₂,t)
    to_return += cont_3
    cont_4 = 1/2*(
        transpose(get_B(n,m,g₂,t)) * get_A(n,m,g₂,t)^(-1) * get_dev_B(n,m,g₂,t) + 
        transpose(get_dev_B(n,m,g₂,t)) * get_A(n,m,g₂,t)^(-1) * get_B(n,m,g₂,t) - 
        transpose(get_B(n,m,g₂,t)) * (get_A(n,m,g₂,t)^(-1) * get_dev_A(n,m,g₂,t) * get_A(n,m,g₂,t)^(-1)) * get_B(n,m,g₂,t)
    )
    to_return += cont_4
    return to_return * I
end


function get_mat(a, b, g₂, t)
    # This function constructs the time dependent density matrix, a is a_00 and b is a_01
    ρ= zeros(Complex{Float64}, 2, 2)
    ρ[1,1] = a * int_total(0,0,g₂,t)
    ρ[2,2] = (1-a) * int_total(1,1,g₂,t)
    ρ[1,2] = b * int_total(0,1,g₂,t)
    ρ[2,1] = conj(b) * int_total(1,0,g₂,t)
    @assert abs(tr(ρ) - 1.0) <  1e-5 "Trace of the density matrix is not 1"
    return ρ
end

function get_dev_mat(a, b, g₂, t)
    # This function constructs the time dependent density matrix, a is a_00 and b is a_01
    dev_ρ= zeros(Complex{Float64}, 2, 2)
    dev_ρ[1,1] = a * dev_int_total(0,0,g₂,t)
    dev_ρ[2,2] = (1-a) * dev_int_total(1,1,g₂,t)
    dev_ρ[1,2] = b * dev_int_total(0,1,g₂,t)
    dev_ρ[2,1] = conj(b) * dev_int_total(1,0,g₂,t)
    return dev_ρ
end  

function get_mat_general(ρ₀,g₂,t)
    # This function constructs the time dependent density matrix, a is a_00 and b is a_01
    ρ= zeros(Complex{Float64}, size(ρ₀)[1], size(ρ₀)[2])
    for i in 1:size(ρ₀,1)
        for j in 1:size(ρ₀,2)
            ρ[i,j] = int_total(i-1,j-1,g₂,t)*ρ₀[i,j]
        end
    end
    return ρ
end

function get_dev_mat_general(ρ₀,g₂,t)
    # This function constructs the time dependent density matrix, a is a_00 and b is a_01
    dev_ρ= zeros(Complex{Float64}, size(ρ₀)[1], size(ρ₀)[2])
    for i in 1:size(ρ₀,1)
        for j in 1:size(ρ₀,2)
            dev_ρ[i,j] = dev_int_total(i-1,j-1,g₂,t)*ρ₀[i,j]
        end
    end
    return dev_ρ
end
function get_L(ρ₀,g₂,t)
    # This function calculates the SLD
    dev_ρ = get_dev_mat_general(ρ₀,g₂,t)
    ρ = get_mat_general(ρ₀,g₂,t)
    L = quadgk(x -> exp(-x* ρ) * dev_ρ * exp(-x* ρ),  0, Inf)[1]
    return 2 * L
end

function QFI_general(ρ,g₂,t)
    # This function calculates the QFI for a given density matrix ρ according to the general formula
    L = get_L(ρ,g₂,t)
    QFI = tr(ρ* L^2)
    return QFI
end

# function QFI_Zhong(a,b,g₂,t)
#     # This function calculates the QFI for a given density matrix ρ according to Zhong et.al.
#     ρ = get_mat(a,b,g₂,t)
#     dev_ρ = get_dev_mat(a,b,g₂,t)
#     α_coef = ((4 * (1-2 * det(ρ)))/(1+2*sqrt(det(ρ)))-1)/(1-4 * det(ρ))
#     β_coef = (8/(1+2 * sqrt(det(ρ)))-1/det(ρ))/(1-4 * det(ρ))
#     QFI = α_coef * tr(dev_ρ)^2 - β_coef * tr(ρ * dev_ρ)^2
#     return QFI
# end

function QFI_Liu(a,b,g₂,t)
    # This function calculates the QFI for a given density matrix ρ according to Liu et.al.
    ρ = get_mat(a,b,g₂,t)
    dev_ρ = get_dev_mat(a,b,g₂,t)
    alpha = 1
    beta = ρ[1,1] - ρ[1,1]^2 - abs(ρ[1,2])^2
    # P denotes the purity of the state
    P = alpha - beta
    # Q = 2 *tr(dev_ρ)^2 + (dev_ρ[1,1] - 2 * ρ[1,1] * dev_ρ[1,1] - 2 * real(ρ[1,2]) * real(dev_ρ[1,2] - 2 * imag(ρ[1,2])*imag(dev_ρ[1,2])))^2 * tr(ρ)^(-1)
    L = 2 * dev_ρ + (dev_ρ[1,1] - 2 * ρ[1,1] * dev_ρ[1,1] - 2 * real(ρ[1,2]) * real(dev_ρ[1,2] - 2 * imag(ρ[1,2])*imag(dev_ρ[1,2])))* inv(ρ)
    Q = tr(ρ * L^2) 
    return Q
end




