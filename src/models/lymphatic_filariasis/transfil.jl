# TRANSFIL (Irvine et al. 2015) with monthly time steps.

"""
    TransfilParams

Fixed (non-inferred) TRANSFIL parameters. Rates are per month, ages in months.
"""
Base.@kwdef struct TransfilParams
    # Host
    tau::Float64 = 0.00167 # host death rate
    max_age::Float64 = 1200.0 # hosts die at 100 years
    a_max::Float64 = 9.0 * 12 # age at which exposure saturates, h(a) = min(a / a_max, 1)

    # Vector
    lambda::Float64 = 10.0 # bites per mosquito
    g::Float64 = 0.37 # proportion of mosquitoes picking up infection
    sigma::Float64 = 5.0 # mosquito death rate
    species::Int = 1 # 0 = Culex, 1 = Anopheles
    kappa::Float64 = 4.395 # L3 uptake saturation (Anopheles)
    r::Float64 = 0.055 # L3 uptake rate (Anopheles)
    kappa_culex::Float64 = 4.406
    r_culex::Float64 = 0.019
    s_N::Float64 = 0.03 # probability of successful feed with a bednet
    d_N::Float64 = 0.41 # probability of death with a bednet

    # Worm
    psi_1::Float64 = 0.414 # proportion of L3 leaving mosquito per bite
    psi_2::Float64 = 0.32 # proportion of L3 leaving mosquito that enter host
    s_2::Float64 = 0.00275 # proportion of L3 entering host that develop into adults
    mu::Float64 = 0.0104 # adult worm death rate
    gamma::Float64 = 0.1 # mf death rate
    initial_larvae::Float64 = 5.0

    # MDA (ivermectin + albendazole defaults, Stolk et al. 2018)
    prop_mf_killed::Float64 = 0.99
    fecundity::Int = 9 # months without mf production after treatment
    sys_ex::Float64 = 0.05 # proportion never treated
    mda_min_age::Float64 = 5.0 * 12

    # Importation
    importation_mode::Symbol = :proportional # or :constant
    importation_worms::Float64 = 10.0 # mean worms of each sex in an imported infection (>= 1)

    # Prevalence outputs
    baseline_min_age::Float64 = 15.0 * 12 # age band compared with the baseline map
    tas_age_range::Tuple{Float64,Float64} = (6.0 * 12, 8.0 * 12) # [6, 8) years
    mf_min_age::Float64 = 5.0 * 12
end

mutable struct TransfilModel{R<:AbstractRNG} <: AbstractTransmissionModel
    p::TransfilParams
    rng::R
    time::Int

    # Inferred / sampled parameters
    pop_size::Int
    vth::Float64 # vector to host ratio
    k::Float64 # gamma shape of individual bite risk
    alpha::Float64 # mf production per female worm per month (per 20 uL)
    importation_rate::Float64 # baseline importation rate per host per month
    prop_worm_killed::Float64 # MDA macrofilaricidal efficacy
    rho::Float64 # systematic adherence (correlation between rounds)

    # Vector state
    larvae::Float64
    bednet_coverage::Float64

    # Individual state
    age::Vector{Float64}
    bite_risk::Vector{Float64}
    WF::Vector{Int}
    WM::Vector{Int}
    Mf::Vector{Float64}
    bednet::BitVector
    excluded::BitVector
    adherence::Vector{Float64} # u_i ~ U(0, 1), see run_mda!
    ttreat::Vector{Int}

    # Interventions
    mda_times::Vector{Int}
    mda_coverages::Vector{Float64}
    bednet_times::Vector{Int}
    bednet_coverages::Vector{Float64}

    # Reference prevalence for the importation scaling (set at the end of burn-in)
    prev_ref::Float64

    # Recorded output
    record_times::Set{Int}
    records::Vector{NamedTuple{(:time, :ag_baseline, :ag_tas, :mf, :n_tas, :ag_all, :total_worms),
                               Tuple{Int,Float64,Float64,Float64,Int,Float64,Int}}}
end

"""
    TransfilModel(pop_size; vth, k, importation_rate, prop_worm_killed = 0.35,
                  rho = 0.35, params = TransfilParams(), rng = Random.default_rng())

Create a population of `pop_size` hosts with uniformly distributed ages and no infection.
"""
function TransfilModel(pop_size::Integer; vth::Real, k::Real, importation_rate::Real,
                       alpha::Real = 0.58, prop_worm_killed::Real = 0.35, rho::Real = 0.35,
                       params::TransfilParams = TransfilParams(),
                       rng::AbstractRNG = Random.default_rng())
    n = Int(pop_size)
    age = floor.(rand(rng, n) .* params.max_age)
    bite_risk = rand(rng, Gamma(k, 1 / k), n)
    adherence = rand(rng, n)
    excluded = rand(rng, n) .< params.sys_ex
    TransfilModel(params, rng, 0, n, Float64(vth), Float64(k), Float64(alpha), Float64(importation_rate),
                  Float64(prop_worm_killed), Float64(rho), params.initial_larvae, 0.0,
                  age, bite_risk, zeros(Int, n), zeros(Int, n), zeros(n), falses(n),
                  excluded, adherence, fill(15 * 12, n),
                  Int[], Float64[], Int[], Float64[], NaN, Set{Int}(),
                  NamedTuple{(:time, :ag_baseline, :ag_tas, :mf, :n_tas, :ag_all, :total_worms),
                             Tuple{Int,Float64,Float64,Float64,Int,Float64,Int}}[])
end

exposure(m::TransfilModel, i) = min(m.age[i] / m.p.a_max, 1.0)

"""
    timestep!(m, n_months; interventions = true)

Advance the model `n_months` months. Prevalence is recorded at the start of a month
whose index is in `m.record_times` (so time `t` is the state before any MDA at `t`).
"""
function timestep!(m::TransfilModel, n_months::Integer; interventions::Bool = true)
    for _ in 1:n_months
        m.time in m.record_times && record_prevalence!(m)
        if interventions
            run_mda!(m)
            update_bednet!(m)
        end
        worm_dynamics!(m)
        mf_dynamics!(m)
        larvae_dynamics!(m)
        aging!(m)
        importation!(m)
        m.time += 1
    end
    return m
end

"""
    burn_in!(m; months = 1200)

Run without interventions to endemic equilibrium, then reset the clock to 0 and set the
pre-intervention reference prevalence used by the importation scaling.
"""
function burn_in!(m::TransfilModel; months::Integer = 1200)
    timestep!(m, months; interventions = false)
    m.time = 0
    m.prev_ref = ag_prevalence_all(m)
    return m
end

function worm_dynamics!(m::TransfilModel)
    p = m.p
    base = 0.5 * p.lambda * m.vth * p.psi_1 * p.psi_2 * p.s_2 * m.larvae
    @inbounds for i in 1:m.pop_size
        bnr = m.bednet[i] ? p.s_N : 1.0
        birth_rate = base * m.bite_risk[i] * exposure(m, i) * bnr
        male_births = birth_rate > 0 ? rand(m.rng, Poisson(birth_rate)) : 0
        female_births = birth_rate > 0 ? rand(m.rng, Poisson(birth_rate)) : 0
        male_deaths = m.WM[i] > 0 ? rand(m.rng, Poisson(p.mu * m.WM[i])) : 0
        female_deaths = m.WF[i] > 0 ? rand(m.rng, Poisson(p.mu * m.WF[i])) : 0
        m.WM[i] = max(0, m.WM[i] + male_births - male_deaths)
        m.WF[i] = max(0, m.WF[i] + female_births - female_deaths)
    end
end

function mf_dynamics!(m::TransfilModel)
    p = m.p
    @inbounds for i in 1:m.pop_size
        m.Mf[i] *= 1 - p.gamma
        if m.WM[i] > 0 && m.ttreat[i] > p.fecundity
            m.Mf[i] += m.alpha * m.WF[i]
        end
    end
end

function larval_uptake(p::TransfilParams, mf)
    if p.species == 0
        return p.kappa_culex * (1 - exp(-p.r_culex * mf / p.kappa_culex))
    else
        return p.kappa * (1 - exp(-p.r * mf / p.kappa))^2
    end
end

function larvae_dynamics!(m::TransfilModel)
    p = m.p
    num = 0.0
    den = 0.0
    @inbounds for i in 1:m.pop_size
        num += larval_uptake(p, m.Mf[i]) * m.bite_risk[i]
        den += m.bite_risk[i]
    end
    mean_larvae = num / den
    c = m.bednet_coverage
    lambda_N = p.lambda * (c * p.s_N + (1 - c))
    sigma_N = p.sigma + p.lambda * c * p.d_N
    m.larvae = lambda_N * p.g * mean_larvae / (sigma_N + lambda_N * p.psi_1)
end

function aging!(m::TransfilModel)
    p = m.p
    death_prob = 1 - exp(-p.tau)
    @inbounds for i in 1:m.pop_size
        m.age[i] += 1
        m.ttreat[i] += 1
        if rand(m.rng) < death_prob || m.age[i] > p.max_age
            m.age[i] = 0.0
            m.WF[i] = 0
            m.WM[i] = 0
            m.Mf[i] = 0.0
            m.ttreat[i] = 15 * 12
            m.bite_risk[i] = rand(m.rng, Gamma(m.k, 1 / m.k))
            m.bednet[i] = rand(m.rng) < m.bednet_coverage
            m.excluded[i] = rand(m.rng) < p.sys_ex
            m.adherence[i] = rand(m.rng)
        end
    end
end

# Touloupou et al. (2022): importation falls in proportion to prevalence after intervention.
function current_importation_rate(m::TransfilModel)
    m.p.importation_mode === :constant && return m.importation_rate
    isnan(m.prev_ref) && return m.importation_rate # burn-in
    m.prev_ref <= 0 && return 0.0
    return m.importation_rate * min(ag_prevalence_all(m) / m.prev_ref, 1.0)
end

# An imported infection adds at least one worm of each sex, with mean `importation_worms`
# per sex. Hosts are exposed in proportion to h(a).
function importation!(m::TransfilModel)
    p = m.p
    rate = current_importation_rate(m)
    rate <= 0 && return
    extra = max(p.importation_worms - 1, 0.0)
    @inbounds for i in 1:m.pop_size
        rand(m.rng) < 1 - exp(-rate * exposure(m, i)) || continue
        m.WF[i] += 1 + (extra > 0 ? rand(m.rng, Poisson(extra)) : 0)
        m.WM[i] += 1 + (extra > 0 ? rand(m.rng, Poisson(extra)) : 0)
    end
end

"""
Controlled correlation scheme (Dyson et al. 2017): individual treatment probabilities
p_i ~ Beta(c(1-rho)/rho, (1-c)(1-rho)/rho) give coverage c and correlation rho between
rounds. p_i is the Beta quantile of a fixed u_i, so individuals keep their rank when
coverage changes between rounds.
"""
function treatment_probability(coverage, rho, u)
    rho <= 0 && return coverage
    c = clamp(coverage, 1e-6, 1 - 1e-6)
    return quantile(Beta(c * (1 - rho) / rho, (1 - c) * (1 - rho) / rho), u)
end

function run_mda!(m::TransfilModel)
    idx = findfirst(==(m.time), m.mda_times)
    idx === nothing && return
    p = m.p
    coverage = m.mda_coverages[idx]
    keep = 1 - m.prop_worm_killed
    @inbounds for i in 1:m.pop_size
        m.excluded[i] && continue
        m.age[i] < p.mda_min_age && continue
        rand(m.rng) < treatment_probability(coverage, m.rho, m.adherence[i]) || continue
        m.WM[i] = m.WM[i] > 0 ? rand(m.rng, Binomial(m.WM[i], keep)) : 0
        m.WF[i] = m.WF[i] > 0 ? rand(m.rng, Binomial(m.WF[i], keep)) : 0
        m.Mf[i] *= 1 - p.prop_mf_killed
        m.ttreat[i] = 0
    end
    larvae_dynamics!(m)
end

function update_bednet!(m::TransfilModel)
    idx = findfirst(==(m.time), m.bednet_times)
    idx === nothing && return
    m.bednet_coverage = m.bednet_coverages[idx]
    @inbounds for i in 1:m.pop_size
        m.bednet[i] = rand(m.rng) < m.bednet_coverage
    end
end

"""
    prevalence(m)

Current prevalences. Antigen (Ag) positivity is the presence of any adult worm; mf
positivity is a Poisson(Mf) > 0 draw (Irvine 2015, Collyer 2020 Table 2).
"""
function prevalence(m::TransfilModel)
    p = m.p
    lo, hi = p.tas_age_range
    n_base = 0; pos_base = 0
    n_tas = 0; pos_tas = 0
    n_mf = 0; pos_mf = 0
    pos_all = 0
    total_worms = 0
    @inbounds for i in 1:m.pop_size
        a = m.age[i]
        worms = m.WF[i] + m.WM[i]
        total_worms += worms
        ag = worms > 0
        pos_all += ag
        if a >= p.baseline_min_age
            n_base += 1; pos_base += ag
        end
        if lo <= a < hi
            n_tas += 1; pos_tas += ag
        end
        if a >= p.mf_min_age
            n_mf += 1
            pos_mf += m.Mf[i] > 0 && rand(m.rng) < 1 - exp(-m.Mf[i])
        end
    end
    ratio(x, n) = n == 0 ? 0.0 : x / n
    return (ag_baseline = ratio(pos_base, n_base), ag_tas = ratio(pos_tas, n_tas),
            mf = ratio(pos_mf, n_mf), n_tas = n_tas, ag_all = ratio(pos_all, m.pop_size),
            total_worms = total_worms)
end

"""Proportion of all hosts with at least one adult worm."""
function ag_prevalence_all(m::TransfilModel)
    pos = 0
    @inbounds for i in 1:m.pop_size
        pos += (m.WF[i] + m.WM[i]) > 0
    end
    return pos / m.pop_size
end

record_prevalence!(m::TransfilModel) = push!(m.records, (time = m.time, prevalence(m)...))

"""
    set_record_times!(m, times)

Record prevalence at the start of each month in `times` (relative to the end of burn-in).
"""
function set_record_times!(m::TransfilModel, times)
    m.record_times = Set{Int}(times)
    return m
end
