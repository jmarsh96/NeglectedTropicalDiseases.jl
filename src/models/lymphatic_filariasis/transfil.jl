mutable struct TransfilModel <: AbstractTransmissionModel
    time::Int

    # Host
    pop_size::Int
    tau::Float64          # host death rate (per month)
    bite_risk::Vector{Float64}
    bednet::Vector{Int}
    bednet_coverage::Float64
    bednet_reduction::Float64
    k::Float64            # gamma shape for bite risk heterogeneity
    importation_rate::Float64

    # Vector (mosquito)
    lambda::Float64       # bite rate (bites/mosquito/month)
    g::Float64            # L3 uptake efficiency
    larvae::Float64       # mean L3 density in mosquitoes
    sigma::Float64        # mosquito death rate
    kappa::Float64        # L3 development parameter
    r::Float64            # L3 development parameter
    species::Int          # 0 = Culex, 1 = Anopheles
    s_N::Float64          # probability of successful feeding (with bednet)
    d_N::Float64          # probability of mosquito death (with bednet)

    # Worm / parasite
    psi_1::Float64        # proportion L3 leaving mosquito per bite
    psi_2::Float64        # proportion of L3 that enter host
    s_2::Float64          # proportion of L3 entering host that develop to adults
    mu::Float64           # adult worm death rate
    gamma::Float64        # microfilariae death rate
    alpha::Float64        # mf production rate per female worm
    vth::Float64          # vector-to-host ratio

    # Individual-level state
    WF::Vector{Int}       # female worm counts
    WM::Vector{Int}       # male worm counts
    Mf::Vector{Float64}   # microfilariae density per 20 ml blood
    age::Vector{Float64}  # age in months

    # MDA state
    treatment_prob::Vector{Float64}
    ttreat::Vector{Int}           # months since last treatment
    excluded::Vector{Int}         # 1 if systematically excluded
    treatment_coverage::Float64
    systematic_compliance::Float64
    fecundity::Int                # months of worm infertility post-MDA
    sys_ex::Float64               # proportion systematically excluded
    prop_mf_killed::Float64
    prop_worm_killed::Float64
    mda_times::Vector{Int}
    mda_coverages::Vector{Float64}

    # Bednet schedule
    bednet_times::Vector{Int}
    bednet_coverages::Vector{Float64}

    # Prevalence time series
    worm_prev::Vector{Float64}
    mf_prev::Vector{Float64}
    worm_prev67::Vector{Float64}
    mf_prev67::Vector{Float64}
    pop67::Vector{Int}
end

function TransfilModel(pop_size::Int)
    # Fixed parameters (min == max in original)
    alpha = 0.58
    kappa = 4.395
    r     = 0.055

    # Drawn parameters
    k                = rand() * (1.6 - 0.01) + 0.01
    vth              = rand() * (180.0 - 1.0) + 1.0
    importation_rate = rand() * 0.0025

    treatment_coverage    = 0.01
    systematic_compliance = 0.35
    sys_ex                = 0.05

    # Per-individual draws
    age        = floor.(rand(pop_size) .* 1200.0)
    bite_risk  = rand(Gamma(k, 1.0 / k), pop_size)

    beta_shape1 = treatment_coverage * (1 - systematic_compliance) / systematic_compliance
    beta_shape2 = (1 - treatment_coverage) * (1 - systematic_compliance) / systematic_compliance
    treatment_prob = rand(Beta(beta_shape1, beta_shape2), pop_size)

    ttreat   = fill(15 * 12, pop_size)
    excluded = rand(Binomial(1, sys_ex), pop_size)
    bednet   = zeros(Int, pop_size)  # bednet_coverage = 0 initially

    TransfilModel(
        0,                        # time
        pop_size,
        0.00167,                  # tau
        bite_risk,
        bednet,
        0.0,                      # bednet_coverage
        0.97,                     # bednet_reduction
        k,
        importation_rate,
        10.0,                     # lambda
        0.37,                     # g
        5.0,                      # larvae
        5.0,                      # sigma
        kappa,
        r,
        1,                        # species (Anopheles)
        0.03,                     # s_N
        0.41,                     # d_N
        0.414,                    # psi_1
        0.32,                     # psi_2
        0.00275,                  # s_2
        0.0104,                   # mu
        0.1,                      # gamma
        alpha,
        vth,
        zeros(Int, pop_size),     # WF
        zeros(Int, pop_size),     # WM
        zeros(Float64, pop_size), # Mf
        age,
        treatment_prob,
        ttreat,
        excluded,
        treatment_coverage,
        systematic_compliance,
        9,                        # fecundity
        sys_ex,
        0.99,                     # prop_mf_killed
        0.35,                     # prop_worm_killed
        Int[],                    # mda_times
        Float64[],                # mda_coverages
        Int[],                    # bednet_times
        Float64[],                # bednet_coverages
        Float64[],                # worm_prev
        Float64[],                # mf_prev
        Float64[],                # worm_prev67
        Float64[],                # mf_prev67
        Int[],                    # pop67
    )
end

function timestep!(m::TransfilModel, n_months::Int)
    for _ in 1:n_months
        run_mda!(m)
        update_bednet!(m)
        worm_dynamics!(m)
        mf_dynamics!(m)
        larvae_dynamics!(m)
        calculate_prevalence!(m)
        aging!(m)
        importation!(m)
    end
end

function burn_in!(m::TransfilModel)
    for _ in 1:1200
        timestep!(m, 1)
        m.time = 0
    end
    # Retain only the final prevalence snapshot
    m.worm_prev   = [last(m.worm_prev)]
    m.mf_prev     = [last(m.mf_prev)]
    m.worm_prev67 = [last(m.worm_prev67)]
    m.mf_prev67   = [last(m.mf_prev67)]
    m.pop67       = [last(m.pop67)]
end

function worm_dynamics!(m::TransfilModel)
    for i in 1:m.pop_size
        age_prop = m.age[i] / (9.0 * 12.0)
        bite_rate = min(age_prop, 1.0)
        BNR = m.bednet[i] == 1 ? m.s_N : 1.0

        birth_rate = 0.5 * m.lambda * m.bite_risk[i] * m.vth * m.psi_1 * m.psi_2 *
                     m.s_2 * bite_rate * m.larvae * BNR

        male_death_rate   = m.WM[i] * m.mu
        female_death_rate = m.WF[i] * m.mu

        male_births   = birth_rate   > 0 ? rand(Poisson(birth_rate))   : 0
        female_births = birth_rate   > 0 ? rand(Poisson(birth_rate))   : 0
        male_deaths   = male_death_rate   > 0 ? rand(Poisson(male_death_rate))   : 0
        female_deaths = female_death_rate > 0 ? rand(Poisson(female_death_rate)) : 0

        m.WM[i] = max(0, m.WM[i] + male_births   - male_deaths)
        m.WF[i] = max(0, m.WF[i] + female_births - female_deaths)
    end
end

function mf_dynamics!(m::TransfilModel)
    for i in 1:m.pop_size
        m.Mf[i] *= (1.0 - m.gamma)
        m.WM[i] == 0 && continue
        m.ttreat[i] <= m.fecundity && continue
        m.Mf[i] += m.alpha * m.WF[i]
        m.Mf[i] = max(0.0, m.Mf[i])
    end
end

function larvae_dynamics!(m::TransfilModel)
    # Note: the original C++ switch-case has a fall-through bug — case 0 (Culex)
    # falls through to case 1 (Anopheles), so both species use the squared formula.
    # This translation faithfully replicates that behaviour.
    L3 = m.kappa .* (1.0 .- exp.(-m.Mf .* m.r ./ m.kappa)) .^
         (m.species == 0 ? 2 : 2)  # both cases → squared (faithful to original)

    mean_larvae  = sum(L3 .* m.bite_risk) / sum(m.bite_risk)
    lambda_bednet = m.lambda * (m.bednet_coverage * (m.s_N - 1.0) + 1.0)
    sigma_bednet  = m.sigma + m.sigma * m.d_N * m.bednet_coverage
    m.larvae = lambda_bednet * m.g * mean_larvae / (sigma_bednet + lambda_bednet * m.psi_1)
end

function calculate_prevalence!(m::TransfilModel)
    worm_pos    = 0.0
    worm67_pos  = 0.0
    mf67_pos    = 0.0
    cur_pop67   = 0

    for i in 1:m.pop_size
        age_i = m.age[i]
        age_i < 12 * 5 && continue
        (m.WM[i] > 0 || m.WF[i] > 0) && (worm_pos += 1)

        age_i < 12 * 6 && continue
        age_i > 12 * 8 && continue
        cur_pop67 += 1
        (m.WM[i] > 0 || m.WF[i] > 0) && (worm67_pos += 1)
        m.Mf[i] > 0                    && (mf67_pos   += 1)
    end

    mf_ind = rand(m.pop_size) .< (1.0 .- exp.(-m.Mf))

    push!(m.worm_prev,   worm_pos   == 0 ? 0.0 : worm_pos   / m.pop_size)
    push!(m.mf_prev,     mean(mf_ind))
    push!(m.worm_prev67, worm67_pos == 0 ? 0.0 : (cur_pop67 == 0 ? 0.0 : worm67_pos / cur_pop67))
    push!(m.mf_prev67,   mf67_pos   == 0 ? 0.0 : (cur_pop67 == 0 ? 0.0 : mf67_pos   / cur_pop67))
    push!(m.pop67, cur_pop67)

    # Adjust importation rate after an MDA timestep
    m.time ∉ m.mda_times && return
    length(m.worm_prev) < 2 && return
    prev_before = m.worm_prev[end - 1]
    prev_before == 0.0 && return
    prevalence_ratio = m.worm_prev[end] / prev_before
    m.importation_rate *= prevalence_ratio
end

function aging!(m::TransfilModel)
    m.age    .+= 1.0
    m.ttreat .+= 1
    m.time    += 1

    reset = (rand(m.pop_size) .< (1.0 .- exp.(-m.tau))) .| (m.age .> 1200.0)
    num_reset = sum(reset)
    num_reset == 0 && return

    m.age[reset]  .= 0.0
    m.WF[reset]   .= 0
    m.WM[reset]   .= 0
    m.Mf[reset]   .= 0.0
    m.ttreat[reset] .= 15 * 12

    m.bite_risk[reset] = rand(Gamma(m.k, 1.0 / m.k), num_reset)
    m.bednet[reset]    = rand(Binomial(1, m.bednet_coverage), num_reset)

    beta_shape1 = m.treatment_coverage * (1 - m.systematic_compliance) / m.systematic_compliance
    beta_shape2 = (1 - m.treatment_coverage) * (1 - m.systematic_compliance) / m.systematic_compliance
    m.treatment_prob[reset] = rand(Beta(beta_shape1, beta_shape2), num_reset)

    # Re-draw exclusion status for replaced hosts
    exclude_candidates = reset .& (rand(m.pop_size) .< m.sys_ex)
    m.excluded[exclude_candidates] .= 0
end

function importation!(m::TransfilModel)
    importation_ind = rand(m.pop_size) .< (1.0 .- exp.(-m.importation_rate))
    any(importation_ind) || return

    mean_larvae = 10.0
    new_worms = floor(Int, 0.5 * m.lambda * m.vth * m.psi_1 * m.psi_2 * m.s_2 * mean_larvae)
    m.WF[importation_ind] .= new_worms
    m.WM[importation_ind] .= new_worms
    m.Mf[importation_ind] .= 0.0
end

function run_mda!(m::TransfilModel)
    m.time ∈ m.mda_times || return

    for (t, cov) in zip(m.mda_times, m.mda_coverages)
        t != m.time && continue
        cov == m.treatment_coverage && continue
        # Coverage changed — redraw treatment probabilities
        m.treatment_coverage = cov
        beta_shape1 = m.treatment_coverage * (1 - m.systematic_compliance) / m.systematic_compliance
        beta_shape2 = (1 - m.treatment_coverage) * (1 - m.systematic_compliance) / m.systematic_compliance
        m.treatment_prob .= rand(Beta(beta_shape1, beta_shape2), m.pop_size)
    end

    U     = rand(m.pop_size)
    treat = U .< m.treatment_prob

    for i in 1:m.pop_size
        m.excluded[i] == 1  && continue
        treat[i]            || continue
        m.age[i] < 5 * 12   && continue

        m.WM[i]  = floor(Int, m.WM[i]  * (1.0 - m.prop_worm_killed))
        m.WF[i]  = floor(Int, m.WF[i]  * (1.0 - m.prop_worm_killed))
        m.Mf[i] *= (1.0 - m.prop_mf_killed)
        m.ttreat[i] = 0
    end

    larvae_dynamics!(m)
end

function update_bednet!(m::TransfilModel)
    m.time ∈ m.bednet_times || return

    for (t, cov) in zip(m.bednet_times, m.bednet_coverages)
        t != m.time && continue
        m.bednet_coverage = cov
        m.bednet .= rand(Binomial(1, cov), m.pop_size)
        return
    end
end
