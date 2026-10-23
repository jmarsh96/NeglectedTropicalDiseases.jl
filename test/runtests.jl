using NeglectedTropicalDiseases
using NeglectedTropicalDiseases: run_mda!, importation!, aging!, ag_prevalence_all
using Random, Statistics, Test

function model(; n=1000, vth=100.0, k=0.3, imp=0.0, seed=1, kw...)
    return TransfilModel(n; vth=vth, k=k, importation_rate=imp, rng=Xoshiro(seed), kw...)
end

@testset "TRANSFIL" begin
    @testset "no larvae and no importation stays uninfected" begin
        m = model(params=TransfilParams(initial_larvae=0.0))
        timestep!(m, 240)
        @test sum(m.WF) + sum(m.WM) == 0
        @test prevalence(m).ag_all == 0
    end

    @testset "equilibrium prevalence increases with V/H" begin
        mean_prev(vth) = mean(1:4) do s
            m = model(vth=vth, k=0.1, seed=s)
            burn_in!(m)
            prevalence(m).ag_baseline
        end
        @test mean_prev(30.0) < mean_prev(80.0) < mean_prev(150.0)
    end

    @testset "same seed reproduces, different seeds differ" begin
        run(seed) = (m=model(seed=seed, imp=1e-3); burn_in!(m; months=240); copy(m.WF))
        @test run(7) == run(7)
        @test run(7) != run(8)
    end

    @testset "MDA kills worms by binomial thinning" begin
        m = model(n=20_000, prop_worm_killed=0.35, rho=0.0)
        fill!(m.WF, 2)
        fill!(m.WM, 2)
        fill!(m.age, 20 * 12)
        fill!(m.excluded, false)
        set_intervention!(m, MDAIntervention([0], [1.0 - 1e-9]))
        run_mda!(m)
        @test mean(m.WF) ≈ 2 * 0.65 atol = 0.03 # floor(2 * 0.65) would give 1
        @test any(==(2), m.WF)
    end

    @testset "MDA respects coverage, age and exclusion" begin
        m = model(n=20_000, rho=0.0)
        fill!(m.WF, 10)
        fill!(m.WM, 10)
        fill!(m.excluded, false)
        m.age .= ifelse.(1:m.pop_size .<= 10_000, 2.0 * 12, 30.0 * 12)
        set_intervention!(m, MDAIntervention([0], [0.65]))
        run_mda!(m)
        treated = m.ttreat .== 0
        @test !any(treated[1:10_000])
        @test mean(treated[10_001:end]) ≈ 0.65 atol = 0.02
    end

    @testset "controlled correlation between rounds (Dyson 2017)" begin
        rounds(covs; rho=0.5) = begin
            m = model(n=40_000, rho=rho)
            fill!(m.age, 30.0 * 12)
            fill!(m.excluded, false)
            set_intervention!(m, MDAIntervention(collect(0:(length(covs) - 1)), covs))
            map(0:(length(covs) - 1)) do t
                m.time = t
                m.ttreat .= 100
                run_mda!(m)
                m.ttreat .== 0
            end
        end
        r1, r2 = rounds([0.6, 0.6])
        @test mean(r1) ≈ 0.6 atol = 0.02
        @test cor(r1, r2) ≈ 0.5 atol = 0.03
        r1, r2 = rounds([0.6, 0.8])
        @test mean(r2) ≈ 0.8 atol = 0.02
        @test cor(r1, r2) > 0.3
        r1, r2 = rounds([0.6, 0.6]; rho=0.0)
        @test abs(cor(r1, r2)) < 0.03
    end

    @testset "systematic exclusion stays near sys_ex" begin
        m = model(n=5000, params=TransfilParams(sys_ex=0.05))
        for _ in 1:1200
            aging!(m)
        end
        @test mean(m.excluded) ≈ 0.05 atol = 0.015
    end

    @testset "imported infections are fertile and do not depend on V/H" begin
        imported(vth; worms=10.0) = begin
            m = model(
                n=20_000,
                vth=vth,
                imp=0.05,
                params=TransfilParams(importation_worms=worms),
            )
            fill!(m.age, 30.0 * 12)
            fill!(m.WF, 3)
            fill!(m.WM, 3)
            importation!(m)
            hit = m.WF .> 3
            @test all(m.WM[hit] .> 3) # at least one worm of each sex
            @test all(>=(3), m.WF) # adds to, never overwrites, the burden
            mean(m.WF[hit] .- 3)
        end
        @test imported(10.0) ≈ 10 atol = 0.5
        @test imported(150.0) ≈ 10 atol = 0.5
        @test imported(10.0; worms=1.0) == 1
    end

    @testset "proportional importation scales with prevalence reduction" begin
        m = model(imp=1e-3, vth=120.0)
        burn_in!(m)
        @test m.prev_ref > 0
        fill!(m.WF, 0)
        fill!(m.WM, 0)
        @test NeglectedTropicalDiseases.current_importation_rate(m) == 0
        mc = model(imp=1e-3, params=TransfilParams(importation_mode=:constant))
        burn_in!(mc)
        fill!(mc.WF, 0)
        fill!(mc.WM, 0)
        @test NeglectedTropicalDiseases.current_importation_rate(mc) == 1e-3
    end

    @testset "bednets reduce transmission" begin
        prev(cov) = mean(1:4) do s
            m = model(vth=120.0, k=0.3, seed=s, imp=1e-4)
            burn_in!(m)
            set_intervention!(m, BednetIntervention([0], [cov]))
            timestep!(m, 120)
            prevalence(m).ag_baseline
        end
        @test prev(0.8) < prev(0.0)
    end

    @testset "prevalence age bands and recording" begin
        m = model(n=4)
        m.age .= [12.0 * 6.5, 12.0 * 8.0, 12.0 * 20, 12.0 * 3]
        m.WF .= [1, 1, 0, 1]
        pr = prevalence(m)
        @test pr.n_tas == 1 && pr.ag_tas == 1.0 # age 8.0 is outside [6, 8)
        @test pr.ag_baseline == 0.0 # only the 20-year-old is >= 15
        @test pr.ag_all == 0.75
        set_record_times!(m, [0, 2])
        timestep!(m, 3)
        @test [r.time for r in m.records] == [0, 2]
    end

    @testset "mf production scales with alpha" begin
        mf(alpha) = (
            m=model(n=10, alpha=alpha);
            fill!(m.WF, 10);
            fill!(m.WM, 1);
            m.ttreat.=100;
            NeglectedTropicalDiseases.mf_dynamics!(m);
            m.Mf[1]
        )
        @test mf(0.58) ≈ 5.8 && mf(0.2) ≈ 2.0
    end

    @testset "Culex and Anopheles uptake differ" begin
        pa = TransfilParams(species=1)
        pc = TransfilParams(species=0)
        @test NeglectedTropicalDiseases.larval_uptake(pa, 10.0) !=
            NeglectedTropicalDiseases.larval_uptake(pc, 10.0)
    end
end
