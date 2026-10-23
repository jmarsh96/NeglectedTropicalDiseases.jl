struct MDAIntervention <: AbstractIntervention
    times::Vector{Int}
    coverages::Vector{Float64}
end

struct BednetIntervention <: AbstractIntervention
    times::Vector{Int}
    coverages::Vector{Float64}
end

function set_intervention!(model::AbstractTransmissionModel, intervention::MDAIntervention)
    append!(model.mda_times, intervention.times)
    return append!(model.mda_coverages, intervention.coverages)
end

function set_intervention!(
    model::AbstractTransmissionModel, intervention::BednetIntervention
)
    append!(model.bednet_times, intervention.times)
    return append!(model.bednet_coverages, intervention.coverages)
end
