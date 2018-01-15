"""

  given a path p and a flattened scattering transform flat, return the transform of that row.
"""
function findp(flat::Vector{T}, p::Vector{S}, layers::layeredTransform) where {T <: Number, S<:Integer}

end

"""
  p = computePath(layers::layeredTransform, layerM::Int64, λ::)

compute the path p that corresponds to location λ in the stType of transform
"""
function computePath(layers::layeredTransform, layerM::Int64, λ::Int64; stType::String="full")
  
end
function numChildren
