abstract type AbstractBuffer end

mutable struct BestTrajectoryBuffer{T} <: AbstractBuffer
    best_trajectory::T
end


function best!(buff::BestTrajectoryBuffer{Trajectory}, trajectory::Trajectory)
    bsize = exp_size(buff.best_trajectory.next_states[end])
    tsize = exp_size(trajectory.next_states[end])
    if bsize < tsize
        return false
    elseif bsize == tsize && length(buff.best_trajectory.next_states) <= length(trajectory.next_states)
        return false
    else
        buff.best_trajectory = trajectory
        return true
    end
end