mutable struct MyReplayBuffer
    capacity::Int
    buffer::Vector{Tuple}
    pointer::Int
end

MyReplayBuffer(capacity) = MyReplayBuffer(capacity, Vector{Tuple{Array{Float32}, Int, Float32, Array{Float32}, Bool}}(), 1)

function add2buffer!(buffer::MyReplayBuffer, experience)
    if buffer.capacity <= length(buffer.buffer)
        buffer.buffer[buffer.pointer] = experience
        buffer.pointer = (buffer.pointer % buffer.capacity) + 1
    else
        push!(buffer.buffer, experience)
    end
end

function sample(buffer::MyReplayBuffer, batch_size::Int)
    sampled_indices = rand(1:length(buffer.buffer), batch_size)
    # return [buffer.buffer[i] for i in sampled_indices]
    return buffer.buffer[sampled_indices]
end
