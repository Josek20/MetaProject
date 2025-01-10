
########################################################################
# Functions needed to hack rewriting systems
########################################################################

function get_value(x::OnlyNode)
    x.head == :integer && return(Int32(x.v))
    x.head == :float && return(Float32(x.v))
    x
end

function typeof_value(x::OnlyNode)
    x.head == :integer && return(Int32)
    x.head == :float && return(Float32)
    x.left == nullid && return(Symbol)
    return(OnlyNode)
end
