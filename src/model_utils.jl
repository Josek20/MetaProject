

const const_left = [1, 0]
const const_right = [0, 1]
const const_one = [0, 0]
const const_third = [1, 1]
const const_both = [1 0; 0 1]
const const_nth = [1 0 1; 0 1 1]


my_sigmoid(x, k = 0.01, m = 0) = 1 / (1 + exp(-k * (x - m)))


function get_head_and_args(ex::NodeID)
    node = nc[ex]
    fun_name = node.head
    (;args=[node.left, node.right], fun_name=fun_name)
end


function get_head_and_args(ex::Expr)
    if ex.head == :call
        fun_name, args =  ex.args[1], ex.args[2:end]
    elseif ex.head in all_symbols
        fun_name = ex.head
        args = ex.args
    else
        error("unknown head $(ex.head)")
    end
    (;args=args, fun_name=fun_name)
end


function get_leaf_args(ex::NodeID)
    node = nc[ex]
    symbol_index = node.head ∈ (:integer, :float) ? :Number : node.head
    encoding_value = node.head ∈ (:integer, :float) ? my_sigmoid(node.v) : 1
    (;symbol_index=symbol_index,encoding_value=encoding_value)
end


function get_leaf_args(ex::Union{Symbol, Int})
    symbol_index = ex isa Symbol ? ex : :Number
    encoding_value = ex isa Symbol ? 1 : my_sigmoid(node.v)
    (;symbol_index=symbol_index,encoding_value=encoding_value)
end
    
get_inference_type(node::Expr) = Expr
get_inference_type(node::Symbol) = Symbol
get_inference_type(node::Number) = Symbol

function get_inference_type(node::NodeID)
    inference_type = nc[node].iscall ? Expr : Symbol
    return(inference_type)
end


