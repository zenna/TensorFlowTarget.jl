"Conversion to and from tensorflow graphs"
module TensorFlowTarget

using Arrows
import Arrows: Arrow, Target, CompArrow
import AlioAnalysis: optimize
using PyCall
using NamedTuples
using Spec
import TensorFlow
const tf = TensorFlow

import TensorFlow: Operation, Graph, Tensor, Session, get_op
import Base: convert, hash, isequal, ==, take!

include("apply.jl")
include("extensions.jl")
# include("python.jl")
# include("types.jl")
# include("to_arrow.jl")
include("decode.jl")
include("optimize.jl")
include("templates/mlp.jl")

"Tensorflow target for dispatch"
struct TFTarget <: Target end

"Compiles `arr` into `Expr` tensorflow program"
compile(arr::Arrow, target::Type{TFTarget}) = Graph(arr)

export TFTarget,
       optimize,
       mlp_template
# graph_to_arrow,
#        PyTensor,
#        PyOperation,
#        PyGraph

end
