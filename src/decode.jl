# Convert an arrow to a tensorflow graph
Args = Vector{<:tf.AbstractTensor}
conv(::PowArrow, args::Args)::Vector{Tensor} = [tf.pow(args...)]
conv(::LogArrow, args::Args)::Vector{Tensor} = [tf.log(args...)]
conv(::LogBaseArrow, args::Args)::Vector{Tensor} =
  [tf.log(args[1]) / tf.log(args[1])] # no logbase, use: log _{b}(x)=log _{k}(x)}/log _{k}(b)
conv(::AddArrow, args::Args)::Vector{Tensor} = [tf.add(args...)]
conv(::MulArrow, args::Args)::Vector{Tensor} = [tf.multiply(args...)]
conv(::DivArrow, args::Args)::Vector{Tensor} = [args[1] / args[2]]
conv(::SqrArrow, args::Args)::Vector{Tensor} = [tf.square(args...)]
conv(::SqrtArrow, args::Args)::Vector{Tensor} = [tf.sqrt(args...)]
conv(::MeanArrow, args::Args)::Vector{Tensor} = [tf.reduce_mean(tf.stack(args), axis=1)]
conv(::Arrows.ReduceVarArrow, args::Args)::Vector{Tensor} = [reduce_var(args)]
conv(::SinArrow, args::Args)::Vector{Tensor} = [tf.sin(args...)]
conv(::SubtractArrow, args::Args)::Vector{Tensor} = [args[1] - args[2]]
conv(::CosArrow, args::Args)::Vector{Tensor} = [tf.cos(args...)]
conv(::ASinArrow, args::Args)::Vector{Tensor} = [tf.asin(args...)]
conv(::ACosArrow, args::Args)::Vector{Tensor} = [tf.acos(args...)]
conv(::Arrows.MaxArrow, args::Args)::Vector{Tensor} = [tf.maximum(args...)]
conv{N}(arr::DuplArrow{N}, args::Args)::Vector{Tensor} = [args[1] for i = 1:N]
conv(::IdentityArrow, args::Args)::Vector{Tensor} = [tf.identity(args...)]
conv(::InvDuplArrow, args::Args)::Vector{Tensor} = [args[1]]
conv(arr::UnknownArrow, args::Args)::Vector{Tensor} = arr.func(args)
conv(::Arrows.AbsArrow, args::Args)::Vector{Tensor} = [tf.abs(args...)]
conv(::Arrows.ReshapeArrow, args::Args)::Vector{Tensor} = [tf.reshape(args...)]
# Automatic Broadcasting
conv(::Arrows.BroadcastArrow, args::Args)::Vector{Tensor} = [tf.identity(args...)]
function conv(::GatherNdArrow, args::Args)::Vector{Tensor}
  params, indices_ = args[1], args[2]
  indices_ = indices_ + convert(tf.Tensor{eltype(indices_)}, 1)
  [tf.gather_nd(params, indices_)]
end
function conv(::ScatterNdArrow, args)::Vector{Tensor}
  updates, indices_, shape = args[1], args[2], args[3]
  indices_ = indices_ + convert(tf.Tensor{eltype(indices_)}, 1)
  [tf.scatter_nd(indices_, updates, shape)]
end
conv(::NegArrow, args::Args)::Vector{Tensor} = [tf.neg(args...)]
conv(::ExpArrow, args::Args)::Vector = [tf.exp(args...)]
#conv(arr::Arrows.ReduceSumArrow, args)::Vector{Tensor} = [tf.reduce_sum(args...; axis=arr.axis)]
conv(::Arrows.LessThanArrow, args::Args)::Vector{Tensor} = [tf.less(args...)]
conv(::Arrows.GreaterThanArrow, args::Args)::Vector{Tensor} = [tf.greater(args...)]
conv(::Arrows.EqualArrow, args::Args)::Vector{Tensor} = [tf.equal(args...)]
function conv(::Arrows.IfElseArrow, args::Args)::Vector{Tensor}
  a, b, c = args
  [a .* c .+ b .* (1.0 - c)]
end
sanitizeconst(value::Tuple) = [value...]
sanitizeconst(value) = value
function conv(arr::SourceArrow, args::Args)::Vector{Tensor}
  [tf.constant(sanitizeconst(arr.value))]
end
function conv(carr::CompArrow, args::Args)::Vector{Tensor}
  @pre length(args) == num_in_ports(carr)
  variable_scope(string(name(carr))) do
    # @assert false
    interpret(conv, carr, args)
  end
end

conv(arr::Arrow, args::Vector) = (@pre isempty(args); conv(arr, Tensor[])) # Since, `interpret` may not type args

function conv(sarr::SubArrow, xs::Vector)
  conv(deref(sarr), xs)
end

"""
Construct `Graph` from `arr`

# Arguments:
- `intens`: tensors (typically placeholders) to act as inputs to arro
                 intens[i] = in_port(arrow, i)
  port_grab:
# Returns
- `graph`: `TensorFlow` `Graph` equivalent to `arr`
"""
function Graph(carr::CompArrow,
               graph,
               intens::Vector{<:Tensor})
  intens_wrapped = tf.identity(intens)
  out = interpret(conv, carr, intens_wrapped)
  return @NT(in=intens, out=out, graph=graph)
end

function Graph(carr::CompArrow,
               graph=Graph())
  # inputs need to be wrapped in identiy
  tf.as_default(graph) do
    intens = [placeholder(prt, graph) for prt in ▸(carr)]
    Graph(carr, graph, intens)
  end
end

"Converts a port to a placeholder"
placeholder(prt::Port, graph=tf.get_def_graph()) =
  tf.placeholder(Float32, name="inp_$(prt.port_id)")
  # tf.placeholder(Float32, name=name(prt).name)

# FIXME: Get the type for the placeholder from the type of the port
