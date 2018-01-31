# Convert an arrow to a tensorflow graph
Args = Vector{<:tf.AbstractTensor}

conv(::PowArrow, args::Args)::Args = [tf.pow(args...)]
conv(::LogArrow, args::Args)::Args = [tf.log(args...)]
conv(::LogBaseArrow, args::Args)::Args =
  [tf.log(args[1]) / tf.log(args[1])] # no logbase, use: log _{b}(x)=log _{k}(x)}/log _{k}(b)
conv(::AddArrow, args::Args)::Args = [tf.add(args...)]
conv(::MulArrow, args::Args)::Args = [tf.multiply(args...)]
conv(::DivArrow, args::Args)::Args = [args[1] / args[2]]
conv(::SqrArrow, args::Args)::Args = [tf.square(args...)]
conv(::SqrtArrow, args::Args)::Args = [tf.sqrt(args...)]
conv(::MeanArrow, args::Args)::Args = [tf.reduce_mean(tf.stack(args), axis=1)]

# Reductions
conv(::Arrows.ReduceVarArrow, arg::Args)::Args = [reduce_var(arg)]
conv(arr::Arrows.ReduceMeanArrow, arg::Args)::Args =
  [tf.reduce_mean(arg...; axis = arr.axis, keep_dims = arr.keepdims)]
conv(arr::Arrows.ReduceSumArrow, arg::Args)::Args =
  [tf.reduce_sum(arg...; axis = arr.axis, keep_dims=arr.keepdims)]

conv(::SinArrow, args::Args)::Args = [tf.sin(args...)]
conv(::SubtractArrow, args::Args)::Args = [args[1] - args[2]]
conv(::CosArrow, args::Args)::Args = [tf.cos(args...)]
conv(::ASinArrow, args::Args)::Args = [tf.asin(args...)]
conv(::ACosArrow, args::Args)::Args = [tf.acos(args...)]
conv(::Arrows.MaxArrow, args::Args)::Args = [tf.maximum(args...)]
conv{N}(arr::DuplArrow{N}, args::Args)::Args = [args[1] for i = 1:N]
conv(::IdentityArrow, args::Args)::Args = [tf.identity(args...)]
conv(::InvDuplArrow, args::Args)::Args = [args[1]]
conv(arr::UnknownArrow, args::Args)::Args = arr.func(args)
conv(::Arrows.AbsArrow, args::Args)::Args = [tf.abs(args...)]
conv(::Arrows.ReshapeArrow, args::Args)::Args = [tf.reshape(args...)]

# Automatic Broadcasting
conv(::Arrows.BroadcastArrow, args::Args)::Args = [tf.identity(args...)]
function conv(::GatherNdArrow, args::Args)::Args
  params, indices_ = args[1], args[2]
  indices_ = indices_ + convert(tf.Tensor{eltype(indices_)}, 1)
  [tf.gather_nd(params, indices_)]
end
function conv(::ScatterNdArrow, args)::Args
  updates, indices_, shape = args[1], args[2], args[3]
  indices_ = indices_ + convert(tf.Tensor{eltype(indices_)}, 1)
  [tf.scatter_nd(indices_, updates, shape)]
end
conv(::NegArrow, args::Args)::Args = [tf.neg(args...)]
conv(::ExpArrow, args::Args)::Vector = [tf.exp(args...)]
conv(::Arrows.LessThanArrow, args::Args)::Args = [tf.less(args...)]
conv(::Arrows.GreaterThanArrow, args::Args)::Args = [tf.greater(args...)]
conv(::Arrows.EqualArrow, args::Args)::Args = [tf.equal(args...)]
function conv(::Arrows.IfElseArrow, args::Args)::Args
  a, b, c = args
  [a .* c .+ b .* (1.0 - c)]
end
sanitizeconst(value::Tuple) = [value...]
sanitizeconst(value) = value
function conv(arr::SourceArrow, args::Args)::Args
  [tf.constant(sanitizeconst(arr.value))]
end
function conv(carr::CompArrow, args::Args)::Args
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
