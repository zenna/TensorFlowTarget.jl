using Arrows
import Arrows: psl, supervised, traceprop!, UnknownArrow, simpletracewalk, trace_values, supervisedloss, AbValues
import Arrows: in_trace_values, out_trace_values

function test_psl(arr::Arrow, batch_size=64)
  orig = deepcopy(arr)
  invarr = aprx_invert(arr)
  pslarr = psl(invarr)
  superarr = supervised(orig, pslarr)
  suploss = supervisedloss(superarr)
  insizes = Dict(sprt => AbValues(:size => Size([batch_size, 10, 10])) for sprt in ▹(suploss))
  abtvals = traceprop!(suploss, insizes)
  nnettarr = first(filter(tarr -> deref(tarr) isa UnknownArrow, simpletracewalk(x->x, suploss)))
  @show insizes = [abtvals[tval][:size]  for tval in in_trace_values(nnettarr)]
  @show outsizes = [abtvals[tval][:size]  for tval in out_trace_values(nnettarr)]
  insizes, outsizes
end

function flatsizes(szs)
  @show szs = get.(szs)
  [reduce(*, sz[2:end]) for sz in szs]
end

# Unstacking
# "Separate t by channel: output[i] takes t[:, 0:sizes[i], :, :] channels"
# function split_channel(t, sizes, channel_dim=1, slice_dim=1)
#   channels = [size[channel_dim] for size in sizes]
#   if len(sizes) == 1
#     return t
#   else
#     outputs = []
#     c0 = 0
#     for c in channels
#       # print("Split ", c0, ":", c0 + c)
#       outputs.append(t.narrow(slice_dim, c0, c))
#       c0 = c
#     end
#   end
#   tuple(outputs)
# end

"Multiplayer Perceptron"
function mlp_template(args, insizes, outsizes)
  @show batch_size = get(insizes[1])[1]
  @show inflatsizes, outflatsizes = flatsizes(insizes), flatsizes(outsizes)
  nin = sum(inflatsizes)
  nout = sum(outflatsizes)
  flatargs = [reshape(args[i], (batch_size, inflatsizes[i])) for i = 1:length(args)]
  x = cat(2, flatargs...)
  # Combine inputs
  # Make parameters
  W = Variable(zeros(Float64, nin, nout))
  b = Variable(zeros(Float64, nout))
  y = nn.relu(x*W + b)
  lb = 1
  outputs = []
  for (i, sz) in enumerate(outsizes)
    ub = lb+outflatsizes[i] - 1
    @show lb, ub
    out = y[1:batch_size, lb:ub]
    push!(outputs, reshape(out, get(outsizes[i])))
    lb = ub + 1
  end
  outputs
end

using TensorFlow
function test_mlp_template()
  nin = rand(2:2)
  nout = rand(3:3)
  batch_size = 10
  insizes = [Size([batch_size, rand(3:4), rand(2:5)]) for i = 1:nin]
  outsizes = [Size([batch_size, rand(2:7), rand(2:3)]) for i = 1:nout]
  phs = [placeholder(Float64, shape=get(sz)) for sz in insizes]
  outputs = mlp_template(phs, insizes, outsizes)
  phs, outputs
  sess = Session()
  run(sess, global_variables_initializer())
  invalues = Dict(phs[i] => rand(get(insizes[i])...) for i = 1:length(insizes))
  run(sess, outputs, invalues)
end

# function test_mlp_run()
