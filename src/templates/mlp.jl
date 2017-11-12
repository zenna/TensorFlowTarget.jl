"(10, 2, 4) -> (10, 8)"
function flatsizes(szs)
  @show szs = get.(szs)
  [reduce(*, sz[2:end]) for sz in szs]
end

"Multiplayer Perceptron"
function mlp_template(args, insizes, outsizes)
  @show args
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
  y = nn.elu(x*W + b)
  y = y + 1
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
