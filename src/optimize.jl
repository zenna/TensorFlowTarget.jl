@enum Stage Pre Run Post

apl(f, data) = f(data)

"""
Optimization.
# Arguments
- `step!`: takes a gradient step and returns the loss
- `writer`: Summary writer to log results to tensorboardX
- `close_writer`: Close writer after finish?
- `pre_callbacks`: functions/generators called before optimization
- `callbacks`: functions called with data every iteration, e.g for viz
- `post_callbacks`: functions/generators called after optimization
- `maxiters`: num of iterations
- `cont`: function to determine when to stop (overrides maxiters)
- `resetlog`: reset log data after every iteration if true
- `logdir`: directory to store data/logs (used by callbacks)
- `optimize`: optimize? (compute grads/change weights)
- `start_i`: what index is this starting at (used by callbacks)
"""
function optimize(step!::Function;
                  pre_callbacks=[],
                  callbacks=[],
                  post_callbacks=[],
                  cont=data -> data.i < 100000,
                  resetlog::Bool=true,
                  logdir::String="",
                  optimize::Bool=true,
                  start_i::Integer=0)
  i = 0
  cb_data = @NT(start_i=start_i, i=i, Stage=Pre)

  # Called once before optimization
  foreach(cb->apl(cb, cb_data), pre_callbacks)

  while cont(cb_data)
    if optimize
      @show cur_loss = step!()
    end

    cb_data = @NT(start_i=start_i, i=i, Stage=Run, loss=cur_loss)
    foreach(cb->apl(cb, cb_data), callbacks)
    i += 1
    # resetlog && reset_log()
  end
  # Post Callbacks
  cb_data = @NT(start_i=start_i, i=i, Stage=Post)
  foreach(cb->apl(cb, cb_data), post_callbacks)
end

take!(x::Real) = x
take!(x::Array{<:Real}) = x
take!(f::Function) = f()

take1(rep) = collect(Base.Iterators.take(rep, 1))[1]

"""
argmin_θ(ϵprt): find θ which minimizes ϵprt

# Arguments
- `callbacks`: functions to be called
- `over`: ports to optimize over
- `ϵprt`: out port to minimize
- `init`: initial input values
# Result
- `θ_optim`: minimal value of ϵprt found
- `argmin`: argmin of `over` found
"""
function optimize(carr::CompArrow,
                  ϵprt::Port,
                  iters,
                  target=Type{TFTarget};
                  kwargs...)
  length(iters) == length(▸(carr)) || throw(ArgumentError("Need iteraator foreach in port"))
  graph = tf.Graph()
  sess = tf.Session(graph)

  summary = TensorFlow.summary
  # weight_summary = summary.histogram("Parameters", weights)

  # Create a summary writer

  tf.as_default(graph) do
    @show collect(TensorFlow.get_operations(graph))
    intens = Tensor[]
    phs = Dict{Tensor, Int}()
    for (i, prt) in enumerate(▸(carr))
      ph = tf.placeholder(Float64, name="inp_$i")
      push!(intens, ph)
      phs[ph] = i
      # end
    end
    tfarr = Graph(carr, graph, intens)
    ϵid = findfirst(◂(ϵprt.arrow), ϵprt)
    losses = tfarr.out[ϵid]
    meanloss = TensorFlow.reduce_mean(losses)
    optimizer = train.AdamOptimizer()
    minimize_op = train.minimize(optimizer, meanloss)
    alpha_summmary = summary.scalar("Learning rate", meanloss)
    merged_summary_op = summary.merge_all()
    summary_writer = summary.FileWriter("./my_log_dir")
    @show length(collect(TensorFlow.get_operations(graph)))
    run(sess, global_variables_initializer())
    i = 44
    function step!()
      phsvalmap = Dict(ph => take1(iters[id]) for (ph, id) in phs)
      cur_loss, _ = run(sess, [meanloss, minimize_op], phsvalmap)
      summaries = run(sess, merged_summary_op, phsvalmap)
      write(summary_writer, summaries, i)
      cur_loss
      i = i + 1
      cur_loss
    end
    return optimize(step!; kwargs...)
  end
end
