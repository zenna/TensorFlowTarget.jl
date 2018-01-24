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
  @show length(iters)
  @show length(▸(carr))
  @pre length(iters) == length(▸(carr)) # "Need iteraator foreach in port"
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
      phsvalmap = Dict(ph => Arrows.take1(iters[id]) for (ph, id) in phs)
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
