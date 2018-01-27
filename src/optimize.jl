function showgraphstats(graph)
  opts = collect(TensorFlow.get_operations(graph))
  println("Opts in graph are:\n", opts)
  println("Number of operations is \n", length(opts))
end

function init_placeholders(arr::Arrow)
  intens = Tensor[]
  ph_pid = Dict{Tensor, Int}()
  for (i, prt) in enumerate(▸(arr))
    ph = tf.placeholder(Float64, name="inp_$i")
    push!(intens, ph)
    ph_pid[ph] = i
    # end
  end
  intens, ph_pid
end

# function genstep(iters, indata, ph_pid,)
#   function step!()
#     indata = AlioAnalysis.take1(iters)
#     phsvalmap = populate(indata, ph_pid)
#     cur_loss, _ = run(sess, [meanloss, minimize_op], phsvalmap)
#     summaries = run(sess, merged_summary_op, phsvalmap)
#     write(summary_writer, summaries, i)
#     cur_loss
#     i = i + 1
#     cur_loss
#   end
# end

"""
argmin_θ(ϵprt): find θ which minimizes ϵprt

# Arguments
- `ϵprt`: out port to minimize
- `over`: ports to optimize over
- `initer`: Iterator producing vector, one for each in_port of carr
# Result
- `θ_optim`: minimal value of ϵprt found
- `argmin`: argmin of `over` found
"""
function optimize(carr::CompArrow, # Redundant? #FIXME, remove
                  ϵprt::Port,
                  initer,
                  target=Type{TFTarget};
                  logdir::String = log_dir(),
                  kwargs...)
  @pre ϵprt ∈ ⬧(carr)
  graph = tf.Graph()
  sess = tf.Session(graph)

  summary = TensorFlow.summary
  # weight_summary = summary.histogram("Parameters", weights)

  # Create a summary writer

  ## FIXME: Can't we break this up?
  tf.as_default(graph) do
    showgraphstats(graph)
    # Create input tensor for reach in port and mapping between port ids and placeholders
    intens, ph_pid = init_placeholders(carr)
    tfarr = Graph(carr, graph, intens)
    ϵid = findfirst(◂(ϵprt.arrow), ϵprt)
    losses = tfarr.out[ϵid]

    meanloss = tf.reduce_mean(losses) #FIXME: Should param
    optimizer = tf.train.AdamOptimizer() #FIXME: Should be param
    minimize_op = tf.train.minimize(optimizer, meanloss) # Should be param
    alpha_summmary = summary.scalar("Mean Loss", meanloss)
    merged_summary_op = summary.merge_all() #
    summary_writer = summary.FileWriter(logdir) # FIXME Should be opt?
    showgraphstats(graph)
    run(sess, global_variables_initializer())
    # i = 1
    function step!(cb_data, callbacks)
      indata = AlioAnalysis.take1(initer)
      phsvalmap = populate(indata, ph_pid)
      cur_loss, _ = run(sess, [meanloss, minimize_op], phsvalmap)
      @show cur_loss
      summaries = run(sess, merged_summary_op, phsvalmap)
      write(summary_writer, summaries, cb_data.i)
      # i = i + 1
      cb_data_ = merge(cb_data, @NT(loss = cur_loss))
      foreach(cb->cb(cb_data_), callbacks)
      cur_loss
    end
    return optimize(step!; kwargs...)
  end
end

function populate(indata, ph_pid)
  Dict(ph => indata[id] for (ph, id) in ph_pid)
end

# function populate(indata::Vector, ph_pid)
#   @pre length(ph_pid) == 1
#   Dict(ph => indata for (ph, id) in ph_pid)
# end
