function showgraphstats(graph)
  opts = collect(TensorFlow.get_operations(graph))
  # println("Opts in graph are:\n", opts)
  println("Number of operations is \n", length(opts))
end

"Create placeholders for each in port"
function init_placeholders(arr::Arrow; TensorT=Float32)
  intens = Tensor[]
  ph_pid = Dict{Tensor, Int}()
  for (i, prt) in enumerate(▸(arr))
    ph = tf.placeholder(TensorT, name="inp_$i")
    push!(intens, ph)
    ph_pid[ph] = i
    # end
  end
  intens, ph_pid
end


"""
argmin_θ(ϵprt): find θ which minimizes ϵprt

# Arguments
- `ϵprt`: out port to minimize
- `over`: ports to optimize over
- `ingen`: Iterator producing vector, one for each in_port of carr
# Result
- `θ_optim`: minimal value of ϵprt found
- `argmin`: argmin of `over` found
"""
function optimize(carr::CompArrow, # Redundant? #FIXME, remove
                  ϵprt::Port,
                  ingen,
                  target=Type{TFTarget};
                  testevery = 10,
                  testingen = ingen,
                  logdir::String = log_dir(),
                  TensorT = Float32,
                  kwargs...)
  @pre ϵprt ∈ ⬧(carr)
  graph = tf.Graph()
  sess = tf.Session(graph)

  summary = TensorFlow.summary
  # weight_summary = summary.histogram("Parameters", weights)

  # Create a summary writer

  ## FIXME: Can't we break this up?
  tf.as_default(graph) do
    # Create input tensor for reach in port and mapping between port ids and placeholders
    intens, ph_pid = init_placeholders(carr, TensorT = TensorT)
    tfarr = Graph(carr, graph, intens)
    ϵid = findfirst(◂(ϵprt.arrow), ϵprt)
    losses = tfarr.out[ϵid]

    meanloss = tf.reduce_mean(losses) #FIXME: Should param
    optimizer = tf.train.AdamOptimizer() #FIXME: Should be param
    min_op = tf.train.minimize(optimizer, meanloss) # Should be param
    mean_train_loss = summary.scalar("Loss", meanloss)
    merged_summary_op = summary.merge_all() #
    summary_writer = summary.FileWriter(joinpath(logdir, "train"))
    test_writer = summary.FileWriter(joinpath(logdir, "test"))
    
    run(sess, global_variables_initializer())   
    function step!(cb_data, callbacks)
      if cb_data.i % testevery != 0
        indata = AlioAnalysis.take1(ingen)
        ph_to_val = populate(indata, ph_pid)
        cur_loss, _, summaries = run(sess, [meanloss, min_op, merged_summary_op], ph_to_val)
        write(summary_writer, summaries, cb_data.i)
        @show cur_loss
        cb_data_ = merge(cb_data, @NT(loss = cur_loss))
        foreach(cb->cb(cb_data_), callbacks)
        cur_loss  
      # HACK
      elseif testingen != ingen
        # Don't minimize
        testindata = AlioAnalysis.take1(testingen)
        ph_to_val = populate(testindata, ph_pid)
        train_loss, summaries = run(sess, [meanloss, merged_summary_op], ph_to_val)
        println("Trainloss, ", train_loss)
        write(test_writer, summaries, cb_data.i)
        train_loss
      end
      # summaries = run(sess, merged_summary_op, ph_to_val)
    end
    showgraphstats(graph)
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
