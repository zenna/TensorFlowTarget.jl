using TensorFlow
import TensorFlowTarget: mlp_template
using Arrows

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
