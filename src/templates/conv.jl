# Convolutional Template
using Distributions

"Sample hyper parameters"
function sample_hyper(in_sizes, out_sizes; pbatch_norm=0.5, max_layers=5)
  batch_norm = rand() > pbatch_norm
  nlayers = rand(1:max_layers)
  h_channels = rand([12, 16, 24])
  act = rand([F.relu, F.elu])
  return Dict(:batch_norm => batch_norm,
              :h_channels => h_channels,
              :nhlayers => nlayers,
              :activation => act)
end

stack_channels(xs) = tf.stack(xs, axis=4)

function weight_variable(shape, T=Float32)
  initial = map(T, rand(Normal(0, .001), shape...))
  return Variable(initial)
end

function bias_variable(shape)
  initial = fill(Float32(.1), shape...)
  return Variable(initial)
end

"Simple Convolutional Template"
function conv_template(args,
                       insizes,
                       outsizes;
                       channel_dim=4,
                       batch_norm=false,
                       h_channels=8,
                       nhlayers=4,
                       combine_inputs=stack_channels,
                       activation=tf.nn.elu,
                       T=Float32)
  # @pre same(size.([insizes; outsizes])) "Variable sizes unsupported"
  @grab insizes
  @grab outsizes
  @grab args
  @pre ndims(insizes[1]) == 3 "other nchannels unhandled"


  @show x = combine_inputs(args)
  # [batch, in_height, in_width, in_channels]
  W_conv1 = weight_variable([5, 5, length(args), h_channels])
  @show x = nn.conv2d(x, W_conv1, [1, 1, 1, 1], "SAME", name="conv1WAA")
  @show x = nn.elu(x)
  W_conv2 = weight_variable([5, 5, h_channels, length(outsizes)])
  @show x = nn.conv2d(x, W_conv2, [1, 1, 1, 1], "SAME", name="conv2WOO")

  # Split before softmax
  xs = tf.split(4, length(outsizes), x)
  # xs_sq = map(xs) do x
  #   tf.reshape(nn.softmax(tf.squeeze(x, [2, 4])), get(outsizes[1]))
  # end
end


#   # tf.Ops.conv2d(x, filter)
#   # x = self.conv1(x)

#   # # h layers
#   # for (i, layer) in enumerate(self.hlayers):
#   #   x = layer(x)
#   #   if self.batch_norm:
#   #     x = self.blayers[i](x)
#   #   x = self.activation(x)

#   # x = self.conv2(x)
#   # x = self.activation(x)

#   # # Uncombine
#   split_channels = split_channel(x, self.out_sizes)
#   return tuple(map(slither, split_channels, self.out_sizes)) # FIXME, not all men

# end