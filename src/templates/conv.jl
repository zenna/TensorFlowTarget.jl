# Convolutional Template
using Distributions

"Sample hyper parameters for convolutional network template"
function rand_convnet_hypers(;pbatch_norm = 0.5,
                              max_layers = 5)
  batch_norm = rand() > pbatch_norm
  nlayers = rand(1:max_layers)
  h_channels = rand([16, 32, 8])
  act = rand([tf.nn.relu, tf.nn.elu])

  return Dict(:batch_norm => batch_norm,
              :h_channels => h_channels,
              :nhlayers => nlayers,
              :activation => act)
end

function weight_variable(shape, T=Float32)
  initial = map(T, rand(Normal(0, .001), shape...))
  return tf.Variable(initial)
end

function bias_variable(shape)
  initial = fill(Float32(.1), shape...)
  return tf.Variable(initial)
end

"Simple Convolutional Template"
function conv_template(args,
                       insizes,
                       outsizes;
                       channel_dim=4,   # Which dimension if hte channel
                       batch_norm=false,# Do batch normalization
                       nhlayers=0,      # Nubmer of hidden layers 
                       h_channels=8,    # Numbers of channels in each hidden layer
                       combine_inputs=xs->tf.stack(xs, axis=channel_dim),
                       activation=tf.nn.elu,
                       T=Float32,
                       kwargs...)

  @show h_channels
  @show activation
  @show nhlayers
  h_channels = 8
  activation = tf.nn.elu
  nhlayers = 0
  # @pre same(size.([insizes; outsizes])) "Variable sizes unsupported"
  @pre ndims(insizes[1]) == 3 "nchannels != 3 is unhandled"
  in_channels = length(args)
  out_channels = length(outsizes)
  fh = 1  # Filter width
  fw = 1  # Filter height
  strides = [1, 1, 1, 1]

  x = combine_inputs(args)
  W_conv1 = weight_variable([fw, fh, in_channels, h_channels])
  x = nn.conv2d(x, W_conv1, strides, "SAME", name="conv1")

  # Hidden layers
  for i = 1:nhlayers
    W_conv = weight_variable([fw, fh, h_channels, h_channels])
    x = nn.conv2d(x, W_conv, strides, "SAME", name="conv_h_$i")
    x = activation(x)
  end

  # Final Convolution
  W_final = weight_variable([fw, fh, h_channels, out_channels])
  x = nn.conv2d(x, W_final, strides, "SAME", name="conv_final")
  x = activation(x)
  
  # Split before softmax
  xs = tf.split(4, length(outsizes), x)
  xs_sq = map(xs) do x
    tf.reshape(nn.softmax(tf.squeeze(x, [2, 4])), get(outsizes[1]))
  end
end