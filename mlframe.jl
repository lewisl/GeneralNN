# function convnet()
#     # some useful dimensions

#     # topology of multiple images
#     m,n,c,k  # m rows, n columns, c channels, k images

#     # net architecture by layer
#     arch = Dict(
#                 1 => Dict(
#                     "type" => "conv", # one of conv, max, avg, fc, classify???, relu, l_relu, sigmoid, softmax, none
#                     "dims" => [24,24,3]
#                 ),
#                 2 => Dict(
#                     "type" => "conv", 
#                     "dims" => [24,24,3]
#                 ),
#                 3 => Dict(
#                     "type" => "conv", 
#                     "dims" => [24,24,3]
#                 ),
#                 4 => Dict(
#                     "type" => "max", # one of conv, max, avg, fc, classify
#                     "dims" => [24,24,3]
#                      ),
#                  5 => Dict(
#                      "type" => "fc"
#                      "dims" => "auto"
#                      ),
#
#                  6 => Dict(
#                      "type" => "relu"
#                      "dims" => "auto"
#                       ),
#
#                  7 => Dict(
#                      "type" => "softmax"
#                      "dims" => ["previous", 10]  # this isn't right
#                       ),
#             )

#     # number  layers
#     num_layers = maximum(keys(arch))

#     # layer functions -- canonical set
#     func_def = Dict(
#         "conv" = convolve,
#         "relu" = relu,
#         "l_relu" = l_relu,
#         "sigmoid" = sigmoid,
#         "softmax" = softmax,
#         "max" = maxpooling,
#         "avg" = avgpooling,
#         "fc" = fc,
#         "classify" = classify,
#         "none" = donothing,
#     )

#     # set layer functions to use per layer of network architecture
#     layer_funcs = [func_def[arch[i]["type"]] for i in 1:num_layers]

# end