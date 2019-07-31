#DONE

#BRANCH TODO: layer-based API 
#   change to layer based api, with simplified api as alternate front-end
#   Convolutional networks
    #   convolutional layers
    #   pooling layers
# Recurrent Neural networks
    #   implement early stopping
# LSTM networks

#TODO
#   digit plots fail unless cost or learning plotted first
#   try to speed up saving the plot stats
#   use goodness function to hold either accuracy or r_squared
#   implement incremental load and train for massive datasets?
#   fix dropout: should delta_theta be averaged using number of included units?
        #   are we scaling units during training?
#   set an explicit data size variable for both test and train for preallocation?
#   accuracy for logistic or other single class classifiers should allow -1 or 0 for "bad"
#   TODO use bias in the output layer with no batch norm?  YES
#   implement precision and recall
#   stats on individual regression parameters
#   separate reg from the cost calculation and the parameter updates
#   do check on existence of matfname file and that type is .mat
#   implement one vs. all for logistic classification with multiple classes
#   factor out extract data--provide as utility; start process with datafiles
#   is dropout dropping the same units on backprop as feedfwd?  seems like it but results are still poor
#   set a directory for training stats (keep out of code project directory)
#   try different versions of ensemble predictions_vector
#   augment MINST data by perturbing the images
#   separate plotdef from plot descriptive data?
#   check for type stability: @code_warntype pisum(500,10000)
#   still lots of memory allocations despite the pre-allocation
        # You can devectorize r -= d[j]*A[:,j] with r .= -.(r,d[j]*A[:.j]) 
        #        to get rid of some more temporaries. 
        # As @LutfullahTomak said sum(A[:,j].*r) should devectorize as dot(view(A,:,j),r) 
        #        to get rid of all of the temporaries in there. 
        # To use an infix operator, you can use \cdot, as in view(A,:,j)⋅r.
#   figure out memory use between train set and minibatch set
#   implement a gradient checking function with option to run it



"""
Module GeneralNN:

Includes the following functions to run directly:

- train_nn() -- train neural networks for up to 9 hidden layers
- test_score() -- cost and accuracy given test data and saved theta
- save_params() -- save all model parameters
- load_params() -- load all model parameters
- predictions_vector() -- predictions given x and saved theta
- accuracy() -- calculates accuracy of predictions compared to actual labels
- extract_data() -- extracts data for MNIST from matlab files
- normalize_inputs() -- normalize via standardization (mean, sd) or minmax
- normalize_replay!() --
- nnpredict() --  calculate predictions from previously trained parameters and new data
- display_mnist_digit() --  show a digit
- wrong_preds() --  return the indices of wrong predictions against supplied correct labels
- right_preds() --  return the indices of correct predictions against supplied correct labels
- plot_output() --  plot learning, cost for training and test for previously saved training runs

To use, include() the file.  Then enter using .GeneralNN to use the module.

These data structures are used to hold parameters and data:

- NN_weights holds theta, bias, delta_w, delta_b, theta_dims, output_layer, k
- Model_data holds inputs, targets, a, z, z_norm, epsilon, gradient_function
- Batch_norm_params holds gam (gamma), bet (beta) for batch normalization and intermediate
    data used for backprop with batch normalization: delta_gam, delta_bet, 
    delta_z_norm, delta_z, mu, stddev, mu_run, std_run
- Hyper_parameters holds user input hyper_parameters and some derived parameters.
- Batch_view holds views on model data to hold the subset of data used in minibatches. 
  (This is not exported as there is no way to use it outside of the backprop loop.)

"""
module GeneralNN


# ----------------------------------------------------------------------------------------

# data structures for neural network
export 
    Batch_view,
    NN_weights, 
    Model_data, 
    Batch_norm_params, 
    Hyper_parameters

# functions you can use
export 
    train_nn, 
    test_score, 
    save_params, 
    load_params, 
    accuracy,
    extract_data, 
    shuffle_data!,
    normalize_inputs!, 
    nnpredict,
    display_mnist_digit,
    wrong_preds,
    right_preds,
    plot_output,
    dodigit

using MAT
using JLD2
using FileIO
using Statistics
using Random
using Printf
using Dates
using LinearAlgebra
using SparseArrays

using Plots
# plotlyjs()  # PlotlyJS backend to local electron window
pyplot()

import Measures: mm # for some plot dimensioning

# be careful: order of includes matters!
include("nn_data_structs.jl")
include("training_loop.jl")
include("layer_functions.jl")
include("setup.jl")
include("utilities.jl")
include("training.jl")

# for TOML support for arguments file
using TOML

const l_relu_neg = .01  # makes the type constant; value can be changed.

end  # module GeneralNN