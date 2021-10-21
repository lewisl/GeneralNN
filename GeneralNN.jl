#DONE


   


#BRANCH TODO: layer-based API 
#   Convolutional networks
    #   convolutional layers
    #   pooling layers
# Recurrent Neural networks
    #   implement early stopping
# LSTM networks
# SVM

#TODO
# look at Julia Computing paper on Zygote.  Look at article on integrating zygote and xgboost
# allow choice in what to return from training: basic should exclude data (default), full should be everything;
    # none is none, maybe allow individual choices
# pooling: do we need asymmetrical pooling? pooling doesn't work with padding or same
# should stride allow asymmetrical strides?  probably...
# pooling doesn't work with example dimension.  does it need to?
# should we split cached values from basic params?  should we combine bn parameters with weights?
# save a model_def more easily
# reuse a model def and run it with different data
# should we have a bigger model obj that includes function stacks, hyper_parameters, and weights?
# verify formula for Xavier initialization: decide whether to use normal or uniform (offer both?)
# implement Kaiming initialization
# enable plot display of train cost but not test cost
# fix the accuracy calc to be more general or provide several versions
#   what's the difference between shuffling data to make different batches v. 
            #choosing batches randomly (from data shuffled just once)
            # we'd see all the data in each epoch for both approaches
            # composition of batches different v.
            # seeing the batches in a different order
#   change alphamod to alpha, change alpha to alpha_in??
#   fix testgrad.jl: check_grads, prep_check for changes to predict using model and bn params
#   fix nnpredict or get rid of it
#   implement gradient clipping
#   add ability to choose BN per layer (inc. for output layer or not)
#   add ability to put bn after linear or after activation
#   investigate Zygote.jl for automatic differentiation
#   test if we can use batch normalization with full batch training
#   check what happens to test cost with regularization:  it gets very big
#   should we have a normalize data option built-in (as we do)? or make the user do it when
         # preparing their own data?
#   export onehot--decide how to transpose data
#   create a cost function method without L2 regularization
#   create a cost prediction method that embeds the feedfwd predictions
#   figure out memory requirements of pre-allocation.  is there someway to reduce and still get speed benefit?
#   in testgrad, test for categorical data or provide other way to do onehot encoding if needed
#   utilities for plotdef will break on old plotdefs because they are now called stats
#   use goodness function to hold either accuracy or r_squared?
#   implement incremental load and train for massive datasets?
#   set an explicit data size variable for both test and train for preallocation?
#   accuracy for logistic or other single class classifiers should allow -1 or 0 for "bad"
#   implement precision and recall
#   stats on individual regression parameters
#   do check on existence of matfname file and that type is .mat
#   implement one vs. all for logistic classification with multiple classes
#   set a directory for training stats (keep out of code project directory)
#   try different versions of ensemble predictions_vector
#   augment MINST data by perturbing the images
#   check for type stability: @code_warntype pisum(500,10000)
#   still lots of memory allocations despite the pre-allocation
        # You can devectorize r -= d[j]*A[:,j] with r .= -.(r,d[j]*A[:.j]) 
        #        to get rid of some more temporaries. 
        # As @LutfullahTomak said sum(A[:,j].*r) should devectorize as dot(view(A,:,j),r) 
        #        to get rid of all of the temporaries in there. 
        # To use an infix operator, you can use \cdot, as in view(A,:,j)⋅r.
#   figure out memory use between train set and minibatch set

# DEFER
#   implement layer normalization with or without minibatches. should be easy, but side effects 


#Document
#   say which properties in hyper_parameters are "calculated" and which are directly from user input



"""
Module GeneralNN:

Includes the following functions to run directly:

- train() -- train neural networks for up to 9 hidden layers
- setup_params -- load hyper_parameters and create Hyper_parameters object
- extract_data() -- extracts data for MNIST from matlab files
- shuffle_data!() -- shuffles training examples (do this before minibatch training)
- test_score() -- cost and accuracy given test data and saved theta
- save_params() -- save all model parameters
- load_params() -- load all model parameters
- predictions_vector() -- predictions given x and saved theta
- accuracy() -- calculates accuracy of predictions compared to actual labels
- normalize_inputs() -- normalize via standardization (mean, sd) or minmax
- normalize_replay!() --
- nnpredict() --  calculate predictions from previously trained parameters and new data
- display_mnist_digit() --  show a digit
- wrong_preds() -- return the indices of wrong predictions against supplied correct labels
- right_preds() -- return the indices of correct predictions against supplied correct labels
- plot_output() -- plot learning, cost for training and test for previously saved training runs

Enter using .GeneralNN to use the module.

These data structures are used to hold parameters and data:

- Wgts holds theta, bias, delta_th, delta_b, theta_dims, output_layer, k
- Model_data holds inputs, targets, a, z, z_norm, epsilon, gradient_function
- Batch_norm_params holds gam (gamma), bet (beta) for batch normalization and intermediate
    data used for backprop with batch normalization: delta_gam, delta_bet, 
    delta_z_norm, delta_z, mu, stddev, mu_run, std_run
- Hyper_parameters holds user input hyper_parameters and some derived parameters.
- Batch_view holds views on model data to hold the subset of data used in minibatches. 
  (This is not exported as there is no way to use it outside of the backprop loop.)
- Model_def holds the string names and functions that define the layers of the training
    model.

"""
module GeneralNN


# ----------------------------------------------------------------------------------------

# data structures for neural network
export 
    Batch_view,
    Wgts, 
    Model_data, 
    Batch_norm_params, 
    Hyper_parameters,
    Model_def

# functions you can use
export 
    train,
    setup_params, 
    pretrain,
    prepredict,
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
    dodigit,
    check_grads

using MAT
using JLD2
using FileIO
using Statistics
using Distributions
using Random
using Printf
using Dates
using LinearAlgebra
using SparseArrays
# for TOML support for arguments file
using TOML  # we might need private TOML for Julia < 1.3
using Debugger

# using Plots   # Plots broken by Julia 1.3
# gr()

using PyPlot

import Measures: mm # for some plot dimensioning

import Base.show  # for new methods to pretty-print our objects

# be careful: order of includes matters!
include("nn_data_structs.jl")
include("training_loop.jl")
include("layer_functions.jl")
include("setup_params.jl")
include("setup_data.jl")
include("setup_model.jl")
include("utilities.jl")
include("training.jl")
include("testgrad.jl")



const l_relu_neg = .01  # makes the type constant; value can be changed.


end  # module GeneralNN