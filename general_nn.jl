#TODO
#   DONE: factor data prep
#   DONE: factor setting up plot data structures
#   factor learning algorithm code
#   scale weights for cost regularization to accommodate ReLU normalization
#   fix array type declaration?
#   implement momentum
#   add early stopping
#   add dropout
#   find and fix deprecated Array declaration syntax with new syntax
#   foo



# Includes the following functions to run directly:
#   train_nn() -- train sigmoid/softmax neural networks for up to 5 layers
#   test_score() -- cost and accuracy given test data and saved theta
#   save_theta() -- save theta, which is returned by train_nn
#   predictions_vector() -- predictions given x and saved theta


using MAT
using Devectorize
using PyCall
using Plots
pyplot()  # initialize the backend used by Plots
@pyimport seaborn  # prettier charts


function extract_data(matfname::String)
    # read the data
    df = matread(matfname)

    # Split into train and test datasets, if we have them
    # transpose examples as columns to optimize for julia column-dominant operations
    # e.g.:  rows of a single column are features; each column is an example data point
    if in("train", keys(df))
        inputs = df["train"]["x"]'  
        targets = df["train"]["y"]'
    else
        inputs = df["x"]'
        targets = df["y"]'
    end
    if in("test", keys(df))
        dotest = true
        test_inputs = df["test"]["x"]'  # note transpose operator
        test_targets = df["test"]["y"]'
        testn = size(test_inputs,2)
    else
        dotest = false
        test_inputs = zeros(0,0)
        test_targets = zeros(0,0)
    end
    return inputs, targets, test_inputs, test_targets, dotest
end

"""
Function setup_plots(n_iters::Int64, dotest::Bool, plots::Array{String,1})

Creates data structure to hold everything needed to plot progress of
neural net training by iteration.

A plotdef is a dict containing:

    "plot_switch"=>plot_switch: Dict of bools for each type of results to be plotted.
        Currently used are: "Training", "Test", "Learning".  This determines what 
        data will be collected during training iterations and what data series will be
        plotted.
    "plot_labels"=>plot_labels: array of strings provides the labels to be used in the
        plot legend. 
    "cost_history"=>cost_history: an array of calculated cost at each iteration 
        with iterations as rows and result types as columns.  
        Results can be Training or Test. 
    "fracright_history"=>fracright_history: an array of percentage of correct classification
        at each iteration with iterations as rows and result types as columns. This plots
        a so-called learning curve.  Very interesting indeed. 
        Results can be Training or Test.  
    "col_train"=>col_train: column of the arrays above to be used for Training results
    "col_test"=>col_test: column of the arrays above to be used for Test results

"""
function setup_plots(n_iters::Int64, dotest::Bool, plots::Array{String,1})
    # set up cost_history to track 1 or 2 data series for plots
    # lots of indirection here:  someday might add "validation"
    if size(plots,1) > 3
        warn("Only 3 plot requests permitted. Proceeding with up to 3.")
    end

    valid_plots = ["Training", "Test", "Learning"]
    plot_switch = Dict(pl => false for pl in valid_plots)
    for pl in plots  # plots is the input request for plots
        if in(pl, valid_plots)
                plot_switch[pl] = true
        else
            warn("Plots argument can only include \"Training\", \"Test\", and \"Learning\". Proceeding.")
        end
    end

    # must have test data to plot test results
    if dotest  # test data is present
        # nothing to change
    else
        if plot_switch["Test"]  # input requested plotting test cost
            warn("Can't plot test cost. No test data. Proceeding.")
            plot_switch["Test"] = false
        end
    end
    plot_labels = [pl for pl in keys(plot_switch) if plot_switch[pl] == true &&
        pl != "Learning"]  # Learning is a separate plot, not a series label
    plot_labels = reshape(plot_labels,1,size(plot_labels,1)) # 1 x N row array

    cost_history = zeros(n_iters, size(plot_labels,2))
    fracright_history = zeros(n_iters, size(plot_labels,2))  # shouldn't if not plotting Learning
 
    # set column in cost_history for each data series
    col_train = plot_switch["Training"] ? 1 : 0
    col_test = plot_switch["Test"] ? col_train + 1 : 0

    return Dict("plot_switch"=>plot_switch, "plot_labels"=>plot_labels, 
        "cost_history"=>cost_history, "fracright_history"=>fracright_history, 
        "col_train"=>col_train, "col_test"=>col_test)
end


"""
Method to call train_nn with a single integer as the number of hidden units
in a single hidden layer.
"""
function train_nn(matfname::String, n_iters::Int64, n_hid::Int64; alpha=0.35,
    mb_size=0, lambda=0.015, classify="softmax", units="sigmoid", plots=["Training"])

    train_nn(matfname, n_iters, [n_hid], alpha=alpha,
    mb_size=mb_size, lambda=lambda, classify=classify, units=units, plots=plots)
end


"""
    function train_nn(matfname::String, n_iters::Int64, n_hid::Array{Int64,1}, alpha=0.35,
        mb_size=0, lambda=0.015, classify="softmax", units="sigmoid", plots=["Training"])

    returns theta -- the model weights
    key inputs:
        alpha   ::= learning rate
        lambda  ::= regularization rate
        mb_size ::= mini-batch size

Train sigmoid/softmax neural networks up to 5 layers.  Detects
number of output labels from data. Detects number of features from data. Enables
any size mini-batches that divide evenly into number of examples.  Plots 
by any choice of per "Learning", "Training" iteration (epoch), and for "Test" data.
classify may be "softmax" or "sigmoid", which applies only to the output layer. 
units in other layers may be "sigmoid" or "relu".
"""
function train_nn(matfname::String, n_iters::Int64, n_hid::Array{Int64,1}; alpha=0.35,
    mb_size=0, lambda=0.015, classify="softmax", units="sigmoid", plots=["Training"])
    # creates a nn with 1 input layer, up to 3 hidden layers as an array input,
    # and output units matching the dimensions of the training data y
    # returns theta
    # layers must be 1, 2, or 3 hidden layers. In addition, there will always be
    # 1 input layer and 1 output layer. So, total layers will be 3, 4, or 5.

    #check inputs and prepare data
    if ndims(n_hid) != 1
        error("Input n_hid must be a vector.")
    elseif size(n_hid,1) > 3
        error("n_hid can only contain 1 to 3 integer values for 1 to 3 hidden layers.")
    end

    if alpha < 0.00001
        warn("Alpha learning rate set too small. Setting to default 0.35")
        alpha = 0.35
    elseif alpha > 1.0
        warn("Alpha learning rate set too large. Setting to defaut 0.35")
        alpha = 0.35
    end

    # read file and extract data
    inputs, targets, test_inputs, test_targets, dotest = extract_data(matfname)


    # set some useful variables
    k,n = size(inputs)  # number of features k by no. of examples n
    t = size(targets,1)  # number of output units
    n_hid_layers = size(n_hid, 1)
    output_layer = 2 + n_hid_layers # input layer is 1, output layer is highest value
    mb = mb_size  # shortcut
    lamovern = lambda / (2 * n)  # need this because @devec won't do division. 
        
    #setup mini-batch
    if mb_size == 0
        mb_size = n  # cause 1 (mini-)batch with all of the examples
    elseif mb_size > n
        mb_size = n
    elseif mb_size < 1
        mb_size = 1
    elseif mod(n, mb_size) != 0
        error("Mini-batch size $mb_size does not divide evenly into samples $n.")
    end
    n_mb = Int(n / mb_size)  # number of mini-batches

    if mb_size < n
        # randomize order of samples: 
            # labels in training data sometimes in a block, which will make
            # mini-batch train badly because a batch might not contain mix of labels
        sel_index = randperm(n)
        inputs[:] = inputs[:, sel_index]
        targets[:] = targets[:, sel_index]  
    end

    # pre-allocate mini-batch inputs and targets
    mb_inputs = zeros(size(inputs,1), mb_size)  
    mb_targets = zeros(size(targets,1), mb_size)  

    # setup data structures for plotting
    plotdef = setup_plots(n_iters , dotest, plots)

    # prepare arrays used in training 
    # theta dimensions for each layer of the neural network 
    #    this follows the convention that the outputs to the current layer activation
    #    are rows of theta and the inputs from the layer below are columns
    theta_dims = [[k, 1]] # "weight" dimensions for the input layer--not used
    for i = 2:output_layer-1
        push!(theta_dims, [n_hid[i-1], theta_dims[i-1][1] + 1])  # 2nd index at each layer includes bias term
    end
    push!(theta_dims, [t, n_hid[end]+ 1]) # weight dimensions for the output layer

    # use either sigmoid or softmax for output layer with multiple classification
    if theta_dims[output_layer][1] > 1  # more than one output (unit)
        if classify == "sigmoid"
            class_function = sigmoid
        elseif classify == "softmax"
            class_function = softmax
        else
            error("Function to classify output labels must be \"sigmoid\" or \"softmax\".")
        end
    else
        class_function = sigmoid  # for one output (unit)
    end

    if units == "sigmoid"
        unit_function = sigmoid
    elseif units == "relu"
        unit_function = relu
    end

    # setup cost function
    cost_function = cross_entropy_cost

    # theta = weight matrices in a collection for all calculated layers (e.g., not the input layer)
    # initialize random weight parameters including bias term
    theta = [zeros(2,2)] # initialize collection of 2d float arrays: input layer 1 not used
    interval = 0.5 # random weights will be generated in [-interval, interval]
    for i = 2:output_layer
        push!(theta, rand(theta_dims[i]...) .* (2.0 * interval) .- interval)
    end

    # pre-allocate matrices used in training by layer to enable update-in-place for speed
    a, a_wb, z = preallocate_feedfwd(inputs, theta, output_layer, n)
    mb_a, mb_a_wb, mb_z = preallocate_feedfwd(mb_inputs, theta, output_layer, mb_size)
    if dotest
        testn = size(test_inputs,2)
        a_test, a_wb_test, z_test = preallocate_feedfwd(test_inputs, theta, output_layer, testn)
        test_predictions = deepcopy(a_test[output_layer])
    end

    # initialize matrices for back propagation to enable update-in-place for speed
    eps = deepcopy(mb_a)  # looks like activations of each unit above input layer
    delta = deepcopy(theta)  # structure of gradient matches theta
    # initialize before loop to set scope OUTSIDE of loop
    mb_predictions = deepcopy(mb_a[output_layer])  # predictions = output layer values

    # train the neural network and accumulate mb_cost_history
    for i = 1:n_iters  # loop for "epochs"
        for j = 1:n_mb  # loop for mini-batches

            # grab the mini-batch subset of the data for the input layer 1
            start = (j - 1) * mb_size + 1
            fin = start + mb_size - 1
            mb_a[1][:] = inputs[:,start:fin]  # input layer activation for mini-batch
            mb_a_wb[1][:] = vcat(ones(1,mb_size), mb_a[1])
            mb_targets[:] = targets[:, start:fin]         

            mb_predictions[:] = feedfwd!(theta, output_layer, unit_function, class_function, mb_a, 
                mb_a_wb, mb_z)  
            backprop_gradients!(theta, mb_targets, unit_function, output_layer, 
                mb_size, mb_a, mb_a_wb, mb_z, eps, delta)  

            # calculate new theta
            @fastmath for ii = 2:output_layer
                # regularization term added when lambda > 0
                if lambda > 0.0  
                    delta[ii][:, 2:end] = delta[ii][:, 2:end] .+ lamovern .* theta[ii][:,2:end]  # don't regularize bias
                end
                theta[ii][:] = theta[ii] .- (alpha .* delta[ii])  
            end
        end

        # gather statistics for plotting
        if plotdef["plot_switch"]["Training"]
            plotdef["cost_history"][i, plotdef["col_train"]] = cost_function(mb_targets, 
                mb_predictions, n, n, theta, lambda, output_layer)
            if plotdef["plot_switch"]["Learning"]
                plotdef["fracright_history"][i, plotdef["col_train"]] = accuracy(
                    mb_targets, mb_predictions)
            end
        end
        
        if plotdef["plot_switch"]["Test"]
            test_predictions[:] = feedfwd!(theta, output_layer, unit_function, class_function,
                a_test, a_wb_test, z_test)
            plotdef["cost_history"][i, plotdef["col_test"]] = cost_function(test_targets, 
                test_predictions, testn, testn, theta, lambda, output_layer)
            if plotdef["plot_switch"]["Learning"]
                plotdef["fracright_history"][i, plotdef["col_test"]] = accuracy(test_targets, test_predictions)
            end
        end        

    end
    
    # output training statistics
    predictions = feedfwd!(theta, output_layer, unit_function, class_function, a, a_wb, z)
    println("Fraction correct labels predicted training: ", accuracy(targets, predictions))
    println("Final cost training: ", cost_function(targets, predictions, n,
                    n, theta, lambda, output_layer))

    # output test statistics
    if dotest     
        predictions = feedfwd!(theta, output_layer, unit_function, class_function, a_test, a_wb_test, z_test)
        println("Fraction correct labels predicted test: ", accuracy(test_targets, predictions))
        println("Final cost test: ", cost_function(test_targets, predictions, testn, testn, theta, lambda, output_layer))
    end

    # plot the progress of training cost and/or learning
    plot_output(plotdef)

    return theta
end  # function train_nn


function preallocate_feedfwd(inputs, theta, output_layer, n)
    a = [inputs]
    a_wb = [vcat(ones(1, n), inputs)] # input layer with bias column, never changed in loop
    z = [zeros(2,2)] # not used for input layer
    for i = 2:output_layer-1
        push!(z, zeros(size(theta[i] * a_wb[i-1])))  # z2 and up...  ...output layer set after loop
        push!(a, zeros(size(z[i])))  # a2 and up...  ...output layer set after loop
        push!(a_wb, vcat(ones(1, n), a[i]))  # a2_wb and up... ...but not output layer    
    end
    push!(z, zeros(size(theta[output_layer],1),n))  # z output layer z[output_layer]  similar(targets)  zeros(size(theta[output_layer],1),n)
    push!(a, zeros(size(theta[output_layer],1),n))  # a output layer a[output_layer]  similar(targets)
    return a, a_wb, z
end


function feedfwd!(theta, output_layer, unit_function, class_function, a, a_wb, z)
    # modifies a, a_wb, z in place
    # send it all of the data or a mini-batch
    # receives intermediate storage of a, a_wb, z to reduce memory allocations
    # x is the activation of the input layer, a[1]
    # x[:] enables replace in place--reduce allocations, speed up loop

    # feed forward from inputs to output layer predictions
    @fastmath for ii = 2:output_layer-1  # ii is the current layer
        z[ii][:] = theta[ii] * a_wb[ii-1]
        a[ii][:] = unit_function(z[ii])
        a_wb[ii][2:end, :] = a[ii]  
    end
    @fastmath z[output_layer][:] = theta[output_layer] * a_wb[output_layer-1]
    a[output_layer][:] = class_function(z[output_layer])
end


function backprop_gradients!(theta, targets, unit_function, output_layer, mb, a, a_wb, z, eps, delta)
    # argument delta holds the computed gradients
    # modifies eps, delta in place--caller uses delta
    # use for iterations in training
    # send it all of the data or a mini-batch
    # receives intermediate storage of a, a_wb, z, eps, delta to reduce memory allocations

    oneovermb = 1.0 / mb

    if unit_function == sigmoid
        gradient_function = sigmoid_gradient
    elseif unit_function == relu
        gradient_function = relu_gradient
    end

    eps[output_layer][:] = a[output_layer] .- targets  # eps is epsilon
    @fastmath delta[output_layer][:] = oneovermb .* (eps[output_layer] * a_wb[output_layer-1]')
    @fastmath for jj = (output_layer - 1):-1:2  # don't do input layer
        eps[jj][:] = theta[jj+1][:, 2:end]' * eps[jj+1] .* gradient_function(z[jj]) 
        delta[jj][:] = oneovermb .* (eps[jj] * a_wb[jj-1]') 
    end
    # println(z[2])
    # println(gradient_function(z[2]))
end


# suitable cost function for sigmoid and softmax(?)
function cross_entropy_cost(targets, predictions, n, mb, theta, lambda, output_layer)
    # n is count of all samples in data set--use with regularization term
    # mb is count of all samples used in training batch--use with cost
    # these may be equal
    @devec cost = (-1.0 ./ mb) .* sum(targets .* log(predictions) .+ 
        (1.0 .- targets) .* log(1.0 .- predictions))
    @fastmath if lambda > 0.0
        # need to fix hidden layer regularization for normalized ReLU
        # because the weights grow because the activations (z) are so small (mean = 0)
        # regterm = lambda/(2.0 * n) * sum([sum(theta[i][:, 2:end] .* theta[i][:, 2:end]) 
        #     for i in 2:output_layer]) 
        regterm = lambda/(2.0 * n) * sum(theta[output_layer][:, 2:end] .* theta[output_layer][:, 2:end]) 
        cost = cost + regterm
    end
    return cost
end


function sigmoid(z::Array{Float64,2})
    @devec  ret = 1.0 ./ (1.0 .+ exp(-z))
    return ret
end


function relu(z::Array{Float64,2})
# this is normalized leaky relu
    ret = similar(z)
    zn = (z .- mean(z,1)) ./ (std(z,1))
    for j = 1:size(z,2)
        for i = 1:size(z,1)
            ret[i,j] = zn[i,j] > 0.0 ? zn[i,j] : .01 * zn[i,j]
        end
    end
    return ret
end


function softmax(atop::Array{Float64,2})  # TODO try this with devec
    f = atop .- maximum(atop,1)  # atop => activation of top layer
    expf = exp.(f)  # this gets called within a loop and exp() is expensive
    return @fastmath expf ./ sum(expf, 1)  
end


function sigmoid_gradient(z::Array{Float64,2})
    # derivative of sigmoid function
    sig = sigmoid(z)
    @devec ret = sig .* (1 .- sig)
    return ret  
end


function relu_gradient(z::Array{Float64,2})
    # don't have to normalize z again as gradient depends only on sign
    ret = similar(z)
    for j = 1:size(z,2)
        for i = 1:size(z,1)
            ret[i,j] = z[i,j] > 0.0 ? 1.0 : .01
        end
    end
    return ret
end


#  NEED TO PRE-ALLOCATE predictions_test somewhere with testn
#  Do we need arguments for testn, test_targets?  yup
#  not being used:  absurd number of complex arguments
# function calc_plot_data!(i, plotdef, targets, predictions, n, theta, lambda, output_layer,
#     dotest, testn, test_targets, test_predictions, a_test, a_wb_test, z_test, 
#     unit_function, class_function, cost_function)

#     if plotdef["plot_switch"]["Training"]
#         plotdef["cost_history"][i, plotdef["col_train"]] = cost_function(targets, predictions, n, n, theta, lambda, output_layer)
#         if plotdef["plot_switch"]["Learning"]
#             plotdef["fracright_history"][i, plotdef["col_train"]] = accuracy(targets, predictions)
#         end
#     end
    
#     if plotdef["plot_switch"]["Test"]
#         test_predictions[:] = feedfwd!(theta, output_layer, unit_function, class_function,
#             a_test, a_wb_test, z_test)
#         plotdef["cost_history"][i, plotdef["col_test"]] = cost_function(test_targets, 
#             test_predictions, testn, testn, theta, lambda, output_layer)
#         if plotdef["plot_switch"]["Learning"]
#             plotdef["fracright_history"][i, plotdef["col_test"]] = accuracy(test_targets, test_predictions)
#         end
#     end
# end


function plot_output(plotdef)
    # plot the progress of training cost and/or learning
    if (plotdef["plot_switch"]["Training"] || plotdef["plot_switch"]["Test"])
        plt_cost = plot(plotdef["cost_history"], title="Cost Function", 
            labels=plotdef["plot_labels"], ylims=(0, Inf))
        display(plt_cost)  # or can use gui()

        if plotdef["plot_switch"]["Learning"]
            plt_learning = plot(plotdef["fracright_history"], title="Learning Progress",
                labels=plotdef["plot_labels"], ylims=(0.0, 1.0), reuse=false) 
                # reuse=  not a great way to open a new plot window
            display(plt_learning)
        end

        println("Press enter to close plot window..."); readline()
        closeall()
    end
end


function accuracy(targets, predictions)
    if size(targets,1) > 1
        # works for output units sigmoid or softmax
        targetmax = [indmax(targets[:,i]) for i in 1:size(targets,2)]
        predmax = [indmax(predictions[:,i]) for i in 1:size(predictions,2)]
        fracright = mean(convert(Array{Int},targetmax .== predmax))
    else
        # works because single output unit is sigmoid
        choices = [j >= 0.5 ? 1.0 : 0.0 for j in predictions]
        fracright = mean(convert(Array{Int},choices .== targets))
    end
    return fracright
end


"""

    function save_theta(theta, mat_fname)

Save the weights, or theta, trained by the neural network as a matlab file.
Can be used to run the model on prediction data or to evaluate other 
test data results (cost and accuracy).
"""
function save_theta(theta, mat_fname)
    # check if output file exists and ask permission to overwrite
    if isfile(mat_fname)
        print("Output file $mat_fname exists. OK to overwrite? ")
        resp = readline()
        if contains(lowercase(resp), "y")
            rm(mat_fname)
        else
            error("File exists. Replied no to overwrite. Quitting.")
        end
    end

    # write the matlab formatted file (based on hdf5)
    outfile = matopen(mat_fname, "w")
    write(outfile, "theta", theta)
    close(outfile) 
end


"""

    function test_score(theta_fname, data_fname, lambda = 0.015, 
        classify="softmax")

Calculate the test accuracy score and cost for a dataset containing test or validation
data.  This data must contain outcome labels, e.g.--y.
"""
function test_score(theta_fname, data_fname, lambda = 0.015, classify="softmax")
    # read theta
    dtheta = matread(theta_fname)
    theta = dtheta["theta"]

    # read the test data:  can be in a "test" key with x and y or can be top-level keys x and y
    df = matread(data_fname)

    if in("test", keys(df))
        inputs = df["test"]["x"]'  # set examples as columns to optimize for julia column-dominant operations
        targets = df["test"]["y"]'
    else
        inputs = df["x"]'
        targets = df["y"]'
    end
    n = size(inputs,2)
    output_layer = size(theta,1) 

    # setup cost
    cost_function = cross_entropy_cost

    predictions = predict(inputs, theta)
    score = accuracy(targets, predictions)
    println("Fraction correct labels predicted test: ", score)
    println("Final cost test: ", cost_function(targets, predictions, n, n, theta, lambda, output_layer))

    return score
end


"""

    function predictions_vector(theta_fname, data_fname, lambda = 0.015, 
        classify="softmax")

    returns vector of all predictions

Return predicted values given inputs and theta.  Not used by training.
Use when theta is already trained and saved to make predictions
for input operational data to use your model. Resolves sigmoid or softmax outputs
to (zero, one) values for each output unit.

Simply does feedforward, but does some data staging first.
"""
function predictions_vector(theta_fname, data_fname, lambda = 0.015, classify="softmax")
  # read theta
    dtheta = matread(theta_fname)
    theta = dtheta["theta"]

    # read the operational data:  can be in a "train" key with x or can be top-level key x
    df = matread(data_fname)

    if in("train", keys(df))
        inputs = df["train"]["x"]'  # set examples as columns to optimize for julia column-dominant operations
    else
        inputs = df["x"]'
    end
    n = size(inputs,2)
    output_layer = size(theta,1) 

    predictions = predict(inputs, theta)
    if size(predictions,1) > 1
        # works for output units sigmoid or softmax
        ret = [indmax(predictions[:,i]) for i in 1:size(predictions,2)]
    else
        # works because single output unit is sigmoid
        ret = [j >= 0.5 ? 1.0 : 0.0 for j in predictions]
    end

    return ret
end

"""

    function predict(inputs, theta)

Generate predictions given theta and inputs.
Not suitable in a loop because of all the additional allocations.
Use with one-off needs like scoring a test data set or 
producing predictions for operational data fed into an existing model.
"""
function predict(inputs, theta)
    # set some useful variables
    k,n = size(inputs) # number of features k by no. of examples n
    output_layer = size(theta,1) 
    t = size(theta[output_layer],1)  # number of output units or 

    # setup cost
    cost_function = cross_entropy_cost

    # setup class function
    if t > 1  # more than one output (unit)
        if classify == "sigmoid"
            class_function = sigmoid
        elseif classify == "softmax"
            class_function = softmax
        else
            error("Function to classify output labels must be \"sigmoid\" or \"softmax\".")
        end
    else
        class_function = sigmoid  # for one output (unit)
    end    
 
    a_test, a_wb_test, z_test = preallocate_feedfwd(inputs, theta, output_layer, n)
    predictions = feedfwd!(theta, output_layer, unit_function, class_function, a_test, 
        a_wb_test, z_test)
end


function printby2(nby2)  # not used currently
    for i = 1:size(nby2,1)
        println(nby2[i,1], "    ", nby2[i,2])
    end
end