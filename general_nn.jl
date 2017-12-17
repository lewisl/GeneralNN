#TODO
#   try dividing lambda by no. of parameters in a layer (even though it doesn't make any sense)
#   modify normalized leaky relu: add a linear transform
#       to the normalized result gamma*z + beta with rho and beta being trained for 
#       each unit
#   implement momentum
#   add early stopping
#   implement L1 regularization
#   add dropout
#   split stats from the plotdef
#   Create a more consistent testing regime:  independent validation set
#   scale weights for cost regularization to accommodate ReLU normalization?





# Includes the following functions to run directly:
#   train_nn() -- train sigmoid/softmax neural networks for up to 5 layers
#   test_score() -- cost and accuracy given test data and saved theta
#   save_theta() -- save theta, which is returned by train_nn
#   predictions_vector() -- predictions given x and saved theta


using MAT
using PyCall
using Plots
pyplot()  # initialize the backend used by Plots
@pyimport seaborn  # prettier charts
# using ImageView    BIG BUG HERE--SEGFAULT--REPORTED




"""
Method to call train_nn with a single integer as the number of hidden units
in a single hidden layer.
"""
function train_nn(matfname::String, n_iters::Int64, n_hid::Int64; alpha=0.35,
    mb_size=0, lambda=0.015, classify="softmax", units="sigmoid", plots=["Training", "Learning"])

    train_nn(matfname, n_iters, [n_hid], alpha=alpha,
    mb_size=mb_size, lambda=lambda, classify=classify, units=units, plots=plots)
end


"""
    function train_nn(matfname::String, n_iters::Int64, n_hid::Array{Int64,1}, alpha=0.35,
        mb_size=0, lambda=0.015, classify="softmax", units="sigmoid", plots=["Training", "Learning"])

    returns theta -- the model weights
    key inputs:
        alpha   ::= learning rate
        lambda  ::= regularization rate
        mb_size ::= mini-batch size

Train sigmoid/softmax neural networks up to 11 layers.  Detects
number of output labels from data. Detects number of features from data. Enables
any size mini-batches that divide evenly into number of examples.  Plots 
by any choice of per "Learning", "Training" iteration (epoch), and for "Test" data.
classify may be "softmax" or "sigmoid", which applies only to the output layer. 
units in other layers may be "sigmoid" or "relu".
"""
function train_nn(matfname::String, n_iters::Int64, n_hid::Array{Int64,1}; alpha=0.35,
    mb_size::Int64=0, lambda::Float64=0.015, classify::String="softmax", 
    units::String="sigmoid", plots::Array{String,1}=["Training", "Learning"])
    # creates a nn with 1 input layer, up to 9 hidden layers as an array input,
    # and output units matching the dimensions of the training data y
    # returns theta

    #check inputs
    if n_iters < 0
        error("Input n_iters must be an integer greater than 0")
    end

    if ndims(n_hid) != 1
        error("Input n_hid must be a vector.")
    elseif size(n_hid,1) > 9
        error("n_hid can only contain 1 to 9 integer values for 1 to 9 hidden layers.")
    end

    if alpha < 0.00001
        warn("Alpha learning rate set too small. Setting to default 0.35")
        alpha = 0.35
    elseif alpha > 1.0
        warn("Alpha learning rate set too large. Setting to defaut 0.35")
        alpha = 0.35
    end

    if mb_size < 0
        error("Input mb_size must be an integer greater than 0")
    end

    if lambda < 0.0
        warn("Lambda regularization rate must be positive floating point value. Setting to 0.")
        lambda = 0.0
    elseif lambda > 5.0
        warn("Lambda regularization rate set too large. Setting to defaut 5.0")
        lambda = 5.0
    end

    if !in(classify, ["softmax", "sigmoid"])
        warn("classify must be \"softmax\" or \"sigmoid\". Setting to default \"softmax\".")
        classify = "softmax"
    end

    if !in(units, ["relu", "sigmoid"])
        error("units must be \"relu\" or \"sigmoid\".")
    end

    valid_plots = ["Training", "Test", "Learning", "Cost"]
    new_plots = [pl for pl in valid_plots if in(pl, plots)]  
    if sort(new_plots) != sort(plots)
        warn("Plots argument can only include \"Training\", \"Test\", \"Learning\", and \"Cost\".\nProceeding with default [\"Training\", \"Learning\"].")
        new_plots = ["Training", "Learning"]
    end

    # read file and extract data
    inputs, targets, test_inputs, test_targets = extract_data(matfname)

    # create plot definition
    dotest = size(test_inputs, 1) > 0  # it's true there is test data
    plotdef = setup_plots(n_iters, dotest, new_plots)

    theta = run_training(inputs, targets, test_inputs, test_targets, n_iters, plotdef,
        n_hid, alpha, mb_size, lambda, classify, units);

    return theta;

end


function run_training(inputs, targets, test_inputs, test_targets, n_iters::Int64, plotdef,
    n_hid::Array{Int64,1}, alpha=0.35, mb_size=0, lambda=0.015, 
    classify="softmax", units="sigmoid")

    # this a nested function to isolate stats collection from the main line
    # all the outer function variables are available as if global except i, the loop
    # counter, which has loop scope and has to be passed
    function gather_stats!(i)  

        if plotdef["plot_switch"]["Training"]
            if plotdef["plot_switch"]["Cost"]
                plotdef["cost_history"][i, plotdef["col_train"]] = cost_function(mb_targets, 
                    mb_predictions, n, n, theta, lambda, output_layer)
            end
            if plotdef["plot_switch"]["Learning"]
                plotdef["fracright_history"][i, plotdef["col_train"]] = accuracy(
                    mb_targets, mb_predictions)
            end
        end
        
        if plotdef["plot_switch"]["Test"]
            if plotdef["plot_switch"]["Cost"]
                test_predictions[:] = feedfwd!(theta, bias, output_layer, unit_function!, class_function!,
                    a_test, z_test)  # a_wb_test, 
                plotdef["cost_history"][i, plotdef["col_test"]] = cost_function(test_targets, 
                    test_predictions, testn, testn, theta, lambda, output_layer)
            end
            if plotdef["plot_switch"]["Learning"]
                test_predictions[:] = feedfwd!(theta, bias, output_layer, unit_function!, class_function!,
                    a_test,  z_test)  # a_wb_test,
                plotdef["fracright_history"][i, plotdef["col_test"]] = accuracy(test_targets, test_predictions)
            end
        end     
    end  # function gather_stats

    ##########################################################
    #   function run_training main line
    ##########################################################

    # start the cpu clock
    tic()

    # seed random number generator.  For runs of identical models the same weight initialization
    # will be used, given the number of parameters to be estimated.
    srand(70653)  # seed int value is meaningless    

    # set some useful variables
    k,n = size(inputs)  # number of features k by no. of examples n
    t = size(targets,1)  # number of output units

    # layers
    n_hid_layers = size(n_hid, 1)
    output_layer = 2 + n_hid_layers # input layer is 1, output layer is highest value
    hid_layers = collect(2:2+n_hid_layers-1)
    layer_units = [k, n_hid..., t]

    # miscellaneous
    dotest = size(test_inputs, 1) > 0
        
    #setup mini-batch
    if mb_size == 0
        mb_size = n  # use 1 (mini-)batch with all of the examples
    elseif mb_size > n
        mb_size = n
    elseif mb_size < 1
        mb_size = n
    elseif mod(n, mb_size) != 0
        error("Mini-batch size $mb_size does not divide evenly into samples $n.")
    end
    n_mb = Int(n / mb_size)  # number of mini-batches 
    alphaovermb = alpha / mb_size  # calc once, use in loop

    if mb_size < n
        # randomize order of all training samples: 
            # labels in training data sometimes in a block, which will make
            # mini-batch train badly because a batch will not contain mix of labels
        sel_index = randperm(n)
        inputs[:] = inputs[:, sel_index]
        targets[:] = targets[:, sel_index]  
    end

    # pre-allocate mini-batch inputs and targets
    mb_inputs = zeros(size(inputs,1), mb_size)  
    mb_targets = zeros(size(targets,1), mb_size)  

    # prepare variable theta to hold network weights 
    # theta dimensions for each layer of the neural network 
    #    this follows the convention that the outputs of the current layer activation
    #    are rows of theta and the inputs from the layer below are columns
    theta_dims = [[k, 1]] # "weight" dimensions for the input layer--not used
    for i = 2:output_layer-1
        push!(theta_dims, [n_hid[i-1], theta_dims[i-1][1]]) 
    end
    push!(theta_dims, [t, n_hid[end]]) #  + 1 # weight dimensions for the output layer


    # initialize and pre-allocate data structures to hold neural net training data
    # theta = weight matrices in a collection for all calculated layers (e.g., not the input layer)
    # initialize random weight parameters including bias term
    theta = [zeros(2,2)] # initialize collection of 2d float arrays: input layer 1 not used
    interval = 0.5 # random weights will be generated in [-interval, interval]
    for i = 2:output_layer
        push!(theta, rand(theta_dims[i]...) .* (2.0 * interval) .- interval)
    end
    # initialize bias used to calculate z, input to activation, at every layer except input
    # bias = [rand(size(th, 1)) .* (2.0 * interval) .- interval for th in theta]
    bias = [zeros(size(th, 1)) for th in theta]

    # initialize batch normalization parameters gamma and beta
    # vector at each layer corresponding to no. of inputs from preceding layer, roughly "features"
    # gamma = scaling factor for normalization variance
    # beta = bias, or new mean instead of zero
    if units == "relu"  # for now, assume we always batch normalize relu
        gamma = [ones(i) for i in layer_units]  
        beta = [zeros(i) for i in layer_units]
    end

    # pre-allocate matrices used in training by layer to enable update-in-place for speed
    a, z = preallocate_feedfwd(inputs, theta, output_layer, n)  
    mb_a, mb_z = preallocate_feedfwd(mb_inputs, theta, output_layer, mb_size)  
    if dotest
        testn = size(test_inputs,2)
        a_test, z_test = preallocate_feedfwd(test_inputs, theta, output_layer, testn) 
        test_predictions = deepcopy(a_test[output_layer])
    end

    # choose or define functions to be used in neural net architecture
    if theta_dims[output_layer][1] > 1  # more than one output (unit)
        if classify == "sigmoid"
            class_function! = sigmoid!
        elseif classify == "softmax"
            class_function! = softmax!
        else
            error("Function to classify output labels must be \"sigmoid\" or \"softmax\".")
        end
    else
        class_function! = sigmoid!  # for one output (unit)
    end

    if units == "sigmoid"
        unit_function! = sigmoid!
    elseif units == "relu"
        unit_function! = n_l_relu!
    end

    if unit_function! == sigmoid!
        gradient_function! = sigmoid_gradient!
    elseif unit_function! == n_l_relu!
        gradient_function! = relu_gradient!
    end

    # setup cost function -- now there's only one.  someday there'll be others.
    cost_function = cross_entropy_cost

    if lambda > 0.0

    # println("sizes of weight matrices")
    # for th in theta
    #     print(size(th), " | ")
    # end
    # println()

    # println("sizes of bias matrices")
    # for th in bias
    #     print(size(th), " | ")
    # end
    # println()

    # println("sizes of a matrices")
    # for th in a
    #     print(size(th), " | ")
    # end
    # println()

    # println("sizes of z matrices")
    # for th in z
    #     print(size(th), " | ")
    # end
    # println()

    # println("sizes of gamma matrices")
    # for th in gamma
    #     print(size(th), " | ")
    # end
    # println()

    # error("done for now--get rid of this")

    # pre-allocate matrices for back propagation to enable update-in-place for speed
    epsilon = deepcopy(mb_a)  # looks like activations of each unit above input layer
    delta_w = deepcopy(theta)  # structure of gradient matches theta
    delta_b = deepcopy(bias)
    # initialize before loop to set scope OUTSIDE of loop
    mb_predictions = deepcopy(mb_a[output_layer])  # predictions = output layer values
    mb_grad = deepcopy(mb_z)


    # train the neural network and gather stats by iteration
    for i = 1:n_iters  # loop for "epochs"
        for j = 1:n_mb  # loop for mini-batches

            # grab the mini-batch subset of the data for the input layer 1
            first_example = (j - 1) * mb_size + 1
            last_example = first_example + mb_size - 1
            mb_a[1][:] = inputs[:,first_example:last_example]  # input layer activation for mini-batch
            mb_targets[:] = targets[:, first_example:last_example]         

            mb_predictions[:] = feedfwd!(theta, bias, output_layer, unit_function!, class_function!, mb_a, mb_z)  
            backprop_gradients!(theta, bias, mb_targets, unit_function!, gradient_function!, 
                output_layer, mb_a, mb_z, epsilon, delta_w, delta_b, mb_grad)  

            # update weights and bias
            @fastmath for il = 2:output_layer               
                if lambda > 0.0  # L2 regularization term added when lambda > 0
                    theta[il][:] = theta[il] .- ((alphaovermb .* delta_w[il]) .+ 
                        (lambda .* theta[il]))
                else
                    theta[il][:] -= alphaovermb .* delta_w[il]
                end
                
                bias[il][:] -= alphaovermb .* delta_b[il]
            end
        end

        gather_stats!(i)    # i has loop scope -- other variables available at outer function scope
    end
    
    # output training statistics
    toc()  # print cpu time since tic()
    predictions = feedfwd!(theta, bias, output_layer, unit_function!, class_function!, a, z)  # a_wb,
    println("Fraction correct labels predicted training: ", accuracy(targets, predictions))
    println("Final cost training: ", cost_function(targets, predictions, n,
                    n, theta, lambda, output_layer))

    # output improvement of last 10 iterations for test samples
    if plotdef["plot_switch"]["Test"]
        if plotdef["plot_switch"]["Learning"]
            println("Test data accuracy in final 10 iterations:")
            printdata = plotdef["fracright_history"][end-10+1:end, plotdef["col_test"]]
            for i=1:10
                @printf("%0.3f : ", printdata[i])
            end
            print("\n")
        end        
    end 

    # output test statistics
    if dotest     
        predictions = feedfwd!(theta, bias, output_layer, unit_function!, class_function!, a_test, 
            z_test) # a_wb_test,
        println("Fraction correct labels predicted test: ", accuracy(test_targets, predictions))
        println("Final cost test: ", cost_function(test_targets, predictions, testn, testn, theta, lambda, output_layer))
    end

    # plot the progress of training cost and/or learning
    plot_output(plotdef)

    return theta;

end  # function run_training


function preallocate_feedfwd(inputs, theta, output_layer, n)
    a = [inputs]
    z = [zeros(2,2)] # not used for input layer
    for i = 2:output_layer-1  # hidden layers
        push!(z, zeros(size(theta[i] * a[i-1])))  # z2 and up...  ...output layer set after loop
        push!(a, zeros(size(z[i])))  #  and up...  ...output layer set after loop
    end
    push!(z, zeros(size(theta[output_layer],1),n))  
    push!(a, zeros(size(theta[output_layer],1),n))  

    return a, z  
end


function feedfwd!(theta, bias, output_layer, unit_function!, class_function!, a, z) 
    # modifies a, a_wb, z in place to reduce memory allocations
    # send it all of the data or a mini-batch

    # feed forward from inputs to output layer predictions
    @fastmath for il = 2:output_layer-1  # hidden layers
        z[il][:] = theta[il] * a[il-1] .+ bias[il]
        unit_function!(z[il],a[il])
    end
    @fastmath z[output_layer][:] = theta[output_layer] * a[output_layer-1] .+ bias[output_layer]
    class_function!(z[output_layer], a[output_layer])
end


function backprop_gradients!(theta, bias, targets, unit_function!, gradient_function!,
    output_layer, a, z, epsilon, delta_w, delta_b, grad)  # a_wb, 
    # argument delta_w holds the computed gradients for weights, delta_b for bias
    # modifies epsilon, delta_w in place--caller uses delta_w, delta_b
    # use for iterations in training
    # send it all of the data or a mini-batch
    # intermediate storage of a, a_wb, z, epsilon, delta_w, delta_b reduces memory allocations

    epsilon[output_layer][:] = a[output_layer] .- targets 
    @fastmath delta_w[output_layer][:] = epsilon[output_layer] * a[output_layer-1]'
    @fastmath for jj = (output_layer - 1):-1:2  # for hidden layers
        gradient_function!(z[jj], grad[jj])
        epsilon[jj][:] = theta[jj+1]' * epsilon[jj+1] .* grad[jj]    # ient_function(z[jj]) 
        delta_w[jj][:] = epsilon[jj] * a[jj-1]'
        delta_b[jj][:] = sum(epsilon[jj],2)  # multiplying times a column of 1's is summing the row
    end

end


function cross_entropy_cost(targets, predictions, n, mb_size, theta, lambda, output_layer)
    # n is count of all samples in data set--use with regularization term
    # mb_size is count of all samples used in training batch--use with cost
    # these may be equal
    cost = (-1.0 ./ mb_size) .* sum(targets .* log.(predictions) .+ 
        (1.0 .- targets) .* log.(1.0 .- predictions))
    @fastmath if lambda > 0.0
        # need to fix hidden layer regularization for normalized ReLU
        # because the weights grow because the activations (z) are so small (mean = 0)
        # regterm = lambda/(2.0 * n) * sum([sum(theta[i][:, 2:end] .* theta[i][:, 2:end]) 
        #     for i in 2:output_layer]) 
        regterm = lambda/(2.0 * n) .* sum(theta[output_layer][:, 2:end] .* theta[output_layer][:, 2:end]) 
        cost = cost + regterm
    end
    return cost
end


function sigmoid!(z::Array{Float64,2}, a::Array{Float64,2}) 
    a[:] = 1.0 ./ (1.0 .+ exp.(-z))
end


function n_l_relu!(z::Array{Float64,2}, a::Array{Float64,2}) 
    # this is normalized leaky relu
    a[:] = (z .- mean(z,1)) ./ (std(z,1))
    for j = 1:size(z,2)  # down each column for speed
        for i = 1:size(z,1)
            @. a[i,j] = a[i,j] >= 0.0 ? a[i,j] : .01 * a[i,j]
        end
    end
end


function softmax!(z::Array{Float64,2}, a::Array{Float64,2})  # TODO try this with devec
    expf = similar(a)
    f = similar(a)
    f[:] = z .- maximum(z,1)  # atop => activation of top layer
    expf[:] = exp.(f)  # this gets called within a loop and exp() is expensive
    a[:] = @fastmath expf ./ sum(expf, 1)  
end


# @devec is worth a 5% improvement overall!, not just for this function
function sigmoid_gradient!(z::Array{Float64,2}, grad::Array{Float64,2})
    sigmoid!(z, grad)
    grad[:] = grad .* (1.0 .- grad)
end


function relu_gradient!(z::Array{Float64,2}, grad::Array{Float64,2})
    for j = 1:size(z, 2)  # calculate down a column for speed
        for i = 1:size(z, 1)
            grad[i,j] = z[i,j] > 0.0 ? 1.0 : .01
        end
    end
end


function plot_output(plotdef)
    # plot the progress of training cost and/or learning
    if (plotdef["plot_switch"]["Training"] || plotdef["plot_switch"]["Test"])

        if plotdef["plot_switch"]["Cost"]
            plt_cost = plot(plotdef["cost_history"], title="Cost Function", 
                labels=plotdef["plot_labels"], ylims=(0.0, Inf))
            display(plt_cost)  # or can use gui()
        end

        if plotdef["plot_switch"]["Learning"]
            plt_learning = plot(plotdef["fracright_history"], title="Learning Progress",
                labels=plotdef["plot_labels"], ylims=(0.0, 1.0), reuse=false) 
                # reuse=false  open a new plot window
            display(plt_learning)
        end

        if (plotdef["plot_switch"]["Cost"] || plotdef["plot_switch"]["Learning"])
            println("Press enter to close plot window..."); readline()
            closeall()
        end
    end
end


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
        test_inputs = df["test"]["x"]'  # note transpose operator
        test_targets = df["test"]["y"]'
    else
        test_inputs = zeros(0,0)
        test_targets = zeros(0,0)
    end
    return inputs, targets, test_inputs, test_targets
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

    valid_plots = ["Training", "Test", "Learning", "Cost"]
    plot_switch = Dict(pl => in(pl, plots) for pl in valid_plots)

    # must have test data to plot test results
    if dotest  # test data is present
        # nothing to change
    else
        if plot_switch["Test"]  # input requested plotting test data results
            warn("Can't plot test data. No test data. Proceeding.")
            plot_switch["Test"] = false
        end
    end

    plot_labels = [pl for pl in keys(plot_switch) if plot_switch[pl] == true &&
        (pl != "Learning" && pl != "Cost")]  # Cost, Learning are separate plots, not series labels
    plot_labels = reshape(plot_labels,1,size(plot_labels,1)) # 1 x N row array required by pyplot

    plotdef = Dict("plot_switch"=>plot_switch, "plot_labels"=>plot_labels)

    if plot_switch["Cost"]
        cost_history = zeros(n_iters, size(plot_labels,2))
        plotdef["cost_history"] = cost_history
    end
    if plot_switch["Learning"]
        fracright_history = zeros(n_iters, size(plot_labels,2))
        plotdef["fracright_history"] = fracright_history
    end
 
    # set column in cost_history for each data series
    col_train = plot_switch["Training"] ? 1 : 0
    col_test = plot_switch["Test"] ? col_train + 1 : 0

    plotdef["col_train"] = col_train
    plotdef["col_test"] = col_test

    return plotdef
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
            class_function! = sigmoid!
        elseif classify == "softmax"
            class_function! = softmax!
        else
            error("Function to classify output labels must be \"sigmoid\" or \"softmax\".")
        end
    else
        class_function! = sigmoid!  # for one output (unit)
    end    
 
    a_test,  z_test = preallocate_feedfwd(inputs, theta, output_layer, n)  # a_wb_test,
    predictions = feedfwd!(theta, output_layer, unit_function!, class_function!, a_test, 
        z_test) # a_wb_test, 
end


"""
Pass a dim1 x dim2 by 1 column vector holding the image data to display it.
Also pass the dimensions as 2 element vector (default is [28,28]).
"""
function display_mnist_digit(digit_data, digit_dims=[28,28])
    imshow(reshape(digit_data, digit_dims...)'); # transpose because inputs were transposed
    println("Press enter to close image window..."); readline()
    ImageView.closeall()
end


function printby2(nby2)  # not used currently
    for i = 1:size(nby2,1)
        println(nby2[i,1], "    ", nby2[i,2])
    end
end