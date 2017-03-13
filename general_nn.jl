# train_nn() -- generalizing sigmoid/softmax neural networks for up to 5 layers
# test_score() -- cost and accuracy for test data and saved theta
# save_theta() -- save theta, which is returned by train_nn
# predictions_vector() -- predictions given x and saved theta


using MAT
using Devectorize
using PyCall
using Plots
pyplot()  # initialize the backend used by Plots
@pyimport seaborn  # prettier charts

# This is a quicker way to call train_nn with only 1 hidden layer of n_hid units.
function train_nn(matfname::String, n_iters::Int64, n_hid::Int64, alpha=0.35, mb_size=0,
    lambda=0.015, classify="softmax", plots=["Training"])
    train_nn(matfname, n_iters,[n_hid], alpha, mb_size, lambda, classify, plots)
end


"""
Generalized training for sigmoid/softmax neural networks up to 5 layers.  Detects
number of output labels from data. Detects number of features from data. Enables
any size mini-batches that divide evenly into number of examples.  Plots cost
by any choice of per mini-batch, training iteration (epoch), and for test data.
"""
function train_nn(matfname::String, n_iters::Int64, n_hid::Array{Int64,1}, alpha=0.35,
    mb_size=0, lambda=0.015, classify="softmax", plots=["Training"])
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
        end

        # read the data, load into variables, set some useful variables
        df = matread(matfname)

        # Split into train and test datasets, if we have them
        if in("train", keys(df))
            inputs = df["train"]["x"]'  # set examples as columns to optimize for julia column-dominant operations
            targets = df["train"]["y"]'
        else
            inputs = df["x"]'
            targets = df["y"]'
        end
        if in("test", keys(df))
            dotest = true
            test_inputs = df["test"]["x"]'
            test_targets = df["test"]["y"]'
            testn = size(test_inputs,2)
        else
            dotest = false
        end

        k,n = size(inputs)  # number of features k by no. of examples n
        t = size(targets,1)  # number of output units
        n_hid_layers = size(n_hid, 1)
        output_layer = 2 + n_hid_layers # input layer is 1, output layer is highest value
        mb = mb_size  # shortcut
        lamovern = lambda / n  # need this because @devec won't do division. 
        
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

        mb_inputs = zeros(size(inputs,1), mb_size)  # pre-allocate mini-batch of inputs
        mb_targets = zeros(size(targets,1), mb_size)  # pre-allocate mini-batch of targets

    # set up cost_history to track 1, 2, or 3 cost series
        plot_labels = []
        if size(plots,1) > 3
            warn("Only 3 plot requests permitted. Proceeding with up to 3.")
        end

        valid_plots = ["Training", "Test", "Mini-batch"]
        plot_switch = Dict(i => false for i in valid_plots)
        for it in plots
            if in(it, valid_plots)
                if !in(it, plot_labels)  # can only add a plot once!
                    push!(plot_labels, it)
                    plot_switch[it] = true
                end
            else
                warn("plots argument can only include \"Training\", \"Test\", and \"Mini-batch\". Proceeding.")
            end
        end
        if plot_switch["Test"]
            if !in("test", keys(df))  # must have test data
                warn("Can't plot test cost. No test data. Proceeding.")
                plot_switch["Test"] = false
                deleteat!(plot_labels, findin(plot_labels, ["Test"]))
            end
        end

        sz = plot_switch["Mini-batch"] ? n_iters * n_mb : n_iters
        cost_history = zeros(sz, size(plot_labels,1))
        plot_labels = reshape(plot_labels, 1, size(plot_labels,1)) # make 1 row of columns
        # which column to save the cost data to be plotted?
        col_mb = plot_switch["Mini-batch"] ? 1 : 0
        col_tra = plot_switch["Training"] ? col_mb + 1 : 0
        col_test = plot_switch["Test"] ? col_mb + col_tra + 1 : 0
        col_test = col_test > 3 ? 3 : col_test

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

        # setup cost
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
            a_test, a_wb_test, z_test = preallocate_feedfwd(test_inputs, theta, output_layer, testn)
        end

        # initialize matrices for back propagation to enable update-in-place for speed
        eps = deepcopy(mb_a)  # looks like activations of each unit above input layer
        delta = deepcopy(theta)  # structure of gradient matches theta

    # train the neural network and accumulate mb_cost_history
    for i = 1:n_iters  # loop for "epochs"
        for j = 1:n_mb  # loop for mini-batches
            start = (j - 1) * mb_size + 1
            fin = start + mb_size - 1
            mb_a[1][:] = inputs[:,start:fin]  # input layer activation for mini-batch
            mb_a_wb[1][:] = vcat(ones(1,mb_size), mb_a[1])
            mb_targets[:] = targets[:, start:fin]         

            feedfwd!(theta, output_layer, class_function, mb_a, mb_a_wb, mb_z)  
            backprop_gradients!(theta, mb_targets, output_layer, 
                mb_size, mb_a, mb_a_wb, mb_z, eps, delta)  

            # calculate new theta
            @fastmath for ii = 2:output_layer
                if lambda > 0.0  # use regularization
                    delta[ii][:, 2:end] = delta[ii][:, 2:end] .- lamovern .* theta[ii][:,2:end]  # don't regularize bias
                end
                theta[ii][:] = theta[ii] .- alpha .* delta[ii]  # the reg term will be added
            end
            if plot_switch["Mini-batch"]
                predictions = feedfwd!(theta, output_layer, class_function, mb_a, mb_a_wb, mb_z) 
                cost_history[(i-1)*n_mb+j,1] = cost_function(mb_targets, predictions, n,
                    mb_size, theta, lambda, output_layer)
            end
        end

        if plot_switch["Training"]
            predictions = feedfwd!(theta, output_layer, class_function, a, a_wb, z) 
            beg = plot_switch["Mini-batch"] ? (i-1)*n_mb+1 : i
            fin = plot_switch["Mini-batch"] ? i*n_mb : i
            cost_history[beg:fin, col_tra] = cost_function(targets, predictions, n,
                n, theta, lambda, output_layer)
        end
        if plot_switch["Test"]
            predictions = feedfwd!(theta, output_layer, class_function, a_test, a_wb_test, z_test)
            beg = plot_switch["Mini-batch"] ? (i-1)*n_mb+1 : i
            fin = plot_switch["Mini-batch"] ? i*n_mb : i
            cost_history[beg:fin, col_test] = cost_function(test_targets, predictions, testn,
                testn, theta, lambda, output_layer)
        end
    end
    
    # output training statistics
    predictions = feedfwd!(theta, output_layer, class_function, a, a_wb, z)
    println("Fraction correct labels predicted training: ", accuracy(targets, predictions))
    println("Final cost training: ", cost_function(targets, predictions, n,
                    n, theta, lambda, output_layer))

    # output test statistics
    if dotest     
        predictions = feedfwd!(theta, output_layer, class_function, a_test, a_wb_test, z_test)
        println("Fraction correct labels predicted test: ", accuracy(test_targets, predictions))
        println("Final cost test: ", cost_function(test_targets, predictions, testn, testn, theta, lambda, output_layer))
    end

    # plot the progress of training cost
    if (plot_switch["Mini-batch"] || plot_switch["Training"] || plot_switch["Test"])
        plt = plot(cost_history, title="Cost Function", labels=plot_labels, ylims=(0, Inf))
        display(plt)  # or can use gui()
        println("Press enter to close plot window..."); readline()
        closeall()
    end

    return theta
end


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


function feedfwd!(theta, output_layer, class_function, a, a_wb, z)
    # modifies a, a_wb, z in place
    # send it all of the data or a mini-batch
    # receives intermediate storage of a, a_wb, z to reduce memory allocations
    # x is the activation of the input layer, a[1]
    # x[:] enables replace in place--reduce allocations, speed up loop

    # feed forward from inputs to output layer predictions
    @fastmath for ii = 2:output_layer-1  # ii is the current layer
        z[ii][:] = theta[ii] * a_wb[ii-1]
        a[ii][:] = sigmoid(z[ii])
        a_wb[ii][2:end, :] = a[ii]  
    end
    @fastmath z[output_layer][:] = theta[output_layer] * a_wb[output_layer-1]
    a[output_layer][:] = class_function(z[output_layer])
end


function backprop_gradients!(theta, targets, output_layer, mb, a, a_wb, z, eps, delta)
    # argument delta holds the computed gradients
    # modifies theta, eps, delta in place--caller uses delta
    # use for iterations in training
    # send it all of the data or a mini-batch
    # receives intermediate storage of a, a_wb, z, eps, delta to reduce memory allocations

    oneovermb = 1.0 / mb

    eps[output_layer][:] = a[output_layer] .- targets  # eps is epsilon
    @fastmath delta[output_layer][:] = oneovermb .* (eps[output_layer] * a_wb[output_layer-1]')
    @fastmath for jj = (output_layer - 1):-1:2  # don't do input layer
        eps[jj][:] = theta[jj+1][:, 2:end]' * eps[jj+1] .* sigmoid_gradient(z[jj]) 
        delta[jj][:] = oneovermb .* (eps[jj] * a_wb[jj-1]') 
    end
end


# suitable cost function for sigmoid and softmax
function cross_entropy_cost(targets, predictions, n, mb, theta, lambda, output_layer)
    # n is count of all samples in data set--use with regularization term
    # mb is count of all samples used in training batch--use with cost
    # these may be equal
    @devec cost = (-1.0 ./ mb) .* sum(targets .* log(predictions) + 
        (1.0 .- targets) .* log(1.0 .- predictions))
    @fastmath if lambda != 0.0
        regterm = lambda/(2.0 * n) * sum([sum(theta[i][:, 2:end] .* theta[i][:, 2:end]) 
            for i in 2:output_layer]) 
        cost = cost + regterm
    end
    return cost
end


function sigmoid(z::Array{Float64,2})
    @devec  ret = 1.0 ./ (1.0 .+ exp(-z))
    return ret
end


function softmax(atop::Array{Float64,2})  # TODO try this with devec
    f = atop .- maximum(atop,1)  # atop => activation of top layer
    expf = exp(f)  # this gets called within a loop and exp() is expensive
    return @fastmath expf ./ sum(expf, 1)  
end


function sigmoid_gradient(z::Array{Float64,2})
    # derivative of sigmoid function
    sig = sigmoid(z)
    @devec ret = sig .* (1 .- sig)
    return ret  # sig .* (1 - sig)
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
Save the weights, or theta, trained by the neural network as a matlab file.
Can be used to run the model on prediction data or to evaluate other 
test data outcomes.
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
Calculate the test score and cost for a dataset containing test or validation
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
Returns predicted values given inputs and theta.  Not used by training.
Use when theta is already trained and saved to make predictions
for input operational data to use your model. Resolves sigmoid or softmax outputs
to zero, one values for each output unit.

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
Simplified function to generate predictions given theta and inputs.
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
    predictions = feedfwd!(theta, output_layer, class_function, a_test, a_wb_test, z_test)
end


function printby2(nby2)  # not used currently
    for i = 1:size(nby2,1)
        println(nby2[i,1], "    ", nby2[i,2])
    end
end