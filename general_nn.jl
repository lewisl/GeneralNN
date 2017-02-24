# generalizing sigmoid neural networks for up to 5 layers

# TODO Find a better way to handle bias

using MAT
using Devectorize
using Plots
pyplot()

# This is a quicker way to call general_nn with only 1 hidden layer with n_hid units.
function general_nn(matfname::String, n_iters::Int64, n_hid::Int64, alpha=0.35, 
    classify="sigmoid", compare=false)
    general_nn(matfname, n_iters,[n_hid], alpha, classify, compare)
end

function general_nn(matfname::String, n_iters::Int64, n_hid::Array{Int64,1}, alpha=0.35,
    classify="sigmoid", compare=false)
    # creates a nn with 1 input layer, up to 3 hidden layers with n_hid units,
    # and a single output logistic neuron to do 1-way classification
    # returns theta

    # digits5000by784.mat",2000, [400,400], .4, sigmoid   .913 on test set

    # layers must be 1, 2, or 3 hidden layers. In addition, there will always be
    # 1 input layer and 1 output layer. So, total layers will be 3, 4, or 5.
    if ndims(n_hid) != 1
        error("Input n_hid must be a vector.")
    elseif size(n_hid,1) > 3
        error("n_hid can only contain 1 to 3 integer values for 1 to 3 hidden layers.")
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

    k,m = size(inputs) # number of features k by no. of examples m
    t = size(targets,1) # number of output units
    n_hid_layers = size(n_hid, 1)
    output_layer = 2 + n_hid_layers # input layer is 1, output layer is highest value
    
    # weight dims include bias term the 2nd index at each layer
    #    this follows the convention for theta: the outputs to the current layer activation
    #    are rows of theta and the inputs from the next layer down are columns
    w_dims = [[k, 1]] # "weight" dimensions for the input layer--not used
    for i = 2:output_layer-1
        push!(w_dims, [n_hid[i-1], w_dims[i-1][1]+1])
    end
    push!(w_dims, [t, n_hid[end]+ 1]) # weight dimensions for the output layer

    # use either sigmoid or softmax for output layer with multiple classification
    if w_dims[output_layer][1] > 1  # more than one output (unit)
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
    cost_history = zeros(n_iters+1)

    # theta = weight matrices for all calculated layers (except the input layer)
    # initialize random weight parameters including bias term
    theta = [zeros(2,2)] # initialize collection of 2d float arrays: input layer 1 not used
    interval = 0.5 # random weights will be generated in [-interval, interval]
    for i = 2:output_layer
        push!(theta, rand(w_dims[i]...) .* (2.0 * interval) .- interval)
    end

    # initialize matrices used in training by layer to enable update-in-place for speed
    a, a_wb, z = initialize_feedfwd(inputs, targets, theta, output_layer, m)

    # initialize matrices for back propagation
    eps = [rand(2,2)]  # not used at input layer
    delta = [rand(2,2)]  # not used at input layer
    for i = 2:output_layer
        push!(eps, zeros(size(a[i])))
        push!(delta, zeros(size(theta[i])))
    end

    # train the neural network and accumulate cost_history
    for i = 1:n_iters+1
        feedfwd!(theta, targets, output_layer, class_function,
            a, a_wb, z)
        predictions = a[output_layer]
        cost_history[i] = cost_function(targets, predictions, m)
        backprop!(theta, targets, output_layer, alpha, m, a, a_wb, z, eps, delta)
    end
    
    # output some statistics 
    # training data
    println("Fraction correct labels predicted training: ", score(targets, a[output_layer]))
    println("Final cost training: ", cost_history[end-1])

    # test statistics
    if in("test", keys(df))
        inputs = df["test"]["x"]'
        targets = df["test"]["y"]'
        testm = size(inputs,2)

        #feedforward for test data
        a, a_wb, z = initialize_feedfwd(inputs, targets, theta, output_layer, testm)
        feedfwd!(theta, targets, output_layer, class_function, a, a_wb, z)
        predictions = a[output_layer]
        println("Fraction correct labels predicted test: ", score(targets, predictions))
        println("Final cost test: ", cost_function(targets, predictions, m))
    end

    # plot the progress of training cost
    plot(cost_history[1:end-1], lab="Training Cost", ylims=(0, Inf))
    gui()

    if compare
        printby2(hcat(choices, targets))
    end

    return theta
end


function initialize_feedfwd(inputs, targets, theta, output_layer, m)
    a = [inputs]
    a_wb = [vcat(ones(1, m), inputs)] # input layer with bias column, never changed in loop
    z = [zeros(2,2)] # not used for input layer
    for i = 2:output_layer-1
        push!(z, zeros(size(theta[i] * a_wb[i-1])))  # z2 and up...  ...output layer set after loop
        push!(a, zeros(size(z[i])))  # a2 and up...  ...output layer set after loop
        push!(a_wb, vcat(ones(1, m), a[i]))  # a2_wb and up... ...but not output layer    
    end
    push!(z, similar(targets))  # z output layer z[output_layer]
    push!(a, similar(targets))  # a output layer a[output_layer]
    return a, a_wb, z
end


function feedfwd!(theta, targets, output_layer, class_function,
    a, a_wb, z)
    # modifies a, a_wb, z in place
    # send it all of the data or a mini-batch
    # receives intermediate storage of a, a_wb, z to reduce memory allocations
    # x is the activation of the input layer, a[1]
    # x[:] enables replace in place--reduce allocations, speed up loop

    # feed forward from inputs to output layer predictions
    @fastmath for ii = 2:output_layer-1
        z[ii][:] = theta[ii] * a_wb[ii-1]
        a[ii][:] = sigmoid(z[ii])
        a_wb[ii][2:end, :] = a[ii]  
    end
    z[output_layer][:] = theta[output_layer] * a_wb[output_layer-1]
    a[output_layer][:] = class_function(z[output_layer])
end


function backprop!(theta, targets, output_layer, alpha, m, a, a_wb, z, eps, delta)
    # modifies theta, eps, delta in place
    # use for iterations in training
    # send it all of the data or a mini-batch
    # receives intermediate storage of a, a_wb, z, eps, delta to reduce memory allocations

    for jj = output_layer:-1:2
        if jj == output_layer
            eps[jj][:] = a[jj] .- targets  # eps is epsilon, this is the output layer
        else
            eps[jj][:] = theta[jj+1][:, 2:end]' * eps[jj+1] .* sigmoid_gradient(z[jj])
        end
        delta[jj][:] = eps[jj] * a_wb[jj-1]'
        alphaoverm = alpha / m  # need this because @devec won't do division. 
        @devec gradterm = theta[jj] .- alphaoverm .* delta[jj]
        theta[jj][:] = gradterm
    end
end


function cross_entropy_cost(targets, predictions, m)
    @fastmath cost = -1.0 / m * sum(targets .* log(predictions) + 
            (1.0 .- targets) .* log(1.0 .- predictions))
    return cost
end


function sigmoid(z::Array{Float64,2})
    @devec  ret = 1.0 ./ (1.0 .+ exp(-z))
    return ret
end


function softmax(atop::Array{Float64,2})  # TODO try this with devec
    f = atop .- maximum(atop,1)
    # @devec ret = exp(f) ./ sum(exp(f), 1) # didn't work
    return exp(f) ./ sum(exp(f), 1)  
end


function sigmoid_gradient(z::Array{Float64,2})
    # derivative of sigmoid function
    sig = sigmoid(z)
    return sig .* (1 - sig)
end


function score(targets, predictions)
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


function printby2(nby2)
    for i = 1:size(nby2,1)
        println(nby2[i,1], "    ", nby2[i,2])
    end
end