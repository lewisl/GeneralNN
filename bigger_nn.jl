# a very simple neural network example

using MAT
using Plots
pyplot()

function bigger_nn(matfname::String, n_iters::Int64, n_hid::Int64, alpha=.35, compare::Bool=false)
    # creates a nn with 2 inputs, a hidden layer with n-hid units,
    # and a single logistic neuron to do 1-way classification

    # read the data, load into variables, set some useful variables
    df = matread(matfname)
    inputs = df["x"]
    targets = df["y"]
    m, k = size(inputs) # number of examples by features
    layers = 3 # input layer is 1, output layer is highest value
    
    # weight dims for first hidden layer up to the output layer: 2 through layers
    # weight dims[1] is not used, but holds no. of features + bias
    # include bias term for the 2nd index, which is the no. of inputs to a layer
    w_dims = Dict([(1, (k, 1)), (2, (n_hid, k + 1)), (3, (1, n_hid+1))]) 
    cost = 0.0
    cost_history = zeros(n_iters+1)

    # initialize random weight parameters including bias term
    interval = 0.5 # random weights will be generated in [-interval, interval]
    theta2 = rand(w_dims[2]) .* (2.0 * interval) .- interval
    theta3 = rand(w_dims[3]) .* (2.0 * interval) .- interval

    # println("theta2: ", size(theta2))
    # println("theta3: ", size(theta3))

    # initialize variables used in for-loop to enable update-in-place for speed
    a1_wb = hcat(ones(m), inputs) # add bias term
    z2 = similar(a1_wb * theta2')
    a2 = similar(z2)
    a2_wb = [ones(m) a2]
    z3 = similar(targets)
    a3 = similar(targets)
    eps3 = similar(targets)
    eps2 = similar(a2)
    delta3 = similar(theta3)
    delta2 = similar(theta2)

    # train the neural network and accumulate statistics
    for i = 1:n_iters+1

        # feed forward from inputs to output layer predictions
            # a1_wb = inputs_bias # never changes across loop iterations
            # x[:] enables replace in place--reduce allocations, speed up loop
        z2[:] = a1_wb * theta2' 
        a2[:] = sigmoid(z2)
        a2_wb[:] = [ones(m) a2]  # a2 with bias
        z3[:] = a2_wb * theta3' 
        a3[:] = sigmoid(z3)


        cost = -1.0 / m * sum(targets .* log(a3) + (1.0 .- targets) .* log(1.0 .- a3))
        cost_history[i] = cost[1]

        # back propagation

        eps3[:] = a3 .- targets # eps is epsilon
        eps2[:] = eps3 * theta3[:, 2:end] .* sigmoid_gradient(z2)  
        delta3[:] = eps3' * a2_wb  

        delta2[:] = eps2' * a1_wb  
        theta3[:] = theta3 .- (alpha/m) .* delta3
        theta2[:] = theta2 .- (alpha/m) .* delta2


    end

    # output some statistics
    choices = zeros(size(a3,1))
    for i = 1:size(choices,1)
        choices[i,1] =  a3[i,1] >= 0.5 ? 1.0 : 0.0
    end
    # println("choices: ", size(choices))
    pct_right = mean(convert(Array{Float64},choices .== targets))
    println("Fraction correct labels predicted: ", pct_right)
    println("Final cost: ", cost[1])

    # plot the progress of cost
    plot(cost_history[1:end-1], lab="Cost")
    gui()

    if compare
        printby2(hcat(choices, targets))
    end

    return theta2, theta3
end

function sigmoid(z)
    return 1.0 ./ (1.0 .+ exp(-z))
end

function sigmoid_gradient(z)
    sig = sigmoid(z)
    return sig .* (1 - sig)
end

function printby2(nby2)
    for i = 1:size(nby2,1)
        println(nby2[i,1], "    ", nby2[i,2])
    end
end