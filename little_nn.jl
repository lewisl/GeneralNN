# a very simple neural network example

using MAT
using Plots
pyplot()


function little_nn(matfname,n_iters, alpha=.25, compare=false)
    # creates a nn with 2 inputs and a single logistic neuron to do 1-way classification

    # read the data, load into variables, set some useful variables
    df = matread(matfname)
    inputs = df["x"]
    targets = df["y"]

    m = size(inputs, 1) # number of examples
    inputs_bias = hcat(ones(m), inputs) # add bias term
    k = size(inputs_bias, 2) # number of features + 1 for bias term
    cost = 0.0
    accumulated_cost = zeros(n_iters+1)

    # initialize random weight parameters including bias term
    interval = .5 # random weights will be generated in [-interval, interval]
    theta = rand(3) .* (2*interval) .- interval
    # println(theta)
    # println("theta: ", size(theta))

    # train the neural network and accumulate statistics
    for i = 1:n_iters+1
        active = sigmoid(inputs_bias * theta) # activation value of neuron

        cost = -1.0 / m * sum(targets .* log(active) + (1.0 .- targets) .* log(1.0 .- active))
        accumulated_cost[i] = cost[1]
        eps = active .- targets # eps is epsilon
        gradients = reshape(alpha / m .* sum(inputs_bias .* eps, 1),(k,1))
        # println("gradients: ", size(gradients))
        theta = theta .- gradients
    end


    # output some statistics
    active = sigmoid(inputs_bias * theta)
    # println("active: ", size(active))
    # println("targets: ", size(targets))
    choices = zeros(size(active,1))
    for i = 1:size(choices,1)
        choices[i,1] =  active[i,1] > 0.5 ? 1.0 : 0.0
    end
    # println("choices: ", size(choices))
    pct_right = mean(convert(Array{Float64},choices .== targets))
    # println(theta)
    println("Fraction correct labels predicted: ", pct_right)
    println("Final cost: ", cost[1])
    # compare = hcat(targets, choices)
    # for i = 1:size(compare,1)
    #     println(compare[i,1], "    ", compare[i,2])
    # end

    # plot the progress of cost
    plot(accumulated_cost[2:n_iters],lab = "Cost", c=:blue)
    gui()
    # legend()

    if compare
        printby2(hcat(choices, targets))
    end

    return theta # inputs_bias, theta

end

function sigmoid(input)
    return 1.0 ./ (1.0 .+ exp(-input))
end

function printby2(nby2)
    for i = 1:size(nby2,1)
        println(nby2[i,1], "    ", nby2[i,2])
    end
end
