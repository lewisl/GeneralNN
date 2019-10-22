using Plots
using JLD2
using Printf
using LinearAlgebra
using Random
using MAT


"""
function extract_data(matfname::String, norm_mode::String="none")

Extract data from a matlab formatted binary file.

The matlab file may contain these keys: 
- "train", which is required, 
- "test", which is optional.

Within each top-level key the following keys must be present:
- "x", which holds the examples in variables as columns (no column headers should be used)
- "y", which holds the labels as columns for each example (it is possible to have multiple output columns for categories)

Multiple Returns:
-    inputs..........2d array of float64 with rows as features and columns as examples
-    targets.........2d array of float64 with columns as examples
-    test_inputs.....2d array of float64 with rows as features and columns as examples
-    test_targets....2d array of float64 with columns as examples
-    norm_factors.....1d vector of float64 containing [x_mu, x_std] or [m_min, x_max]
.....................note that the factors may be vectors for multiple rows of x

"""
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
    function shuffle_data!(x,y; returnidx = false)

Use this function to shuffle data once as inputs to training with minibatches.
The shuffle hyper-parameter of Train\\_nn is OK for small datasets or very
huge, sparse datasets because it is slow each time a minibatch is selected. 
Shuffling all of the training data BEFORE training and choosing shuffle=false
will be much faster.  Actually, that hyper-parameter is no longer supported.

shuffle_data shuffles the arrays in place.

x holds the training examples, assuming columns are examples and rows are features.
y holds the targets, assuming columns are examples.  y must have one row for 
  a single category (0,1 or -1, 1 or a float output for regression, etc.) or 
  mulitple rows for multiple categories, 
  e.g., one-hot encoding suitable for softmax classification.

Setting returnidx parameter to true returns the permuted index so that you can compare the shuffled training data to unshuffled source data.
"""
function shuffle_data!(x,y;returnidx = false)
    randidx = Random.randperm(size(x,2))
    x[:] = x[:,randidx]
    y[:] = y[:,randidx]
    returnidx && return randidx # return the permuted indices if true
end


"""

    function save_params(jld2_fname, nnw, bn, hp; train_preds=[], test_preds=[])

Save the trained parameters: nnw, batch_norm parameters: bn, and hyper parameters: hp,
as a JLD2 file.  Note:  be sure to use the jld2 file extension NOT jld as the formats are 
not compatible.

Can be used to run the model on prediction data or to evaluate other
test data results (cost and accuracy).
"""
function save_params(jld_fname, nnw, bn, hp; train_y=[], test_y=[])
    # check if output file exists and ask permission to overwrite
    if isfile(jld_fname)
        print("Output file $jld_fname exists. OK to overwrite? ")
        resp = readline()
        if contains(lowercase(resp), "y")
            rm(jld_fname)
        else
            error("File exists. Replied no to overwrite. Quitting.")
        end
    end

    # to translate to unnormalized regression coefficients: m = mhat / stdx, b = bhat - (m*xmu)

    # write the JLD formatted file (based on hdf5)
    jldopen(jld_fname, "w") do f
        f["nnp"] = nnw
        f["hp"] = hp
        f["bn"] = bn
        if size(train_y) != (0, ) 
              f["train_y"] = train_y 
        end
        if size(test_y) != (0, )
            f["test_y"] = test_y 
        end
    end

end


"""

    function load_params(jld_fname)

Load the trained parameters: nnw, batch_norm parameters: bn, and hyper parameters: hp,
from a JLD file.

Can be used to run the model on prediction data or to evaluate other
test data results (cost and accuracy).

returns: nnw, bn, hp
These are mutable structs.  Use fieldnames(nnw) to list the fields.
"""
function load_params(jld_fname)
    f = jldopen(jld_fname, "r")
    ret = Dict(j=>f[j] for j in keys(f))
    close(f)
    f = []   # flush it before gc gets to it
    return ret
end


function save_plotdef(plotdef; fname="")
    if fname == ""  # build a filename
        fname = repr(Dates.now())
        fname = "plotdef-" * replace(fname, r"[.:]" => "-") * ".jld2"
    end
    jldopen(fname, "w") do f
        f["plotdef"] = plotdef
    end
end


function load_plotdef(fname::String)
    f = jldopen(fname, "r")
    plotdef = f["plotdef"]
    close(f)
    return plotdef
end


function printby2(nby2)  # not used currently
    for i = 1:size(nby2,1)
        println(nby2[i,1], "    ", nby2[i,2])
    end
end


"""Print the sizes of matrices you pass in as Dict("name"=>var,...)"""
function printdims(indict)
    println("\nSizes of matrices\n")
    for (n, item) in indict
        println(n,": ",size(item))
    end
    println("")
end


"""
Pass a dim1 x dim2 by 1 column vector holding the image data to display it.
Also pass the dimensions as 2 element vector (default is [28,28]).
"""
function display_mnist_digit(digit_data, digit_dims=[28,28])
    # plotlyjs(size=(400,400))
    gr()
    clibrary(:misc)  # collection of color palettes
    img = reshape(digit_data, digit_dims...)'
    pldigit = plot(img, seriestype=:heatmap, color=:grays,  
        showaxis=false, legend=false, yflip=true, size=(500,500)) # right_margin=6mm, bottom_margin=6mm
    display(pldigit)
    println("Press enter to close image window..."); readline()
    closeall()
end


"""
    Save, print and plot training statistics after all epochs

"""
function output_stats(datalist, nnw, bn, hp, training_time, plotdef)

    if size(datalist, 1) == 1
        train = datalist[1]
        dotest = false
    elseif size(datalist,1) == 2
        train = datalist[1]
        test = datalist[2]
        dotest = true
    else
        error("Datalist contains wrong number of elements.")
    end


    # file for simple training stats
    fname = repr(Dates.now())
    fname = "nnstats-" * replace(fname, r"[.:]" => "-") * ".txt"
    println(fname)

    # print to file and console
    open(fname, "w") do stats
        println(stats, "Training time: ",training_time, " seconds")  # cpu time since tic() =>  toq() returns secs without printing

        # output for entire training set
        feedfwd!(train, nnw, bn, hp, istrain=false)  
        println(stats, "Fraction correct labels predicted training: ",
                hp.classify == "regression" ? r_squared(train.targets, train.a[nnw.output_layer])
                    : accuracy(train.targets, train.a[nnw.output_layer]))
        println(stats, "Final cost training: ", cost_function(train.targets, train.a[nnw.output_layer], train.n,
                        nnw.theta, hp, nnw.output_layer))

        # output improvement of last few iterations for training data
        if plotdef["plot_switch"]["train"]
            if plotdef["plot_switch"]["learning"]
                tailcount = min(10, hp.epochs)
                println(stats, "Training data accuracy in final $tailcount iterations:")
                printdata = plotdef["accuracy"][end-tailcount+1:end, plotdef["train"]]
                for i=1:tailcount
                    @printf(stats, "%0.3f : ", printdata[i])
                end
                print("\n\n")
            end
        end

        # output test statistics
        if dotest
            feedfwd!(test, nnw, bn,  hp, istrain=false)
            println(stats, "\n\nFraction correct labels predicted test: ",
                    hp.classify == "regression" ? r_squared(test.targets, test.a[nnw.output_layer])
                        : accuracy(test.targets, test.a[nnw.output_layer]))
            println(stats, "Final cost test: ", cost_function(test.targets, test.a[nnw.output_layer], test.n,
                nnw.theta, hp, nnw.output_layer))
        end

        # output improvement of last 10 iterations for test data
        if plotdef["plot_switch"]["test"]
            if plotdef["plot_switch"]["learning"]
                tailcount = min(10, hp.epochs)
                println(stats, "Test data accuracy in final $tailcount iterations:")
                printdata = plotdef["accuracy"][end-tailcount+1:end, plotdef["test"]]
                for i=1:tailcount
                    @printf(stats, "%0.3f : ", printdata[i])
                end
                print("\n\n")
            end
        end

        # output number of incorrect predictions
        train_wrongs = GeneralNN.wrong_preds(train.targets, train.a[nnw.output_layer]);
        println(stats, "\nThere are ", length(train_wrongs), " incorrect training predictions.")
        if dotest
            test_wrongs = GeneralNN.wrong_preds(test.targets, test.a[nnw.output_layer]);
            println(stats, "\nThere are ", length(test_wrongs), " incorrect test predictions.")
        end

        # output hyper hyper_parameters
        hp.alpha = (   hp.do_learn_decay   # back out alpha to original input
                        ? round(hp.alpha * (1.0 / hp.learn_decay[1]) ^ (hp.learn_decay[2] - 1.0), digits=5) 
                        : hp.alpha    )
        println(stats, "\nHyper-parameters")
        pretty_print_hp(stats, hp)

    end  # end do stats  --  done with stats stream->closed

    # print the stats
    println(read(fname, String))

    # save cost and accuracy from training
    save_plotdef(plotdef)
    
    # plot now?
    hp.plot_now && plot_output(plotdef)

end


"""
    plot_output(plotdef::Dict)

    Plots the plotdef and creates 1 or 2 PyPlot plot windows of the learning (accuracy)
    and/or cost from each training epoch.

"""
function plot_output(plotdef::Dict)
    # plot the progress of training cost and/or learning
    if (plotdef["plot_switch"]["train"] || plotdef["plot_switch"]["test"])
        # plotlyjs(size=(600,400)) # set chart size defaults
        gr()

        if plotdef["plot_switch"]["cost"]
            plt_cost = plot(
                plotdef["cost_history"], 
                title="Cost Function",
                labels=plotdef["plot_labels"], 
                legend=:bottomright,
                ylims=(0.0, Inf), 
                bottom_margin=7mm, 
                size=(600,400))
            display(plt_cost)  # or can use gui()
        end

        if plotdef["plot_switch"]["learning"]
            plt_learning = plot(
                plotdef["accuracy"], 
                title="Learning Progress",
                labels=plotdef["plot_labels"], 
                legend=:bottomright,
                ylims=(0.0, 1.05), 
                bottom_margin=7mm,
                size=(600,400)) 
            display(plt_learning)
        end

        if (plotdef["plot_switch"]["cost"] || plotdef["plot_switch"]["learning"])
            println("Press enter to close plot window..."); readline()
            closeall()
        end
    end
end

"""
Print and plot a predicted digit and the actual digit.
"""
function dodigit(n, test_wrongs, test_inputs, test_targets, predmax)
    example = test_wrongs[n]
    digit_data = test_inputs[:, example]
    correct = findmax(test_targets[:,example])[2]  # get the row in the column that contains value 1
    correct = correct == 10 ? 0 : correct
    predicted = predmax[example]
    predicted = predicted == 10 ? 0 : predicted
    println("\n\nThe neural network predicted: $predicted")
    println("The correct value is: $correct")
    GeneralNN.display_mnist_digit(digit_data, [28,28])
end


"""
    function onehot(vect, cnt; dim=1)
    function onehot(vect, cnt, result_type; dim=1)

Create a matrix of onehot vectors with cnt categories from a vector that
contains the category number from 1 to cnt.
Elements of the returned matrix will have eltype of vect or of the
argument result_type.
"""
function onehot(vect, cnt; dim=1)
    et = eltype(vect)
    onehot(vect, cnt, et, dim=dim)
end


function onehot(vect,cnt,result_type; dim=1)
    et = result_type

    # if ndims(vect) == 1
    # end

    eye = zeros(et, cnt,cnt)
    setone = convert(et,1)
    for i = 1:cnt
        eye[i,i] = setone
    end
    if dim == 1
        return eye[vect,:]
    else
        return eye[:,vect]
    end
end


# to file and console
function pretty_print_hp(io, hp)
    for item in fieldnames(typeof(hp))
        @printf(io, "%16s = %s\n", item, getfield(hp,item))
    end
end

# to console
function pretty_print_hp(hp)
    for item in fieldnames(typeof(hp))
        @printf("%16s = %s\n",item, getfield(hp,item))
    end
end

"""
    function indmax(arr; dim::Int=1)

Simplifies obtaining index of maximum value:

    dim=1 returns a ROW, index of maximum ROW of each column  
    dim=2 returns a COL, index of maximum COL of each row  

(Note: like argmax, but extracts single value from CartesianIndex.)

"""
function indmax(arr; dims::Int=1)
    ret = findmax(arr,dims=dims)[2]
    map(x->x[dims],ret)
end


function toml_test(fn)
    return TOML.parsefile(fn)
end



##############################################################
#
#    TEST EVERYTHING BELOW:  PROBABLY DON'T WORK
#
##############################################################




function wrong_preds(targets, preds, cf = !isequal)
    if size(targets,1) > 1
        # targetmax = ind2sub(size(targets),vec(findmax(targets,dims=1)[2]))[1]
        # predmax = ind2sub(size(preds),vec(findmax(preds,dims=1)[2]))[1]
        targetmax = vec(map(x -> x[1], argmax(targets,dims=1)));
        predmax = vec(map(x -> x[1], argmax(preds,dims=1)));
        wrongs = findall(cf.(targetmax, predmax))
    else
        # works because single output unit is sigmoid--well, what do we do about regression? we use r_squared
        choices = [j >= 0.5 ? 1.0 : 0.0 for j in preds]
        wrongs = findall(cf.(choices, targets))
    end
    return wrongs
end


function right_preds(targets, preds)
    return wrong_preds(targets, preds, isequal)
end



"""

    function test_score(theta_fname, data_fname, lambda = 0.01,
        classify="softmax")

Calculate the test accuracy score and cost for a dataset containing test or validation
data.  This data must contain outcome labels, e.g.--y.
"""
function test_score(theta_fname, data_fname, lambda = 0.01, classify="softmax")
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
    println("Final cost test: ", cost_function(targets, predictions, n, theta, hp, output_layer))

    return score
end


"""

    function nnpredict()

Two methods:
    with a .mat file as input: 
        function predict(matfname::String, hp, nnw, bn; test::Bool=false, norm_mode::String="")
    with arrays as input:
        function predict(inputs, targets, hp, nnw, bn, norm_factors)

Generate predictions given previously trained parameters and input data.
Not suitable in a loop because of all the additional allocations.
Use with one-off needs like scoring a test data set or
producing predictions for operational data fed into an existing model.
"""
function nnpredict(matfname::String, hp, nnw, bn; test::Bool=false)
    if !test
        inputs, targets, _, __ = extract_data(matfname)  # training data
    else
        _, __, inputs, targets = extract_data(matfname)  # test data
    end

    nnpredict(inputs, targets, hp, nnw, bn)
end


function nnpredict(inputs, targets, hp, nnw, bn, istest)
    dataset = Model_data()
        dataset.inputs = inputs
        dataset.targets = targets
        dataset.in_k, dataset.n = size(inputs)  # number of features in_k (rows) by no. of examples n (columns)
        dataset.out_k = size(dataset.targets,1)  # number of output units

    if hp.norm_mode == "standard" || hp.norm_mode == "minmax"
        normalize_replay!(dataset.inputs, hp.norm_mode, nnw.norm_factors)
    end

    preallocate_data!(dataset, nnw, dataset.n, hp)

    setup_functions!(hp.units, dataset.out_k, hp.opt, hp.classify, istest)  # for feedforward calculations

    feedfwd!(dataset, nnw, bn, hp, istrain=false)  # output for entire dataset

    println("Fraction correct labels predicted: ",
        hp.classify == "regression" ? r_squared(dataset.targets, dataset.a[nnw.output_layer])
                                    : accuracy(dataset.targets, dataset.a[nnw.output_layer], hp.epochs))
    return dataset.a[nnw.output_layer]
end