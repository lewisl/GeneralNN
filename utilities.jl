using Plots
using JLD2

"""

    function save_params(jld2_fname, nnp, bn, hp; train_preds=[], test_preds=[])

Save the trained parameters: nnp, batch_norm parameters: bn, and hyper parameters: hp,
as a JLD2 file.  Note:  be sure to use the jld2 file extension NOT jld as the formats are 
not compatible.

Can be used to run the model on prediction data or to evaluate other
test data results (cost and accuracy).
"""
function save_params(jld_fname, nnp, bn, hp; train_y=[], test_y=[])
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
        f["nnp"] = nnp
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

Load the trained parameters: nnp, batch_norm parameters: bn, and hyper parameters: hp,
from a JLD file.

Can be used to run the model on prediction data or to evaluate other
test data results (cost and accuracy).

returns: nnp, bn, hp
These are mutable structs.  Use fieldnames(nnp) to list the fields.
"""
function load_params(jld_fname)
    f = jldopen(jld_fname, "r")
    ret = Dict(j=>f[j] for j in keys(f))
    close(f)
    f = []   # flush it before gc gets to it
    return ret
end


function save_plotdef(plotdef)
    fname = repr(Dates.now())
    fname = "plotdef-" * replace(fname, r"[.:]" => "-") * ".jld2"
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
        showaxis=false, legend=false, yflip=true, size=(400,400)) # right_margin=6mm, bottom_margin=6mm
    display(pldigit)
    println("Press enter to close image window..."); readline()
    closeall()
end


"""
    Save, print and plot training statistics after all epochs

"""
function output_stats(train, test, nnp, bn, hp, training_time, dotest, plotdef, plot_now)

    # file for simple training stats
    fname = repr(Dates.now())
    fname = "nnstats-" * replace(fname, r"[.:]" => "-") * ".txt"
        println(fname)
    open(fname, "w") do stats
        println(stats, "Training time: ",training_time, " seconds")  # cpu time since tic() =>  toq() returns secs without printing

        feedfwd!(train, nnp, bn, hp, istrain=false)  # output for entire training set
        println(stats, "Fraction correct labels predicted training: ",
                hp.classify == "regression" ? r_squared(train.targets, train.a[nnp.output_layer])
                    : accuracy(train.targets, train.a[nnp.output_layer],hp.epochs))
        println(stats, "Final cost training: ", cost_function(train.targets, train.a[nnp.output_layer], train.n,
                        nnp.theta, hp, nnp.output_layer))

        # output test statistics
        if dotest
            feedfwd!(test, nnp, bn,  hp, istrain=false)
            println(stats, "Fraction correct labels predicted test: ",
                    hp.classify == "regression" ? r_squared(test.targets, test.a[nnp.output_layer])
                        : accuracy(test.targets, test.a[nnp.output_layer], hp.epochs))
            println(stats, "Final cost test: ", cost_function(test.targets, test.a[nnp.output_layer], test.n,
                nnp.theta, hp, nnp.output_layer))
        end

        # output improvement of last 10 iterations for test data
        if plotdef["plot_switch"]["Test"]
            if plotdef["plot_switch"]["Learning"]
                println(stats, "Test data accuracy in final 10 iterations:")
                printdata = plotdef["fracright_history"][end-10+1:end, plotdef["col_test"]]
                for i=1:10
                    @printf(stats, "%0.3f : ", printdata[i])
                end
                print("\n")
            end
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
    plot_now && plot_output(plotdef)

end


"""
    plot_output(plotdef::Dict)

    Plots the plotdef and creates 1 or 2 PyPlot plot windows of the learning (accuracy)
    and/or cost from each training epoch.

"""
function plot_output(plotdef::Dict)
    # plot the progress of training cost and/or learning
    if (plotdef["plot_switch"]["Training"] || plotdef["plot_switch"]["Test"])
        # plotlyjs(size=(600,400)) # set chart size defaults
        gr()

        if plotdef["plot_switch"]["Cost"]
            plt_cost = plot(plotdef["cost_history"], title="Cost Function",
                labels=plotdef["plot_labels"], ylims=(0.0, Inf), bottom_margin=7mm, size=(400,400))
            display(plt_cost)  # or can use gui()
        end

        if plotdef["plot_switch"]["Learning"]
            plt_learning = plot(plotdef["fracright_history"], title="Learning Progress",
                labels=plotdef["plot_labels"], ylims=(0.0, 1.05), bottom_margin=7mm) 
            display(plt_learning)
        end

        if (plotdef["plot_switch"]["Cost"] || plotdef["plot_switch"]["Learning"])
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
