# sample training run



println(
    """ 
    Sample run of neural net training with MNIST digits.
    """
)

println("........ Loading code ........")

println("\n........ enter runjob() -> code compiles the first time ........")

using Printf
using GeneralNN


function runjob(argfile="nninputs.toml", matfname="digits5000by400.mat"; testgrad=false)

    println("........ Loading training and test data ........")
    train_x, train_y, test_x, test_y = extract_data(matfname);  #

    # fix the funny thing that MAT file extraction does to the type of the arrays
    train_x = convert(Array{Float64,2}, train_x)
    train_y = convert(Array{Float64,2}, train_y)
    test_x = convert(Array{Float64,2}, test_x)
    test_y = convert(Array{Float64,2}, test_y)

    # println("size of test_x ", size(test_x))

    # shuffle in advance for training with minibatches and  batchnorm
    shuffle_data!(train_x, train_y)

    # debug
    # println("train_x ", typeof(train_x), " train_y ", typeof(train_y))

    println("........ Setup training ........")
    hp = setup_training([train_x, train_y, test_x, test_y], argfile)

    println("........ Training the neural network ........")
    if size(test_x) == (0,0)
        dotest = false
        results = train([train_x, train_y], hp, testgrad)
    else
        dotest = true
        results = train([train_x, train_y, test_x, test_y], hp, testgrad)
    end

    if testgrad
        println("\nNo training results because we stop after checking gradients.")
        return
    end

     # train_inputs, train_targets, train_preds, test_inputs, test_targets, test_preds, wgts, batchnorm_params, 
     # hyper_params

    dotest && (predmax = vec(map(x -> x[1], argmax(results["test_preds"],dims=1))));

    # which predictions are wrong?
    dotest && (test_wrongs = wrong_preds(results["test_targets"], results["test_preds"]));
    train_wrongs = wrong_preds(results["train_targets"], results["train_preds"]);

    dotest && println("\n\nThere are ", length(test_wrongs), " incorrect test predictions.")
    println("There are ", length(train_wrongs), " incorrect training predictions.")


    # look at wrong test predictions
    dotest && while true  
        println("\nPick a wrong test prediction to display or q to quit:")
        print("   enter a number between 1 and $(length(test_wrongs))> ")
        resp = chomp(readline())
        n = 0
        # response cases
        if resp == "q"
            break
        else
            try
                n = parse(Int, resp)
            catch
                continue
            end
            if n > length(test_wrongs) || n < 1
                println("Oops, try again...")
                continue
            else
                dodigit(n, test_wrongs, results["test_inputs"], results["test_targets"], predmax)
                continue
            end # response case: display a wrong prediction
        end  # response cases
    end  # prompt loop

    println("\nThat's all folks!.....")
    return results
end

####################################################################


