# sample training run



println(
    """ 
    Sample run of neural net training with MNIST digits.
    """
)

println("........ Loading code ........")

println("\n........ enter runjob() -> code compiles the first time ........")

using Printf
using JSON
include("GeneralNN.jl")
using .GeneralNN


function runjob(jsoninputs="nninputs.json", matfname="digits60000by784.mat")

    println("........ Loading training and test data ........")
    train_x, train_y, test_x, test_y = extract_data(matfname);

    # fix the funny thing that MAT file extraction does to the type of the arrays
    train_x = convert(Array{Float64,2}, train_x)
    train_y = convert(Array{Float64,2}, train_y)
    test_x = convert(Array{Float64,2}, test_x)
    test_y = convert(Array{Float64,2}, test_y)

    # debug
    # println("train_x ", typeof(train_x), " train_y ", typeof(train_y))

    println("........ Training the neural network ........")
    results = train_nn(train_x, train_y, jsoninputs)
     # train_inputs, train_targets, train_preds, test_inputs, test_targets, test_preds, Wgts, batchnorm_params, 
     # hyper_params


    predmax = vec(map(x -> x[1], argmax(results["train_preds"],dims=1)));

    # which predictions are wrong?
    # test_wrongs = GeneralNN.wrong_preds(results["test_targets"], results["test_preds"]);
    train_wrongs = GeneralNN.wrong_preds(results["train_targets"], results["train_preds"]);

    # println("\n\nThere are ", length(test_wrongs), " incorrect test predictions.")
    println("There are ", length(train_wrongs), " incorrect training predictions.")


    # look at wrong train predictions
    while true  
        println("\nPick a wrong test prediction to display or q to quit:")
        print("   enter a number between 1 and $(length(train_wrongs))> ")
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
            if n > length(train_wrongs) || n < 1
                println("Oops, try again...")
                continue
            else
                GeneralNN.dodigit(n, train_wrongs, results["train_inputs"], results["train_targets"], predmax)
                continue
            end # response case: display a wrong prediction
        end  # response cases
    end  # prompt loop

    println("\nThat's all folks!.....")

end

####################################################################


