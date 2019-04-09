# sample training run



println(
    """ 
    Sample run of neural net training with MNIST digits.
    """
)

println("........ Loading code ........")

println("\n........ enter runjob() -> code compiles the first time ........")

using Printf
using Revise
using JSON
includet("GeneralNN.jl")
using .GeneralNN


function runjob(jsoninputs="nninputs.json")



    println("........ Training the neural network ........")
    results = GeneralNN.train_nn(jsoninputs);
     # train_inputs, train_targets, train_preds, test_inputs, test_targets, test_preds, nn_params, batchnorm_params, 
     # hyper_params


    predmax = vec(map(x -> x[1], argmax(results["test_preds"],dims=1)));

    # which predictions are wrong?
    test_wrongs = GeneralNN.wrong_preds(results["test_targets"], results["test_preds"]);
    train_wrongs = GeneralNN.wrong_preds(results["train_targets"], results["train_preds"]);

    println("\n\nThere are ", length(test_wrongs), " incorrect test predictions.")
    println("There are ", length(train_wrongs), " incorrect training predictions.")


    # look at wrong test predictions
    while true  
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
                GeneralNN.dodigit(n, test_wrongs, results["test_inputs"], results["test_targets"], predmax)
                continue
            end # response case: display a wrong prediction
        end  # response cases
    end  # prompt loop

    println("\nThat's all folks!.....")

end

####################################################################


