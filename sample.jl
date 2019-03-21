# sample training run



println(
    """ 
    Sample run of neural net training with MNIST digits.
    """
)

println("........ Loading code ........")

using Printf
using Revise
using JSON
includet("GeneralNN.jl")
using .GeneralNN


println("........ Training the neural network ........")
train_inputs, train_targets, train_preds, test_inputs, test_targets, test_preds, nnp, bn, hp = train_nn("nninputs.json");



predmax = vec(map(x -> x[1], argmax(test_preds,dims=1)));

# which predictions are wrong?
test_wrongs = GeneralNN.wrong_preds(test_targets, test_preds);
train_wrongs = GeneralNN.wrong_preds(train_targets, train_preds);

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
            GeneralNN.dodigit(n, test_wrongs, test_inputs, test_targets, predmax)
            continue
        end # response case: display a wrong prediction
    end  # response cases
end  # prompt loop

println("\nThat's all folks!.....")


####################################################################


