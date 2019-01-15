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
train_inputs, train_targets, train_preds, test_inputs, test_targets, test_preds, nnp, bn, hp = train_nn("nninputs_1_13_2019.json");



    # Convert columns of 0,1 predictions to array single value outcomes
    # 1. get the index of the maximum value of each column (e.g, proceed by rows down each column)
    # 2. select the second result, which is the index (rather than the value itself)
    # 3. vec stacks the results into a 1-column array
    # 4. convert the index into the orginal array into subscripts for the case of a 2D or higher order array
# predmax = ind2sub(size(test_preds),vec(findmax(test_preds,1)[2]))[1]
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


