# sample training run

println(
    """ 
    Sample run of neural net training with MNIST digits.
    """
)

println("........ Compiling the program ........")

using Printf
using Revise
include("GeneralNN.jl")
Revise.track("GeneralNN.jl")
using .GeneralNN
dfname = "digits10000by784.mat"

println("........ Training the neural network ........")
train_inputs, train_targets, train_preds, test_preds, nnp, bn, hp = train_nn(
    dfname, 
    20,  # epochs
    [80];  # hidden units  [300,200,100]
    alpha = .8,
    reg = "",  # L1, L2, or ""
    lambda = 0.00096,
    learn_decay = [0.52,3.0],
    mb_size_in = 50, 
    norm_mode = "none",    # or "none" or "standard"
    do_batch_norm=true, 
    opt="adam", 
    units="relu", 
    plots=["Training", "Learning", "Test"]
    );

    # Convert columns of 0,1 predictions to array single value outcomes
    # 1. get the index of the maximum value of each column (e.g, proceed by rows down each column)
    # 2. select the second result, which is the index (rather than the value itself)
    # 3. vec stacks the results into a 1-column array
    # 4. convert the index into the orginal array into subscripts for the case of a 2D or higher order array
predmax = ind2sub(size(test_preds),vec(findmax(test_preds,1)[2]))[1]

_, _, test_inputs, test_targets = GeneralNN.extract_data(dfname);

test_wrongs = wrong_preds(test_targets, test_preds);

train_wrongs = wrong_preds(train_targets, train_preds);

# println("size of predmax: ", size(predmax))
# println("size of test_targets: ", size(test_targets))
# println("size of test_preds:   ", size(test_preds))
# println("size train_preds ", size(train_preds))
# println("train pred 100 ")
# println(train_preds[:,100])
# println("train target 100 ")
# println()

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
            example = test_wrongs[n]
            digit_data = test_inputs[:, example]
            correct = find(test_targets[:,example])[1]  # get the row in the column that contains value 1
            correct = correct == 10 ? 0 : correct
            predicted = predmax[example]
            predicted = predicted == 10 ? 0 : predicted
            println("\n\nThe neural network predicted: $predicted")
            println("The correct value is: $correct")
            display_mnist_digit(digit_data, [28,28])
            continue
        end # response case: display wrong digit
    end  # response cases
end  # prompt loop

println("\nThat's all folks!.....")