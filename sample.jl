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
dfname = "digits60000by784.mat"

println("........ Training the neural network ........")
train_inputs, train_targets, train_preds, test_preds, nnp, bn, hp = train_nn(
    dfname, 
    15,  # epochs
    [80];  # hidden units  [300,200,100]
    alpha = .84,
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
# predmax = ind2sub(size(test_preds),vec(findmax(test_preds,1)[2]))[1]
predmax = vec(map(x -> x[1], argmax(test_preds,dims=1)));

_, _, test_inputs, test_targets = GeneralNN.extract_data(dfname);

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
            dodigit(n)
            continue
        end # response case: display a wrong prediction
    end  # response cases
end  # prompt loop

println("\nThat's all folks!.....")

function dodigit(n)
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
