# TODO
#   add ability to use all of the data
#   add a seed for rand to enable a consistent choice or use system entropy


# open an MNIST .mat file and save it as a matlab file
# for running mnist digit recognition tests

using MAT

const mnist_test = 1028  
const mnist_cols = 784  # bitmap dimensions of each digit

function mkdigits(matoutname, training_samples=1000; makeall=false)
    # assumes mnist_all.mat is in current directory
    test_samples = floor(Int, .5 * training_samples)
    
    # check if output file exists and ask permission to overwrite
    if isfile(matoutname)
        print("Output file $matoutname exists. OK to overwrite? ")
        resp = readline()
        if contains(lowercase(resp), "y")
            rm(matoutname)
        else
            error("File exists. You said not to overwrite.")
        end
    end

    #set number of training and test each to put in output file

    # hold the outputs while building them
    train = Dict([("x",zeros(1, mnist_cols)), 
        ("y", zeros(1,10))])
    test = Dict([("x",zeros(1, mnist_cols)), 
        ("y", zeros(1,10))])

    #read the mnist datafile, open the output file
    df = matread("mnist_all.mat")
    outfile = matopen(matoutname, "w")

    
    for i = 0:9
        # training samples
        sourcekey = "train$i"
        examples = size(df[sourcekey],1)  # this is NOT constant:  different for each digit
        # if training_samples > examples
        #     warn("Number of training examples $training_samples for each digit\n exceeds available train data $examples. Rerun with lower input.")
        #     close(outfile)
        #     return
        # end
        if makeall  # use all samples for each digit
            select = collect(1:examples)
            training_samples = examples
        else  
            if training_samples > examples
                warn("Number of training examples $training_samples for each digit\n exceeds available train data $examples. Rerun with lower input.")
                close(outfile)
                return
            end
            select = rand(1:examples,training_samples)  # random subset of cnt training_samples 
        end
        train["x"] = vcat(train["x"], df[sourcekey][select,:])  # append rows for the next digit
    
        if i == 0  # use 10 for "0" hand-written digits when outputting
            train["y"] = vcat(train["y"],
                repmat(reshape(eye(10)[10,:],1,10),training_samples,1))
        else
            train["y"] = vcat(train["y"],
                repmat(reshape(eye(10)[i,:],1,10),training_samples,1))            
        end

        # test samples
        sourcekey = "test$i"
        examples = size(df[sourcekey],1)  # this is NOT constant:  different for each digit

        if makeall
            select = collect(1:examples)
            test_samples = examples
        else   
            if test_samples > examples
                warn("Number of examples $test_samples for each digit exceeds available test data $examples. Rerun with lower input.")
                close(outfile)
                return
            end
            select = rand(1:examples,test_samples)
        end
        test["x"] = vcat(test["x"], df[sourcekey][select,:])
        if i == 0
            test["y"] = vcat(test["y"],
                repmat(reshape(eye(10)[10,:],1,10),test_samples,1))
        else
            test["y"] = vcat(test["y"],
                repmat(reshape(eye(10)[i,:],1,10),test_samples,1))        
        end
    end

    #remove first row of each array  -- I kinda forget why
    train["x"] = train["x"][2:end,:]
    train["y"] = train["y"][2:end,:]
    test["x"] = test["x"][2:end,:]
    test["y"] = test["y"][2:end,:]
    
    write(outfile, "train", train)
    write(outfile, "test", test)
    close(outfile) 

end

