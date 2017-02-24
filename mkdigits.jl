# open an MNIST .mat file and save it as a matlab file
# for running mnist digit recognition tests

using MAT

const mnist_test = 1028  
const mnist_cols = 784  # bitmap dimensions of each digit

function mkdigits(matoutname, eachtrain=1000)
    # assumes mnist_all.mat is in current directory
    eachtest = floor(Int, .5 * eachtrain)
    
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
        # training
        sourcekey = "train$i"
        examples = size(df[sourcekey],1)  # this is NOT constant:  different for each digit
        if eachtrain > examples
            warn("Number of examples $eachtrain for each digit exceeds available train data $examples. Rerun with lower input.")
            close(outfile)
            return
        end
        select = rand(1:examples,eachtrain)
        train["x"] = vcat(train["x"], df[sourcekey][select,:])
        # use 10 for "0" hand-written digits when outputing
        if i == 0
            train["y"] = vcat(train["y"],
                repmat(reshape(eye(10)[10,:],1,10),eachtrain,1))
        else
            train["y"] = vcat(train["y"],
                repmat(reshape(eye(10)[i,:],1,10),eachtrain,1))            
        end

        # test
        sourcekey = "test$i"
        examples = size(df[sourcekey],1)  # this is NOT constant:  different for each digit
        if eachtest > examples
            warn("Number of examples $eachtest for each digit exceeds available test data $examples. Rerun with lower input.")
            close(outfile)
            return
        end
        select = rand(1:examples,eachtest)
        test["x"] = vcat(test["x"], df[sourcekey][select,:])
        if i == 0
            test["y"] = vcat(test["y"],
                repmat(reshape(eye(10)[10,:],1,10),eachtest,1))
        else
            test["y"] = vcat(test["y"],
                repmat(reshape(eye(10)[i,:],1,10),eachtest,1))        
        end
    end

    #remove first row of each array
    train["x"] = train["x"][2:end,:]
    train["y"] = train["y"][2:end,:]
    test["x"] = test["x"][2:end,:]
    test["y"] = test["y"][2:end,:]
    
    write(outfile, "train", train)
    write(outfile, "test", test)
    close(outfile) 

end

