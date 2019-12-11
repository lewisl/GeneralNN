    # create model data holding only one example.  Need z, a, and targets only.
    #     this uses an existing example from the dataset
    #     can base a way to do predictions on a small number of samples on this code fragment
    onedat = Batch_view()
    preallocate_minibatch!(onedat::Batch_view, wgts, hp)
    update_batch_views!(onedat, dat, wgts, hp, example:example)