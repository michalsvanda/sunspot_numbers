import datetime as dt

config = {
    "W": dt.timedelta(days=5000), # width of the evaluation window
    "dW": dt.timedelta(days=100),  # width of the prediction window
    # dealing with low g and low f values that are not to be considered in the calculation
    "minspots": 5, # when doing the k table, how many spots there have to be at minimum?
    "mingroups": 1, # when doing the k table, how many groups there have to be at minimum?
    # reduction to the optimal conditions [R, f, g]
    "Qreduction_slopes": [-0.09048021, -0.07796618, -0.03477956], # fit with errors in Q +/- 0.5
    # turning on/off reductions
    "Qreduction_eval_switch": 1, # reduce during the evaluation period, default=1
    "Qreduction_predict_switch": 0, # reduce during the prediction period, default=0
    "Niterations": 2, # how many passes; default=2
    "database_file": 'databaze.hdf', # file with database
    "do_plot_results": True, # do we plot the results somehow
}
