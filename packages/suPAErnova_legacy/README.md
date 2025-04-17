# Workflow:

[https://github.com/georgestein/suPAErnova/blob/main/notebooks/salt_model_make_dataset.ipynb]
    This requires a few files from '/global/homes/g/gstein/src/snfdata/' and "mask_info_wmin_wmax.txt". mask_info_wmin_wmax.txt I just forwarded in an email from Greg
    It then compiles individual SN text files into the 3D arrays we discussed. 
    It also creates salt spectra for each real spectra.
    then saves `snf_data_wSALT.npy', which is what is used for train.test splitting and model training
[https://github.com/georgestein/suPAErnova/blob/main/suPAErnova/make_datasets/make_train_test_data.py]
    Splits into train test sets and does additional processing (masks laser lines)
[https://github.com/georgestein/suPAErnova/blob/main/scripts/train_ae.py 
    Uses the above to train the AE based on params in the config
[https://github.com/georgestein/suPAErnova/blob/main/scripts/train_flow.py]
    Trains the flow using params in the config
[https://github.com/georgestein/suPAErnova/blob/main/scripts/run_posterior_analysis.py]
    Runs posterior analysis (uses the PAE model to fit the observations & gets error bars) 
[https://github.com/georgestein/suPAErnova/blob/main/scripts/slurm/submit_train.slr]
    This is the slurm submission script that does the above 3 steps in 1 go
[https://github.com/georgestein/suPAErnova/blob/main/notebooks/plots_and_analysis.ipynb]
    Makes plots and analyses the outputs of step #5
