* To clone this repository

    * `git clone https://oauth2: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@github.com/KleinWang/ReferenceSpikes_SupervisedLearning.git`
    

* To set up conda env:
    * `cd ./ReferenceSpikes_SupervisedLearning/`
    * `conda env create --file=env.yml`
    * `conda activate ReferenceSpikes`

* Test:
    * `python run_simulation.py --config_path ./config/SHD_test.yml --seed 0 --device gpu`
    * test the code, it may take 3 mins

* To run a simulation:
    * `python run_simulation.py --config_path ./config/MNIST_Sequential_LIF_RFR_3220.yml --seed 0 --device gpu`
    * use different config files to train networks trained on different datasetes with different plastic parameters, including reference spikes, trainable thresholds, adaptive thresholds.

* To read the result:
    * A \*.txt file is generated in directory ./result_*/ with accuracies printed out.

* The configure files are in the directory './config'. The name of the configure files indicate the dataset and the model. The index of congigure file is summed by the options below:

    * 2000 are for the onehot_spk_count_CE loss function (one-hot rate coding for output), 3000 are for the onehot_spk_temporal_square loss function (one-hot temporal coding for output)

    * 000 for baseline model, 100 for trainable thresh, 200 for reference spk, 300 for adaptive_thresh

    * 00 for no delay, 20 for ramdom time delay in range [-12,12]ms 

    * 0-9 for different seeds

    * For example, MNIST_Sequential_LIF_RFR_3220 means the result on dataset MNIST_Sequential of a network with the one_hot_temporal_sqaure loss function, reference spikes, ramdom time delay in range [-12,12]ms, and random seed 0.
