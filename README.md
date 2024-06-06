* To clone this repository

    * `git clone https://github.com/KleinWang/ReferenceSpikesSupervisedLearning.git`
    
* To set up conda env:
    * `cd ./ReferenceSpikes_SupervisedLearning/`
    * `conda env create --file=env.yml` # The package "pyspike" may cause problem. You may need to install it manually after the conda env is created.
    * `conda activate ReferenceSpikes` # Run `conda env list` to check the directory of the created env if your conda failed to find the env.

* Test:
    * `python run_simulation.py --config_path ./config/SHD_test.yml --seed 0 --device gpu` # replace "gpu" with "cpu" if a gpu is not available.
    * Test the code, it may take 3 mins. After the simulation is finished, a file named result_9999.txt will be generated in directory ./result_SHD/ with accuracies printed out.

* To run a simulation:
    * `python run_simulation.py --config_path ./config/[YOUR_CONFIGURE_FILE].yml --seed 0 --device gpu`
    * Use different config files to train networks on different datasetes with different plastic parameters, including reference spikes, trainable thresholds, adaptive thresholds.

* To read the result:
    * A *.txt file is generated in directory ./result_[YOUR_DATASET]/ with accuracies printed out.
    * The accuracies are also saved in a npz file with the same name as the txt file.

* The configure files are in the directory './config'. The name (letters) of the configure files indicate the dataset and the model. The index (4 digits) of congigure file is the result from summing the options below:

    * 2000 are for the onehot_spk_count_CE loss function (rate-encoded output), 3000 are for the onehot_spk_temporal_square loss function (temporally encode output)

    * 000 for baseline model, 100 for trainable thresh, 200 for reference spk, 300 for adaptive_thresh

    * 00 for no delay, 20 for ramdom time delay in range [-12,12]ms 

    * 0-9 for different seeds, # the random seed will be overwriten by the seed provided in the command line.

    * For example, MNIST_Sequential_LIF_RFR_3220 means the result on dataset MNIST_Sequential of a network with the one_hot_temporal_sqaure loss function, reference spikes, ramdom time delay in range [-12,12]ms, and random seed 0.
