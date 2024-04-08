To set up conda env:
    conda env create --name envname --file=env.yml

To clone this repository
    

To run a simulation:
    python run_simulation.py --config_path ./config/MNIST_Sliced_LIF_RFR_3220.yml --seed 0 --device gpu

The index of congigure file is summed by the options below. 
    2000 are for the onehot_spk_count_CE loss function 
    3000 are for the onehot_spk_temporal_square loss function 

    000 for baseline model, 100 for trainable thresh, 200 for reference spk, 300 for adaptive_thresh

    00 for no delay
    20 for ramdom time delay in range [-12,12]ms 

    0-9 for different seeds

    For example, 3220 means the result of a network with the one_hot_temporal_sqaure loss function, reference spikes, ramdom time delay in range [-12,12]ms, and random seed 0


