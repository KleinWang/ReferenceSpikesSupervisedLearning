# MIT License

# Copyright (c) 2024 Zeyuan Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import h5py
import os
from SHD_dataset import create_SHD_dataset

class H5PY(Dataset):
    def __init__(self, dataset_setting, sample_index, validation_Flag=False):

        self.data_path=dataset_setting['data_path'] # './data/hdspikes/
        self.num_example=dataset_setting['num_example']
        self.device=dataset_setting['device']
        self.time_step = dataset_setting['time_step']
        self.duration = dataset_setting['duration'] # the duration in simulation
        self.max_time = dataset_setting['max_time'] # the real time
        self.num_in_neuron = dataset_setting['num_in_neuron']
        self.num_step= self.duration // self.time_step
        train_Flag = dataset_setting['train']
 
        train_file = h5py.File(os.path.join(self.data_path, 'train.h5'), 'r')
        test_file = h5py.File(os.path.join(self.data_path, 'test.h5'), 'r')

        if train_Flag and not validation_Flag:
            self.X = train_file['spikes']
            self.Y = torch.from_numpy(np.array(train_file['labels']).astype(int)).to(self.device)
        else:
            #! the test dataset is the validation dataset for the SHD dataset
            self.X = test_file['spikes']
            self.Y = torch.from_numpy(np.array(test_file['labels']).astype(int)).to(self.device)
            self.num_example=dataset_setting['test_num']
            
        self.firing_times = self.X['times'] 
        self.units_fired = self.X['units']
        
        self.time_bins = np.linspace(0, self.max_time, num=self.num_step)
        
    def __len__(self):
        return self.num_example

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        # compute discrete firing times
        coo = [ [] for i in range(2) ]
        coo[0] = np.digitize(self.firing_times[idx], self.time_bins)
        coo[1] = self.units_fired[idx]
        coo = np.array(coo)

        i = torch.LongTensor(coo).to(self.device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(self.device)
        
        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([self.num_step,self.num_in_neuron])).to(self.device).to_dense()
        y_batch = self.Y[idx]
        
        return X_batch, y_batch

class MNIST_Rate(torchvision.datasets.MNIST):
    '''No random seed is assigned for the spike generation'''
    #*# modify for the validation dataset
    
    def __init__(self, dataset_setting, sample_index, validation_Flag=False):
        
        self.data_path=dataset_setting['data_path'] # './data/hdspikes/
        self.num_example=dataset_setting['num_example']
        self.device=dataset_setting['device']
        self.time_step = dataset_setting['time_step']
        self.duration = dataset_setting['duration'] # the duration in simulation
        self.num_in_neuron = dataset_setting['num_in_neuron']
        self.num_step= self.duration // self.time_step
        self.max_rate = dataset_setting['max_rate']
        self.test_num = dataset_setting['test_num']
        train_Flag = dataset_setting['train']
        if not train_Flag:
            self.num_example = self.test_num
        
        super().__init__(dataset_setting['data_path'], train_Flag, download=True, transform=transforms.ToTensor())
    
    def __len__(self):
        assert self.num_example <= super().__len__()
        return self.num_example
    
    def __getitem__(self, index):
        
        x_data, y_data = super().__getitem__(index%self.num_example)
        
        # convert image to spk trains later
        return self.image_to_spks(x_data.to(self.device)), y_data

    def image_to_spks(self, image):
        '''time step is in unit of ms'''

        probs = image.reshape(-1)*self.max_rate/1000*self.time_step
        
        spk_tensor_list = []
        for step in range(self.num_step):
            spk_tensor_list.append(torch.bernoulli(probs))

        spk_tensor = torch.stack(spk_tensor_list, dim=0)

        return spk_tensor
    
class FashionMNIST_Rate(torchvision.datasets.FashionMNIST):
    '''No random seed is assigned for the spike generation'''
    #*# modify for the validation dataset
    
    def __init__(self, dataset_setting, sample_index, validation_Flag=False):
        
        self.data_path=dataset_setting['data_path'] # './data/hdspikes/
        self.num_example=dataset_setting['num_example']
        self.device=dataset_setting['device']
        self.time_step = dataset_setting['time_step']
        self.duration = dataset_setting['duration'] # the duration in simulation
        self.num_in_neuron = dataset_setting['num_in_neuron']
        self.num_step= self.duration // self.time_step
        self.max_rate = dataset_setting['max_rate']
        self.test_num = dataset_setting['test_num']
        train_Flag = dataset_setting['train']
        if not train_Flag:
            self.num_example = self.test_num
        
        super().__init__(dataset_setting['data_path'], train_Flag, download=True, transform=transforms.ToTensor())
    
    def __len__(self):
        assert self.num_example <= super().__len__()
        return self.num_example
    
    def __getitem__(self, index):
        
        x_data, y_data = super().__getitem__(index%self.num_example)
        
        # convert image to spk trains later
        return self.image_to_spks(x_data.to(self.device)), y_data

    def image_to_spks(self, image):
        '''time step is in unit of ms'''

        probs = image.reshape(-1)*self.max_rate/1000*self.time_step
        
        spk_tensor_list = []
        for step in range(self.num_step):
            spk_tensor_list.append(torch.bernoulli(probs))

        spk_tensor = torch.stack(spk_tensor_list, dim=0)

        return spk_tensor
    
def image_to_sequential_spks(x_data, thresh_list, duration):
    '''the output cue neuron which fires constantly is deleted
    there are two neurons which fires when pixel's gray scale is 1
    the neuron corresponding to pixels crossing thresh=0 decreasingly will be silent always'''
    times = []
    units = []
    for idx_thresh, thresh in enumerate(thresh_list):
        spike_lists=find_onset_offset(x_data.reshape(-1), thresh,duration)
        for idx in range(2):
            times += spike_lists[idx]
            units += [idx_thresh*2+idx for i in range(len(spike_lists[idx]))]
        
    return times, units

def find_onset_offset(y, thresh, duration):
    """
    Given the itorchut signal `y` with samples,
    find the indices where `y` increases and descreases through the value `thresh`.
    Return stacked binary arrays of shape `y` indicating onset and offset thresh crossings.
    `y` must be 1-D numpy arrays.
    """
    if thresh == 1:
        equal = y == thresh
        transition_touch = np.where(equal)[0]

        return [(transition_touch/784*duration).tolist(),(transition_touch/784*duration).tolist()]
    else:
        # Find where y crosses the thresh (increasing).
        lower = y < thresh
        higher = y >= thresh
        transition_onset = np.where(lower[:-1] & higher[1:])[0]
        transition_offset = np.where(higher[:-1] & lower[1:])[0]

        return [(transition_onset/784*duration).tolist(), (transition_offset/784*duration).tolist()]
        
def slice_to_spike_list(slice, time_interval, thresh):
    '''transfer a slice to a spike train '''
    spike_list=[]
    for t, pixel in enumerate(slice):
        if pixel > thresh:
            spike_list.append(t*time_interval+time_interval*(1-(pixel-thresh)/(255-thresh)))

    return spike_list

def image_to_sliced_spks(example, time_interval, thresh):
    
    slice_list=[]
    # append the horizontal slice
    for i in range(0,28):
        slice_list.append(example[i,:])
    # append the vertical slice
    for i in range(0,28):
        slice_list.append(example[:,i]) 
    
    times = []
    units = []
    for idx_neuron, slice in enumerate(slice_list):
        spike_list=slice_to_spike_list(slice,time_interval,thresh)
        times += spike_list
        units += [idx_neuron for i in range(len(spike_list))]
        
    return times, units

def write_h5py(h5py_name, times_examples, units_examples, label_examples):
    # write the dataset in h5py
        
    dt1 = h5py.vlen_dtype(np.dtype('float64'))
    dt2 = h5py.vlen_dtype(np.dtype('int32'))
    # Create an HDF5 file (you can replace 'my_data.h5' with your desired filename)
    data_size = len(times_examples)

    with h5py.File(h5py_name, 'w') as f:
        # Create the 'spikes' group
        spikes_group = f.create_group('spikes')
        # Create the 'times' dataset within the 'spikes' group
        times = spikes_group.create_dataset('times',(data_size,), dtype=dt1)
        # Create the 'units' dataset within the 'spikes' group
        units = spikes_group.create_dataset('units',(data_size,), dtype=dt2)

        # Create the dataset 'labels'
        labels= f.create_dataset('labels', (data_size,))

        times[:data_size] = times_examples
        units[:data_size] = units_examples
        labels[:data_size] = label_examples

def create_sequential_encoded_dataset(dataset_source, num_in_neuron, duration):
    dataset_name=dataset_source+'_Sequential'
    thresh_list = np.linspace(0, 1, num_in_neuron // 2)
    
    if not os.path.exists('./data'):
        os.mkdir('./data')
    
    trainset = getattr(torchvision.datasets, dataset_source)(root='./data', train=True, download=True, transform=None)
    testset = getattr(torchvision.datasets, dataset_source)(root='./data', train=False, download=True, transform=None)
    
    if not os.path.exists('./data/{}'.format(dataset_name)):
        os.mkdir('./data/{}'.format(dataset_name))

    for sub_dataset_str in ['train', 'test']:
        print('generating the {} dataset'.format(sub_dataset_str))
        if sub_dataset_str == 'train':
            sub_dataset=trainset
            data_size=len(sub_dataset)
            h5py_name = './data/{}/'.format(dataset_name)+'train.h5'
        elif sub_dataset_str == 'test':
            sub_dataset=testset
            data_size=len(sub_dataset)
            h5py_name = './data/{}/'.format(dataset_name)+'test.h5'

        label_examples=[]
        times_examples = []
        units_examples = []
        for idx in range(0,data_size):
            example=np.array(trainset[idx][0])
            label_examples.append(trainset[idx][1])
            times_example, units_example = image_to_sequential_spks(example, thresh_list, duration)
            times_examples.append(times_example)
            units_examples.append(units_example)
            if idx%1000 == 0:
                print(idx)
            
        write_h5py(h5py_name, times_examples, units_examples, label_examples)
        
def create_slicing_encoded_dataset(dataset_source, thresh, time_interval):
    dataset_name=dataset_source+'_Sliced'
    
    if not os.path.exists('./data'):
        os.mkdir('./data')
    
    trainset = getattr(torchvision.datasets, dataset_source)(root='./data', train=True, download=True, transform=None)
    testset = getattr(torchvision.datasets, dataset_source)(root='./data', train=False, download=True, transform=None)
    
    if not os.path.exists('./data/{}'.format(dataset_name)):
        os.mkdir('./data/{}'.format(dataset_name))

    for sub_dataset_str in ['train', 'test']:
        print('generating the {} dataset'.format(sub_dataset_str))
        if sub_dataset_str == 'train':
            sub_dataset=trainset
            data_size=len(sub_dataset)
            h5py_name = './data/{}/'.format(dataset_name)+'train.h5'
        elif sub_dataset_str == 'test':
            sub_dataset=testset
            data_size=len(sub_dataset)
            h5py_name = './data/{}/'.format(dataset_name)+'test.h5'

        label_examples=[]
        times_examples = []
        units_examples = []
        for idx in range(0,data_size):
            example=np.array(trainset[idx][0])
            label_examples.append(trainset[idx][1])
            times_example, units_example = image_to_sliced_spks(example, time_interval, thresh)
            times_examples.append(times_example)
            units_examples.append(units_example)
            if idx%1000 == 0:
                print(idx)
            
        write_h5py(h5py_name, times_examples, units_examples, label_examples)

if __name__ == '__main__':
    import sys
    dataset_name = sys.argv[1]
    if __name__ == "__main__":
        if 'Sequential' in dataset_name:
            create_sequential_encoded_dataset(dataset_name[:-11], 56, 100)
        elif 'Sliced' in dataset_name:
            time_interval=100/28
            thresh=50
            create_slicing_encoded_dataset(dataset_name[:-7], thresh, time_interval)
        elif 'SHD' in dataset_name:
            create_SHD_dataset() 
