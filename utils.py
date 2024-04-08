import numpy as np
import pyspike as spk
import torch

def get_spk_distance(spk_train_1, spk_train_2, duration):

    spk_train_1=spk.SpikeTrain(spk_train_1.cpu().numpy(), np.array([0.0, duration]))
    spk_train_2=spk.SpikeTrain(spk_train_2.cpu().numpy(), np.array([0.0, duration]))

    return spk.spike_distance(spk_train_1, spk_train_2)

def spike_lists_to_tensor(spkie_lists_examples, duration, time_step, device):
    '''convert a spike train for the form of lists to tensor
    size of the spkie_lists_examples: (mini_batch_size, num_afferents, timing)
    
    '''
    num_step=int(duration/time_step)
    data=torch.zeros(len(spkie_lists_examples), num_step, len(spkie_lists_examples[0]), device=device) #size, (mini_batch_size, num_step, afferents)
    for index_example in range(len(spkie_lists_examples)):
        for index_afferent in range(len(spkie_lists_examples[0])):
            for spk in spkie_lists_examples[index_example][index_afferent]:
                spk_timing=int(spk/time_step)
                data[index_example, spk_timing, index_afferent]=1
    return data.type(torch.ByteTensor)

def tensor_to_spike_lists(outputs_gotten, time_step):
    '''convert a spike train for the form of lists to tensor
    each spike_train for one afferent is a tensor

    the tensor is a minibatch of examples
    '''

    mini_batch_size=outputs_gotten.shape[0]
    num_output_afferents=outputs_gotten.shape[2]

    outputs_gotten_list=[]
    for sample_index in range(mini_batch_size):
        outputs_gotten_list.append([])
        for afferent_index in range(num_output_afferents):
            outputs_gotten_list[-1].append(torch.argwhere(outputs_gotten[sample_index,:,afferent_index]).reshape(-1)*time_step)
            
    return outputs_gotten_list
    