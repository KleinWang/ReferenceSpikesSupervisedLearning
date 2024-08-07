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
from torch import nn
torch.set_num_threads(1) #important for multiprocessing
import numpy as np
dtype=torch.float32
from utils import get_spk_distance
from torch.optim.lr_scheduler import OneCycleLR

# surrogate gradients
class SurrGradSpike(torch.autograd.Function):
    
    beta = 1
    gamma = 100.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input) # important here
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors # access the saved tensor
        grad_input = grad_output.clone()
        grad = SurrGradSpike.beta * grad_input / (SurrGradSpike.gamma*torch.abs(input)+1.0)**2
        return grad
    
class SurrGradSpike_ATan(torch.autograd.Function):

    alpha = 5

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input) # important here
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors # access the saved tensor
        grad_input = grad_output.clone()
        # grad = SurrGradSpike.beta * grad_input / (SurrGradSpike.gamma*torch.abs(input)+1.0)**2
        grad = SurrGradSpike_ATan.alpha * grad_input / ((SurrGradSpike_ATan.alpha * input * torch.pi/2)**2+1.0) / 2
        return grad

# loss funcs
def onehot_spk_count_MSE(net, outputs_gotten):
    '''the spk count in the last layer is the scalar output'''

    outputs_readout=outputs_gotten # the 'spk{}_rec_tensor' from the last layer, size: [batch_size,time_step,num_neuron]
    output_sum=torch.sum(outputs_readout,1) # size: [batch_size,num_neuron]
    batch_size = output_sum.shape[0]
    num_out_neuron = output_sum.shape[-1]

    # define the loss function
    loss_fn=torch.nn.MSELoss(reduction='mean')

    # calculate loss function
    y_input=torch.zeros(batch_size,num_out_neuron).to(net.device)
    y_input[torch.arange(batch_size),net.y_data]=10 # net.y_data size: [batch_size] #! make 10 adjustable

    loss_value = loss_fn(output_sum, y_input.detach())
    net.label_predict = torch.argmax(output_sum, dim=-1).detach() # net.label_predict size: [batch_size]
    acc_value=torch.sum(net.label_predict==net.y_data)/torch.numel(net.label_predict)
    # net.acc_hist.append(net.acc_batch.detach().cpu().numpy())

    return loss_value, acc_value

def onehot_spk_count_CE(net, outputs_gotten):
    '''the spk count in the last layer is the scalar output'''

    outputs_readout=outputs_gotten # the 'spk{}_rec_tensor' from the last layer, size: [batch_size,time_step,num_neuron]
    output_sum=torch.sum(outputs_readout,1) # size: [batch_size,num_neuron]

    # define the loss function
    loss_fn=torch.nn.functional.cross_entropy

    # calculate loss function
    loss_value = loss_fn(output_sum, net.y_data.detach())
    net.label_predict = torch.argmax(output_sum, dim=-1).detach() # net.label_predict size: [batch_size]
    acc_value=torch.sum(net.label_predict==net.y_data)/torch.numel(net.label_predict)
    # net.acc_hist.append(net.acc_batch.detach().cpu().numpy())

    return loss_value, acc_value

def onehot_spk_temporal_square(net, outputs_gotten):

    outputs_readout=outputs_gotten # the 'spk{}_rec_tensor' from the last layer, size:batch_size,time_step, afferent_size
    batch_size = outputs_readout.shape[0]
    num_out_neuron = outputs_readout.shape[-1]

    output_mid=torch.sum((outputs_readout-0)**2,1)
    for example_index in range(batch_size):
        output_here=outputs_readout[example_index,:,net.y_data[example_index]]
        output_mid[example_index,net.y_data[example_index]]=torch.sum((output_here-net.target_tensor.to(net.device))**2)  #  size: [batch_size,num_neuron], target tensor is for one spk train only,

    # define the loss function
    loss_fn=torch.nn.MSELoss(reduction='mean')

    # calculate loss function
    y_input=torch.zeros(batch_size,num_out_neuron).to(net.device)
    loss_value = loss_fn(output_mid, y_input.detach())

    labels=[]
    if net.onehot_spk_temporal_acc_method=='temporal':
        for example_index in range(batch_size):
            dist_list=[]
            for i in range(num_out_neuron):
                spk_train1=torch.argwhere(net.target_tensor>0.5).reshape(-1)
                spk_train2=torch.argwhere(outputs_gotten[example_index,:,i]>0.5).reshape(-1)
                dist_list.append(get_spk_distance(spk_train1, spk_train2, net.duration))
            labels.append(np.argmin(dist_list))
    elif net.onehot_spk_temporal_acc_method=='rate':
        output_sum=torch.sum(outputs_readout,1)
        labels = torch.argmax(output_sum, dim=1).detach()
    
    net.label_predict = torch.tensor(labels).to(net.device)
    acc_value=torch.sum(net.label_predict==net.y_data)/torch.numel(net.label_predict)
    # net.acc_hist.append(net.acc_batch.detach().cpu().numpy())

    return loss_value, acc_value

# network class
class net_LIF:

    name = 'net_LIF'

    def __init__(self, sample_index, kwargs):

        # load sampling_size and index
        self.sampling_size=kwargs['sampling_size'] # the sampling size of the whole simulation with possible multiple networks running in a parallel pool
        self.sample_index=sample_index
        
        self.device=kwargs.get('device')
        self.layer_struct=kwargs.get('layer_struct')
        self.time_step=kwargs.get('time_step')

        # load parameters for the Neuron
        self.tau_mem=kwargs.get('tau_mem') # add control of constants of neuron
        self.tau_syn=kwargs.get('tau_syn') # add control of constants of neuron
        self.alpha=float(np.exp(-self.time_step/self.tau_syn))
        self.beta=float(np.exp(-self.time_step/self.tau_mem))

        # the prameters for surrogate gradient of the spike generation function
        self.sg_gamma=kwargs.get('sg_gamma')
        self.sg_beta=kwargs.get('sg_beta')
        
        if kwargs['SG'] == 'SurrGradSpike':
            SurrGradSpike.gamma=self.sg_gamma
            SurrGradSpike.beta=self.sg_beta
            self.spike_fn= SurrGradSpike.apply
        elif kwargs['SG'] == 'SurrGradSpike_ATan':
            SurrGradSpike_ATan.alpha=kwargs.get('sg_alpha')
            self.spike_fn= SurrGradSpike_ATan.apply

    # network 
    def run_snn(self):
        '''run a SNN with a feed-forward structure defined by self.layer_struct
        syn{} the synaptic current of a layer, (batch_size, num_neuron)
        mem{} the membrane potential of a layer
        spk{} the spike output of a layer
        spk{}_rec_list, record the spk at different time
        syn_current, the synaptic current of a layer, (batch_size, time step, num_neuron)
        spk{}_rec_tensor, the spk output of a layer, (batch_size, time step, num_neuron)
        '''
        self.dropout_mask_list = []
        for i in range(len(self.layer_struct)-1):
            # if i==1:
            #     prob = torch.ones(1,1,self.layer_struct[i]).to(self.device)
            #     self.dropout_mask_list.append(prob)
            # else:
            prob = torch.zeros(1,1,self.layer_struct[i]).to(self.device)+1-self.dropout_rate
            dropout_mask = torch.bernoulli(prob)
            self.dropout_mask_list.append(dropout_mask/(1-self.dropout_rate))

        self.spk0_rec_tensor=self.x_data # size: [batch_size, time_step, num_neuron]
        batch_size = self.x_data.shape[0]

        for i in range(1, len(self.layer_struct)):
            
            # initial value of syn and mem
            setattr(self, 'syn{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) )  # the synaptic input of per layer
            setattr(self, 'mem{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) ) # the membrane potential of per layer

            # create list to save the spk
            setattr(self, 'spk{}_rec_list'.format(i), [])

            # calculate the synaptic current input each layer from last layer
            syn_current = torch.einsum("btx,xy->bty", (getattr(self, 'spk{}_rec_tensor'.format(i-1))*self.dropout_mask_list[i-1], getattr(self,'w{}'.format(i))))
            
            for t in range(self.num_step):

                setattr(self, 'spk{}'.format(i), self.spike_fn(getattr(self,'mem{}'.format(i))-1.0))
                getattr(self, 'spk{}_rec_list'.format(i)).append(getattr(self, 'spk{}'.format(i)))
                setattr(self, 'mem{}'.format(i), self.beta*(getattr(self, 'mem{}'.format(i)) * (1.0-getattr(self, 'spk{}'.format(i)).detach())+getattr(self, 'syn{}'.format(i))*self.time_step))
                setattr(self, 'syn{}'.format(i), self.alpha*(getattr(self, 'syn{}'.format(i)) + syn_current[:,t,:]))

            # make the spk output of current layer a tensor
            setattr(self, 'spk{}_rec_tensor'.format(i), torch.stack(getattr(self, 'spk{}_rec_list'.format(i)),dim=1)) # size: [batch_size, time_step, num_neuron]

        return getattr(self, 'spk{}_rec_tensor'.format(i))

    # training
    def init_train(self, kwargs):

        # init training method
        self.train_method=kwargs.get('train_method')
        self.loss_func = globals()[self.train_method]
        
        if 'temporal' in self.train_method:
            # create a target spike train for the neuron corresponding to the correct label
            target_spks=kwargs['target_spks']
            target_tensor=torch.zeros(int(self.duration/self.time_step))
            for i in target_spks:
                target_tensor[int(i/self.time_step)]=1
            self.target_tensor=target_tensor
            self.onehot_spk_temporal_acc_method = kwargs.get('onehot_spk_temporal_acc_method')
        
        self.regular_scale=kwargs.get('regular_scale')

        # create empty list to record spk_count
        for i in range(1, len(self.layer_struct)):
                setattr(self, 'spk{}_count_iterations'.format(i), [])
                
        self.opt_lr=kwargs.get('opt_lr')
        self.weight_decay=kwargs.get('weight_decay')
        self.dropout_rate = kwargs['dropout_rate']
        self.dropout_rate_ = kwargs['dropout_rate']
 
    def init_plastic_paras(self, kwargs):
        weight_scale_list=kwargs.get('weight_scale_list') # the scales to control the intialization of the random weights

        # create empty weights matrix for all smaples
        for i in range(1, len(self.layer_struct)):
                setattr(self, 'w{}_temp'.format(i), torch.empty((self.sampling_size, self.layer_struct[i-1], self.layer_struct[i]), device=self.device))

        # init weight for all smaples
        for sample_index in range(self.sampling_size):
            for i in range(1, len(self.layer_struct)):
                if weight_scale_list == 'kaiming':
                    torch.nn.init.kaiming_uniform_(getattr(self,'w{}_temp'.format(i))[sample_index,:,:], nonlinearity='relu')
                else:
                    torch.nn.init.normal_(getattr(self,'w{}_temp'.format(i))[sample_index,:,:], mean=0.0, std=weight_scale_list[i-1])

        # keep the weight for a specifc sample_index
        for i in range(1, len(self.layer_struct)):
            setattr(self, 'w{}'.format(i), getattr(self,'w{}_temp'.format(i))[self.sample_index,:,:])
            getattr(self,'w{}'.format(i)).requires_grad=True

        # init weight clipper
        self.CLIP=kwargs.get('CLIP', False)
        if self.CLIP:
            self.w_range=kwargs.get('w_range')

    def init_optimizer(self, kwargs):
        self.params=[] # the two dicts have different learning rate

        for i in range(1, len(self.layer_struct)):
            self.params.append(getattr(self,'w{}'.format(i)))
            
        self.optimizer = torch.optim.Adam(self.params, lr=kwargs.get('opt_lr'), betas=(0.9,0.999))
        
        if 'scheduler' in kwargs.keys():
            self.scheduler = globals()[kwargs['scheduler']](self.optimizer, **kwargs['scheduler_kwargs'])

    def run_forward(self):
        '''Set train_methond to be none, if use customized grad of loss function.add()
        TO use a train_method, remember to set a desired output
        '''

        # init grad
        for param in self.params:
            if type(param)==dict:
                for i in param['params']:
                    i.grad=None
            else:
                param.grad = None
        
        # clip weight
        if self.CLIP:
            self.clipping()

        # run the simulation
        outputs_gotten = self.run_snn()

        # calculate loss and difference
        self.loss_value, self.acc_value =self.loss_func(self, outputs_gotten.to(self.device))
        # self.loss_hist.append(self.loss_value.detach().cpu().numpy())

        # calculate regularizer
        reg_loss_list = []
        for i in range(1, len(self.layer_struct)-1): # exclude the output layer
            reg_loss=torch.sum(torch.sum(getattr(self, 'spk{}_rec_tensor'.format(i)), dim=1)**2)
            reg_loss_list.append(reg_loss)
        
        self.reg_loss=sum(reg_loss_list)

        # record spk count
        for i in range(1, len(self.layer_struct)):
            getattr(self, 'spk{}_count_iterations'.format(i)).append(torch.mean(torch.sum(getattr(self, 'spk{}_rec_tensor'.format(i)), dim=1)).detach().item())

        return outputs_gotten 
    
    def train_backward(self):

        # updates the parameters
        loss=(self.loss_value+self.regular_scale*self.reg_loss)
        
        if self.device == torch.device("cpu"):
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        else:

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        if 'scheduler' in self.__dict__:
            self.scheduler.step()

        with torch.no_grad():
            for i in range(1, len(self.layer_struct)):
                getattr(self,'w{}'.format(i)).mul_(1 - self.opt_lr * self.weight_decay)

    def clipping(self):
        '''clip the weight, make sure it's still biological'''
        with torch.no_grad():
            for i in range(1, len(self.layer_struct)):
                getattr(self,'w{}'.format(i)).clamp_(*self.w_range)

    # load learned parameters
    def load(self, path_data, sample_index):
        data=np.load(path_data)

        # load the weight
        with torch.no_grad():
            for i in range(1, len(self.layer_struct)):
                setattr(self, 'w{}'.format(i), torch.from_numpy(data[ 'w{}'.format(i)][sample_index]).to(self.device))
                getattr(self, 'w{}'.format(i)).requires_grad=True

    def train(self,):
        self.dropout_rate = self.dropout_rate_
    
    def eval(self,):
        self.dropout_rate = 0
    
class net_rfr_spk(net_LIF):
        
    name = 'net_rfr_spk'
        
    def __init__(self, sample_index, kwargs):
        super().__init__(sample_index, kwargs)
        self.tau_syn_rfr=kwargs['tau_syn_rfr']
        self.rfr_spk_method=kwargs['rfr_spk_method']
        self.alpha_rfr=float(np.exp(-self.time_step/self.tau_syn_rfr))
        self.num_rfr_neuron_layers=kwargs.get('num_rfr_neuron_layers') 

    def run_snn(self):
        '''run a SNN with a feed-forward structure defined by self.layer_struct
        syn{} the synaptic current of a layer, (batch_size, num_neuron)
        mem{} the membrane potential of a layer
        spk{} the spike output of a layer
        spk{}_rec_list, record the spk at different time
        syn_current, the synaptic current of a layer, (batch_size, time step, num_neuron)
        spk{}_rec_tensor, the spk output of a layer, (batch_size, time step, num_neuron)
        '''
        self.dropout_mask_list = []
        for i in range(len(self.layer_struct)-1):
            # if i==1:
            #     prob = torch.ones(1,1,self.layer_struct[i]).to(self.device)
            #     self.dropout_mask_list.append(prob)
            # else:
            prob = torch.zeros(1,1,self.layer_struct[i]).to(self.device)+1-self.dropout_rate
            dropout_mask = torch.bernoulli(prob)
            self.dropout_mask_list.append(dropout_mask/(1-self.dropout_rate))

        self.spk0_rec_tensor=self.x_data.to(self.device) # size: [batch_size, time_step, num_neuron]
        batch_size = self.x_data.shape[0]
        
        # generate rfr spks
        if np.sum(self.num_rfr_neuron_layers)>0:
            self.get_rfr_spk_from_potent()

        for i in range(1, len(self.layer_struct)):
            
            # initial value of syn and mem
            setattr(self, 'syn{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) )  # the synaptic input per layer
            setattr(self, 'mem{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) ) # the membrane potential per layer

            # create list to save the spk
            setattr(self, 'spk{}_rec_list'.format(i), [])

            # calculate the synaptic current input each layer from last layer
            # print(getattr(self, 'spk{}_rec_tensor'.format(i-1)).device)
            # print(getattr(self,'w{}'.format(i)).device)
            syn_current = torch.einsum("btx,xy->bty", (getattr(self, 'spk{}_rec_tensor'.format(i-1))*self.dropout_mask_list[i-1], getattr(self,'w{}'.format(i))))

            # calculate the synaptic current input each layer from corresponding reference neurons
            if self.num_rfr_neuron_layers[i-1]>0:
                setattr(self, 'syn_rfr{}'.format(i), torch.zeros((1, self.layer_struct[i]), device=self.device , requires_grad=True) )  # the batch size is one, the synaptic input from reference neurons per layer
                syn_current_rfr=torch.einsum("btx,xy->bty", (getattr(self, 'spk_rfr{}_rec_tensor'.format(i-1)), (getattr(self,'w_rfr{}'.format(i)))))
            
            for t in range(self.num_step):

                setattr(self, 'spk{}'.format(i), self.spike_fn(getattr(self,'mem{}'.format(i))-1.0))
                getattr(self, 'spk{}_rec_list'.format(i)).append(getattr(self, 'spk{}'.format(i)))
                if self.num_rfr_neuron_layers[i-1]==0:
                    setattr(self, 'mem{}'.format(i), self.beta*(getattr(self, 'mem{}'.format(i)) * (1.0-getattr(self, 'spk{}'.format(i)).detach())+getattr(self, 'syn{}'.format(i))*self.time_step))
                    setattr(self, 'syn{}'.format(i), self.alpha*(getattr(self, 'syn{}'.format(i)) + syn_current[:,t,:]))
                else:
                    setattr(self, 'mem{}'.format(i), self.beta*(getattr(self, 'mem{}'.format(i)) * (1.0-getattr(self, 'spk{}'.format(i)).detach()) +getattr(self, 'syn{}'.format(i))*self.time_step+ getattr(self, 'syn_rfr{}'.format(i))*self.time_step))
                    setattr(self, 'syn{}'.format(i), self.alpha*(getattr(self, 'syn{}'.format(i)) + syn_current[:,t,:]))
                    setattr(self, 'syn_rfr{}'.format(i), self.alpha_rfr*(getattr(self, 'syn_rfr{}'.format(i)) + syn_current_rfr[:,t,:]))
                    
            # make the spk output of current layer a tensor
            setattr(self, 'spk{}_rec_tensor'.format(i), torch.stack(getattr(self, 'spk{}_rec_list'.format(i)),dim=1)) # size: [batch_size, time_step, num_neuron]

        return getattr(self, 'spk{}_rec_tensor'.format(i))

    def init_train(self, kwargs):
        super().init_train(kwargs)

        self.rfr_decay=kwargs.get('rfr_decay')
        self.rfr_regular_scale=kwargs.get('rfr_regular_scale')
    
        for layer_index, num_rfr_neuron in enumerate(self.num_rfr_neuron_layers):
            if num_rfr_neuron>0:
                setattr(self, 'rfr_spk{}_count_iterations'.format(layer_index), [])
    
    def init_plastic_paras(self, kwargs):

        super().init_plastic_paras(kwargs)
        
        # load hyper parameters about reference membrane potential
        rfr_scale_layers=kwargs.get('rfr_scale_layers') # the std to init the rfr potential for each layer
        w_rfr_scale_layers=kwargs.get('w_rfr_scale_layers') # the std to init the w_rfr 
        
        # create empty matrix of rfr current for each layer for all smaples
        for layer_index, num_rfr_neuron in enumerate(self.num_rfr_neuron_layers):
            if num_rfr_neuron>0:
                setattr(self, 'rfr{}_temp'.format(layer_index), torch.empty((self.sampling_size, 1, self.num_step, num_rfr_neuron), device=self.device)) # batch_size = 1

        # initialize rfr for all smaples
        for sample_index in range(self.sampling_size):
            for layer_index, num_rfr_neuron in enumerate(self.num_rfr_neuron_layers):
                if num_rfr_neuron>0:
                    if kwargs['rfr_init'] == 'uniform':
                        torch.nn.init.uniform_(getattr(self,'rfr{}_temp'.format(layer_index))[sample_index], a=0, b=rfr_scale_layers[layer_index])
                    elif kwargs['rfr_init'] == 'normal':
                        torch.nn.init.normal_(getattr(self,'rfr{}_temp'.format(layer_index))[sample_index], mean=0.0, std=rfr_scale_layers[layer_index])

        # set the clipping for rfr
        self.CLIP=kwargs.get('CLIP', False)
        if self.CLIP:
            self.rfr_range=kwargs.get('rfr_range')

        # create empty w_rfr for the connection of a layer and its rfr neuron for all smaples
        for layer_index, num_rfr_neuron in enumerate(self.num_rfr_neuron_layers):
            if num_rfr_neuron>0:
                setattr(self, 'w_rfr{}_temp'.format(layer_index+1), torch.empty((self.sampling_size, num_rfr_neuron, self.layer_struct[layer_index+1]), device=self.device))

        # init w_rfr for all smaples
        for sample_index in range(self.sampling_size):
            for layer_index, num_rfr_neuron in enumerate(self.num_rfr_neuron_layers):
                if num_rfr_neuron>0:
                    if w_rfr_scale_layers == 'kaiming':
                        torch.nn.init.kaiming_uniform_(getattr(self,'w_rfr{}_temp'.format(layer_index+1))[sample_index], nonlinearity='relu')
                    else:
                        torch.nn.init.normal_(getattr(self,'w_rfr{}_temp'.format(layer_index+1))[sample_index], mean=0.0, std=w_rfr_scale_layers[layer_index])
        
        # keep rfr and w_rfr for a specific sample_index
        for layer_index, num_rfr_neuron in enumerate(self.num_rfr_neuron_layers):
            if num_rfr_neuron>0:
                setattr(self, 'w_rfr{}'.format(layer_index+1), getattr(self,'w_rfr{}_temp'.format(layer_index+1))[self.sample_index])
                getattr(self,'w_rfr{}'.format(layer_index+1)).requires_grad=True
                setattr(self, 'rfr{}'.format(layer_index), getattr(self,'rfr{}_temp'.format(layer_index))[self.sample_index])
                getattr(self,'rfr{}'.format(layer_index)).requires_grad=True

    def init_optimizer(self, kwargs):

        self.params=[{},{}] # the two dicts have different learning rate
        self.params[0]['params']=[]
        self.params[0]['lr']=kwargs.get('opt_lr')
        self.params[1]['params']=[]
        self.params[1]['lr']=kwargs.get('rfr_lr')
        
        for i in range(1, len(self.layer_struct)):
            self.params[0]['params'].append(getattr(self,'w{}'.format(i)))
            
        for layer_index, num_rfr_neuron in enumerate(self.num_rfr_neuron_layers):
            if num_rfr_neuron>0:
                self.params[0]['params'].append(getattr(self,'w_rfr{}'.format(layer_index+1)))
                self.params[1]['params'].append(getattr(self,'rfr{}'.format(layer_index)))

        self.optimizer = torch.optim.Adam(self.params, lr=kwargs.get('opt_lr'), betas=(0.9,0.999))
        if 'scheduler' in kwargs.keys():
            self.scheduler = globals()[kwargs['scheduler']](self.optimizer, **kwargs['scheduler_kwargs'])

    def get_rfr_spk_from_potent(self):
        '''generate rfr spk from rfr potent of the reference layer, this process can be indepedent from the main run_snn'''

        for i, num_rfr_neuron in enumerate(self.num_rfr_neuron_layers):
            if num_rfr_neuron>0:

                setattr(self, 'spk_rfr{}_rec_list'.format(i), [])
        
                for t in range(self.num_step):

                    setattr(self, 'spk_rfr{}'.format(i), self.spike_fn(getattr(self, 'rfr{}'.format(i))[:,t,:]-1.0))
                    getattr(self, 'spk_rfr{}_rec_list'.format(i)).append(getattr(self, 'spk_rfr{}'.format(i)))

                setattr(self, 'spk_rfr{}_rec_tensor'.format(i), torch.stack(getattr(self, 'spk_rfr{}_rec_list'.format(i)),dim=1))

    def clipping(self):
        super().clipping()
        with torch.no_grad():
            for i, num_rfr_neuron in enumerate(self.num_rfr_neuron_layers):
                if num_rfr_neuron>0:
                    getattr(self,'w_rfr{}'.format(i+1)).clamp_(*self.w_range)
                    getattr(self,'rfr{}'.format(i)).clamp_(*self.rfr_range)

    def run_forward(self):
        '''Set train_methond to be none, if use customized grad of loss function.add()
        TO use a train_method, remember to set a desired output
        '''

        outputs_gotten = super().run_forward()

        # calculate regularizer for reference spikes
        reg_loss_rfr_list = []

        for i, num_rfr_neuron in enumerate(self.num_rfr_neuron_layers):
            if num_rfr_neuron>0:
                reg_loss_rfr = torch.sum(torch.sum(getattr(self, 'spk_rfr{}_rec_tensor'.format(i)), dim=1)**2)
                reg_loss_rfr_list.append(reg_loss_rfr)
        self.reg_loss_rfr=sum(reg_loss_rfr_list)

        for i, num_rfr_neuron in enumerate(self.num_rfr_neuron_layers):
            if num_rfr_neuron>0:
                getattr(self, 'rfr_spk{}_count_iterations'.format(i)).append(torch.mean(torch.sum(getattr(self, 'spk_rfr{}_rec_tensor'.format(i)), dim=1)).detach().item())
        
        return outputs_gotten 
    
    def train_backward(self):

        # updates the parameters
        loss=(self.loss_value+self.regular_scale*self.reg_loss)

        if np.sum(self.num_rfr_neuron_layers)>0:
            loss=(loss+self.rfr_regular_scale*self.reg_loss_rfr)
        
        if self.device == torch.device("cpu"):
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        else:

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        if 'scheduler' in self.__dict__:
            self.scheduler.step()

        with torch.no_grad():
            for i in range(1, len(self.layer_struct)):
                getattr(self,'w{}'.format(i)).mul_(1 - self.opt_lr * self.weight_decay)

                if self.num_rfr_neuron_layers[i-1]>0:
                    getattr(self,'w_rfr{}'.format(i)).mul_(1 - self.opt_lr * self.weight_decay)
                    getattr(self,'rfr{}'.format(i-1)).mul_(1 - self.opt_lr * self.rfr_decay)

        
    def load(self, path_data, sample_index):
        super().load(path_data,sample_index)
        data=np.load(path_data)

        # load the rfr
        self.num_rfr_neuron_layers=data['num_rfr_neuron_layers']
        with torch.no_grad():
            for i, num_rfr_neuron in enumerate(self.num_rfr_neuron_layers):
                if num_rfr_neuron>0:
                    setattr(self, 'rfr{}'.format(i), torch.from_numpy(data[ 'rfr{}'.format(i)][sample_index]).to(self.device))
                    setattr(self, 'w_rfr{}'.format(i+1), torch.from_numpy(data[ 'w_rfr{}'.format(i+1)][sample_index]).to(self.device))
                    getattr(self, 'w_rfr{}'.format(i+1)).requires_grad=True
                    getattr(self, 'rfr{}'.format(i)).requires_grad=True

class net_rfr_spk_delay(net_rfr_spk):
        
    name = 'net_rfr_spk_delay'
    def __init__(self, sample_index, kwargs):
        super().__init__(sample_index, kwargs)
        self.kernel_size = kwargs['delay_size']

    def delay_propogation(self, i):
        
        kernel = getattr(self, 'd{}_tensor'.format(i))*getattr(self, 'w{}'.format(i)).T.unsqueeze(2)
        spk_tensor_pad = torch.nn.functional.pad(torch.swapaxes(getattr(self, 'spk{}_rec_tensor'.format(i-1))*self.dropout_mask_list[i-1], 1, 2), pad=(self.kernel_size//2,self.kernel_size//2,0,0,0,0,),mode='constant', value=0)  # batch_size, in_neuorn, time_steps
        #The padding size by which to pad some dimensions of input are described starting from the last dimension and moving forward. 
        syn_current = torch.nn.functional.conv1d(spk_tensor_pad,kernel)

        return torch.swapaxes(syn_current, 1, 2)

    def run_snn(self):
        '''run a SNN with a feed-forward structure defined by self.layer_struct
        syn{} the synaptic current of a layer, (batch_size, num_neuron)
        mem{} the membrane potential of a layer
        spk{} the spike output of a layer
        spk{}_rec_list, record the spk at different time
        syn_current, the synaptic current of a layer, (batch_size, time step, num_neuron)
        spk{}_rec_tensor, the spk output of a layer, (batch_size, time step, num_neuron)
        '''
        
        self.dropout_mask_list = []
        for i in range(len(self.layer_struct)-1):
            # if i==1:
            #     prob = torch.ones(1,1,self.layer_struct[i]).to(self.device)
            #     self.dropout_mask_list.append(prob)
            # else:
            prob = torch.zeros(1,1,self.layer_struct[i]).to(self.device)+1-self.dropout_rate
            dropout_mask = torch.bernoulli(prob)
            self.dropout_mask_list.append(dropout_mask/(1-self.dropout_rate))

        self.spk0_rec_tensor=self.x_data.to(self.device) # size: [batch_size, time_step, num_neuron]
        batch_size = self.x_data.shape[0]
        
        # generate rfr spks
        if np.sum(self.num_rfr_neuron_layers)>0:
            self.get_rfr_spk_from_potent()

        for i in range(1, len(self.layer_struct)):
            
            # initial value of syn and mem
            setattr(self, 'syn{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) )  # the synaptic input per layer
            setattr(self, 'mem{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) ) # the membrane potential per layer

            # create list to save the spk
            setattr(self, 'spk{}_rec_list'.format(i), [])

            # calculate the synaptic current input each layer from last layer
            # print(getattr(self, 'spk{}_rec_tensor'.format(i-1)).device)
            # print(getattr(self,'w{}'.format(i)).device)
            syn_current = self.delay_propogation(i)

            # calculate the synaptic current input each layer from corresponding reference neurons
            if self.num_rfr_neuron_layers[i-1]>0:
                setattr(self, 'syn_rfr{}'.format(i), torch.zeros((1, self.layer_struct[i]), device=self.device , requires_grad=True) )  # the batch size is one, the synaptic input from reference neurons per layer
                syn_current_rfr=torch.einsum("btx,xy->bty", (getattr(self, 'spk_rfr{}_rec_tensor'.format(i-1)), (getattr(self,'w_rfr{}'.format(i)))))
            
            for t in range(self.num_step):

                setattr(self, 'spk{}'.format(i), self.spike_fn(getattr(self,'mem{}'.format(i))-1.0))
                getattr(self, 'spk{}_rec_list'.format(i)).append(getattr(self, 'spk{}'.format(i)))
                if self.num_rfr_neuron_layers[i-1]==0:
                    setattr(self, 'mem{}'.format(i), self.beta*(getattr(self, 'mem{}'.format(i)) * (1.0-getattr(self, 'spk{}'.format(i)).detach())+getattr(self, 'syn{}'.format(i))*self.time_step))
                    setattr(self, 'syn{}'.format(i), self.alpha*(getattr(self, 'syn{}'.format(i)) + syn_current[:,t,:]))
                else:
                    setattr(self, 'mem{}'.format(i), self.beta*(getattr(self, 'mem{}'.format(i)) * (1.0-getattr(self, 'spk{}'.format(i)).detach()) +getattr(self, 'syn{}'.format(i))*self.time_step+ getattr(self, 'syn_rfr{}'.format(i))*self.time_step))
                    setattr(self, 'syn{}'.format(i), self.alpha*(getattr(self, 'syn{}'.format(i)) + syn_current[:,t,:]))
                    setattr(self, 'syn_rfr{}'.format(i), self.alpha_rfr*(getattr(self, 'syn_rfr{}'.format(i)) + syn_current_rfr[:,t,:]))
                    
            # make the spk output of current layer a tensor
            setattr(self, 'spk{}_rec_tensor'.format(i), torch.stack(getattr(self, 'spk{}_rec_list'.format(i)),dim=1)) # size: [batch_size, time_step, num_neuron]

        return getattr(self, 'spk{}_rec_tensor'.format(i))
    
    def init_plastic_paras(self, kwargs):
        
        super().init_plastic_paras(kwargs)
        
        # create empty delay matrix for all smaples
        for i in range(1, len(self.layer_struct)):
                setattr(self, 'd{}_temp'.format(i), torch.randint(-(self.kernel_size//2), self.kernel_size//2+1, (self.sampling_size, self.layer_struct[i-1], self.layer_struct[i]), device=self.device))

        # keep the weight for a specifc sample_index
        for i in range(1, len(self.layer_struct)):
            setattr(self, 'd{}'.format(i), getattr(self,'d{}_temp'.format(i))[self.sample_index,:,:])
            setattr(self, 'd{}_tensor'.format(i), torch.zeros((self.layer_struct[i], self.layer_struct[i-1], self.kernel_size)).to(self.device) )
            getattr(self, 'd{}_tensor'.format(i))[torch.arange(self.layer_struct[i]).unsqueeze(1), torch.arange(self.layer_struct[i-1]).unsqueeze(0), self.kernel_size//2-getattr(self, 'd{}'.format(i)).T] = 1
          
class net_LIF_delay(net_LIF):
        
    name = 'net_LIF_delay'
    #*# save and load the time delay later, if this methods is sucessful
    def __init__(self, sample_index, kwargs):
        super().__init__(sample_index, kwargs)
        self.kernel_size = kwargs['delay_size']
    
    def delay_propogation(self, i):
        
        kernel = getattr(self, 'd{}_tensor'.format(i))*getattr(self, 'w{}'.format(i)).T.unsqueeze(2)
        spk_tensor_pad = torch.nn.functional.pad(torch.swapaxes(getattr(self, 'spk{}_rec_tensor'.format(i-1))*self.dropout_mask_list[i-1], 1, 2), pad=(self.kernel_size//2,self.kernel_size//2,0,0,0,0,),mode='constant', value=0)  # batch_size, in_neuorn, time_steps
        #The padding size by which to pad some dimensions of input are described starting from the last dimension and moving forward. 
        syn_current = torch.nn.functional.conv1d(spk_tensor_pad,kernel)

        return torch.swapaxes(syn_current, 1, 2)

    def run_snn(self):
        '''run a SNN with a feed-forward structure defined by self.layer_struct
        syn{} the synaptic current of a layer, (batch_size, num_neuron)
        mem{} the membrane potential of a layer
        spk{} the spike output of a layer
        spk{}_rec_list, record the spk at different time
        syn_current, the synaptic current of a layer, (batch_size, time step, num_neuron)
        spk{}_rec_tensor, the spk output of a layer, (batch_size, time step, num_neuron)
        '''
        self.dropout_mask_list = []
        for i in range(len(self.layer_struct)-1):
            # if i==1:
            #     prob = torch.ones(1,1,self.layer_struct[i]).to(self.device)
            #     self.dropout_mask_list.append(prob)
            # else:
            prob = torch.zeros(1,1,self.layer_struct[i]).to(self.device)+1-self.dropout_rate
            dropout_mask = torch.bernoulli(prob)
            self.dropout_mask_list.append(dropout_mask/(1-self.dropout_rate))
            
        self.spk0_rec_tensor=self.x_data.to(self.device) # size: [batch_size, time_step, num_neuron]
        batch_size = self.x_data.shape[0]

        for i in range(1, len(self.layer_struct)):
            
            # initial value of syn and mem
            setattr(self, 'syn{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) )  # the synaptic input per layer
            setattr(self, 'mem{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) ) # the membrane potential per layer

            # create list to save the spk
            setattr(self, 'spk{}_rec_list'.format(i), [])

            # calculate the synaptic current input each layer from last layer
            # print(getattr(self, 'spk{}_rec_tensor'.format(i-1)).device)
            # print(getattr(self,'w{}'.format(i)).device)
            syn_current = self.delay_propogation(i)
            
            for t in range(self.num_step):

                setattr(self, 'spk{}'.format(i), self.spike_fn(getattr(self,'mem{}'.format(i))-1.0))
                getattr(self, 'spk{}_rec_list'.format(i)).append(getattr(self, 'spk{}'.format(i)))
                setattr(self, 'mem{}'.format(i), self.beta*(getattr(self, 'mem{}'.format(i)) * (1.0-getattr(self, 'spk{}'.format(i)).detach())+getattr(self, 'syn{}'.format(i))*self.time_step))
                setattr(self, 'syn{}'.format(i), self.alpha*(getattr(self, 'syn{}'.format(i)) + syn_current[:,t,:]))

            # make the spk output of current layer a tensor
            setattr(self, 'spk{}_rec_tensor'.format(i), torch.stack(getattr(self, 'spk{}_rec_list'.format(i)),dim=1)) # size: [batch_size, time_step, num_neuron]

        return getattr(self, 'spk{}_rec_tensor'.format(i))
    
    def init_plastic_paras(self, kwargs):
        
        super().init_plastic_paras(kwargs)
        
        # create empty delay matrix for all smaples
        for i in range(1, len(self.layer_struct)):
                setattr(self, 'd{}_temp'.format(i), torch.randint(-(self.kernel_size//2), self.kernel_size//2+1, (self.sampling_size, self.layer_struct[i-1], self.layer_struct[i]), device=self.device))

        # keep the weight for a specifc sample_index
        for i in range(1, len(self.layer_struct)):
            setattr(self, 'd{}'.format(i), getattr(self,'d{}_temp'.format(i))[self.sample_index,:,:])
            setattr(self, 'd{}_tensor'.format(i), torch.zeros((self.layer_struct[i], self.layer_struct[i-1], self.kernel_size)).to(self.device) )
            getattr(self, 'd{}_tensor'.format(i))[torch.arange(self.layer_struct[i]).unsqueeze(1), torch.arange(self.layer_struct[i-1]).unsqueeze(0), self.kernel_size//2-getattr(self, 'd{}'.format(i)).T] = 1
    
    def load(self, path_data, sample_index):
        super().load(path_data, sample_index)
        data=np.load(path_data)

        # load the weight
        with torch.no_grad():
            for i in range(1, len(self.layer_struct)):
                setattr(self, 'd{}'.format(i), torch.from_numpy(data[ 'd{}'.format(i)][sample_index]).to(self.device))
                setattr(self, 'd{}_tensor'.format(i), torch.zeros((self.layer_struct[i], self.layer_struct[i-1], self.kernel_size)).to(self.device) )
                getattr(self, 'd{}_tensor'.format(i))[torch.arange(self.layer_struct[i]).unsqueeze(1), torch.arange(self.layer_struct[i-1]).unsqueeze(0), self.kernel_size//2-getattr(self, 'd{}'.format(i)).T] = 1

class net_rfr_spk_random_mem(net_rfr_spk):
        
    name = 'net_rfr_spk_random_mem'
        
    def __init__(self, sample_index, kwargs):
        super().__init__(sample_index, kwargs)

        self.betas_list = []
        for i in range(1, len(self.layer_struct)):
            tau_mems = torch.rand(1, self.layer_struct[i]).to(self.device)*10+10
            betas = torch.exp(-self.time_step/tau_mems)
            self.betas_list.append(betas)

    def run_snn(self):
        '''run a SNN with a feed-forward structure defined by self.layer_struct
        syn{} the synaptic current of a layer, (batch_size, num_neuron)
        mem{} the membrane potential of a layer
        spk{} the spike output of a layer
        spk{}_rec_list, record the spk at different time
        syn_current, the synaptic current of a layer, (batch_size, time step, num_neuron)
        spk{}_rec_tensor, the spk output of a layer, (batch_size, time step, num_neuron)
        '''
        # generate the nernoulli mask to dropout neurons in each layer
        
        self.dropout_mask_list = []
        for i in range(len(self.layer_struct)-1):
            prob = torch.zeros(1,1,self.layer_struct[i]).to(self.device)+1-self.dropout_rate
            dropout_mask = torch.bernoulli(prob)
            self.dropout_mask_list.append(dropout_mask/(1-self.dropout_rate))

        self.spk0_rec_tensor=self.x_data.to(self.device) # size: [batch_size, time_step, num_neuron]
        batch_size = self.x_data.shape[0]
        
        # generate rfr spks
        if np.sum(self.num_rfr_neuron_layers)>0:
            self.get_rfr_spk_from_potent()

        for i in range(1, len(self.layer_struct)):
            
            # initial value of syn and mem
            setattr(self, 'syn{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) )  # the synaptic input per layer
            setattr(self, 'mem{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) ) # the membrane potential per layer

            # create list to save the spk
            setattr(self, 'spk{}_rec_list'.format(i), [])

            # calculate the synaptic current input each layer from last layer
            # print(getattr(self, 'spk{}_rec_tensor'.format(i-1)).device)
            # print(getattr(self,'w{}'.format(i)).device)
            syn_current = torch.einsum("btx,xy->bty", (getattr(self, 'spk{}_rec_tensor'.format(i-1))*self.dropout_mask_list[i-1], getattr(self,'w{}'.format(i))))

            # calculate the synaptic current input each layer from corresponding reference neurons
            if self.num_rfr_neuron_layers[i-1]>0:
                setattr(self, 'syn_rfr{}'.format(i), torch.zeros((1, self.layer_struct[i]), device=self.device , requires_grad=True) )  # the batch size is one, the synaptic input from reference neurons per layer
                syn_current_rfr=torch.einsum("btx,xy->bty", (getattr(self, 'spk_rfr{}_rec_tensor'.format(i-1)), (getattr(self,'w_rfr{}'.format(i)))))
            
            for t in range(self.num_step):

                setattr(self, 'spk{}'.format(i), self.spike_fn(getattr(self,'mem{}'.format(i))-1.0))
                getattr(self, 'spk{}_rec_list'.format(i)).append(getattr(self, 'spk{}'.format(i)))
                if self.num_rfr_neuron_layers[i-1]==0:
                    setattr(self, 'mem{}'.format(i), self.betas_list[i-1]*(getattr(self, 'mem{}'.format(i)) * (1.0-getattr(self, 'spk{}'.format(i)).detach())+getattr(self, 'syn{}'.format(i))*self.time_step))
                    setattr(self, 'syn{}'.format(i), self.alpha*(getattr(self, 'syn{}'.format(i)) + syn_current[:,t,:]))
                else:
                    setattr(self, 'mem{}'.format(i), self.betas_list[i-1]*(getattr(self, 'mem{}'.format(i)) * (1.0-getattr(self, 'spk{}'.format(i)).detach()) +getattr(self, 'syn{}'.format(i))*self.time_step+ getattr(self, 'syn_rfr{}'.format(i))*self.time_step))
                    setattr(self, 'syn{}'.format(i), self.alpha*(getattr(self, 'syn{}'.format(i)) + syn_current[:,t,:]))
                    setattr(self, 'syn_rfr{}'.format(i), self.alpha_rfr*(getattr(self, 'syn_rfr{}'.format(i)) + syn_current_rfr[:,t,:]))
                    
            # make the spk output of current layer a tensor
            setattr(self, 'spk{}_rec_tensor'.format(i), torch.stack(getattr(self, 'spk{}_rec_list'.format(i)),dim=1)) # size: [batch_size, time_step, num_neuron]

        return getattr(self, 'spk{}_rec_tensor'.format(i))

class net_LIF_trainable_thresh(net_LIF):
        
    name = 'net_LIF_trainable_bias'

    # network 
    def run_snn(self):
        '''run a SNN with a feed-forward structure defined by self.layer_struct
        syn{} the synaptic current of a layer, (batch_size, num_neuron)
        mem{} the membrane potential of a layer
        spk{} the spike output of a layer
        spk{}_rec_list, record the spk at different time
        syn_current, the synaptic current of a layer, (batch_size, time step, num_neuron)
        spk{}_rec_tensor, the spk output of a layer, (batch_size, time step, num_neuron)
        '''
        self.dropout_mask_list = []
        for i in range(len(self.layer_struct)-1):
            # if i==1:
            #     prob = torch.ones(1,1,self.layer_struct[i]).to(self.device)
            #     self.dropout_mask_list.append(prob)
            # else:
            prob = torch.zeros(1,1,self.layer_struct[i]).to(self.device)+1-self.dropout_rate
            dropout_mask = torch.bernoulli(prob)
            self.dropout_mask_list.append(dropout_mask/(1-self.dropout_rate))

        self.spk0_rec_tensor=self.x_data # size: [batch_size, time_step, num_neuron]
        batch_size = self.x_data.shape[0]

        for i in range(1, len(self.layer_struct)):
            
            # initial value of syn and mem
            setattr(self, 'syn{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) )  # the synaptic input of per layer
            setattr(self, 'mem{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) ) # the membrane potential of per layer

            # create list to save the spk
            setattr(self, 'spk{}_rec_list'.format(i), [])

            # calculate the synaptic current input each layer from last layer
            syn_current = torch.einsum("btx,xy->bty", (getattr(self, 'spk{}_rec_tensor'.format(i-1))*self.dropout_mask_list[i-1], getattr(self,'w{}'.format(i))))
            
            for t in range(self.num_step):

                setattr(self, 'spk{}'.format(i), self.spike_fn(getattr(self,'mem{}'.format(i))-getattr(self,'b{}'.format(i))))
                getattr(self, 'spk{}_rec_list'.format(i)).append(getattr(self, 'spk{}'.format(i)))
                setattr(self, 'mem{}'.format(i), self.beta*(getattr(self, 'mem{}'.format(i)) * (1.0-getattr(self, 'spk{}'.format(i)).detach())+getattr(self, 'syn{}'.format(i))*self.time_step))
                setattr(self, 'syn{}'.format(i), self.alpha*(getattr(self, 'syn{}'.format(i)) + syn_current[:,t,:]))

            # make the spk output of current layer a tensor
            setattr(self, 'spk{}_rec_tensor'.format(i), torch.stack(getattr(self, 'spk{}_rec_list'.format(i)),dim=1)) # size: [batch_size, time_step, num_neuron]

        return getattr(self, 'spk{}_rec_tensor'.format(i))

    # training
    def init_plastic_paras(self, kwargs):
        super().init_plastic_paras(kwargs)

        # the bias are intialized as one
        for i in range(1, len(self.layer_struct)):
                setattr(self, 'b{}_temp'.format(i), torch.ones((self.sampling_size, self.layer_struct[i]), device=self.device))

        # keep the bias for a specifc sample_index
        for i in range(1, len(self.layer_struct)):

            setattr(self, 'b{}'.format(i), getattr(self,'b{}_temp'.format(i))[self.sample_index,:])
            getattr(self,'b{}'.format(i)).requires_grad=True

        # init clipper
        self.CLIP=kwargs.get('CLIP', False)
        if self.CLIP:
            self.b_range=kwargs.get('b_range')

    def init_optimizer(self, kwargs):
        self.params=[]

        for i in range(1, len(self.layer_struct)):
            self.params.append(getattr(self,'w{}'.format(i)))
            self.params.append(getattr(self,'b{}'.format(i)))
            
        self.optimizer = torch.optim.Adam(self.params, lr=kwargs.get('opt_lr'), betas=(0.9,0.999))
        if 'scheduler' in kwargs.keys():
            self.scheduler = globals()[kwargs['scheduler']](self.optimizer, **kwargs['scheduler_kwargs'])

    # load learned parameters
    def load(self, path_data, sample_index):
        super().load(path_data, sample_index)
        data=np.load(path_data)

        # load the weight
        with torch.no_grad():
            for i in range(1, len(self.layer_struct)):
                setattr(self, 'b{}'.format(i), torch.from_numpy(data[ 'b{}'.format(i)][sample_index]).to(self.device))
                getattr(self, 'b{}'.format(i)).requires_grad=True
                
    def clipping(self):
        '''clip the weight, make sure it's still biological'''
        with torch.no_grad():
            for i in range(1, len(self.layer_struct)):
                getattr(self,'w{}'.format(i)).clamp_(*self.w_range)
                getattr(self,'b{}'.format(i)).clamp_(*self.b_range)

class net_LIF_adaptive_thresh(net_LIF):
        
    name = 'net_LIF_adaptive_thresh'

    def __init__(self, sample_index, kwargs):
        super().__init__(sample_index, kwargs)
        self.b0_adptive_thresh=kwargs['b0_adptive_thresh']
        self.beta_adaptive_thresh=kwargs['beta_adaptive_thresh']
        self.rho_adaptive_thresh = np.exp(-self.time_step/kwargs['tau_adaptive_thresh'])

    # network 
    def run_snn(self):
        '''run a SNN with a feed-forward structure defined by self.layer_struct
        syn{} the synaptic current of a layer, (batch_size, num_neuron)
        mem{} the membrane potential of a layer
        spk{} the spike output of a layer
        spk{}_rec_list, record the spk at different time
        syn_current, the synaptic current of a layer, (batch_size, time step, num_neuron)
        spk{}_rec_tensor, the spk output of a layer, (batch_size, time step, num_neuron)
        '''
        self.dropout_mask_list = []
        for i in range(len(self.layer_struct)-1):
            # if i==1:
            #     prob = torch.ones(1,1,self.layer_struct[i]).to(self.device)
            #     self.dropout_mask_list.append(prob)
            # else:
            prob = torch.zeros(1,1,self.layer_struct[i]).to(self.device)+1-self.dropout_rate
            dropout_mask = torch.bernoulli(prob)
            self.dropout_mask_list.append(dropout_mask/(1-self.dropout_rate))

        self.spk0_rec_tensor=self.x_data # size: [batch_size, time_step, num_neuron]
        batch_size = self.x_data.shape[0]

        for i in range(1, len(self.layer_struct)):
            
            # initial value of syn, mem, and thresh
            setattr(self, 'syn{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) )  # the synaptic input of per layer #! check if require_grad is necessary
            setattr(self, 'mem{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) ) # the membrane potential of per layer
            setattr(self, 'b{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=False) ) # the b from the threshold of neurons per layer, which are different per batch

            # create list to save the spk
            setattr(self, 'spk{}_rec_list'.format(i), [])

            # calculate the synaptic current input each layer from last layer
            syn_current = torch.einsum("btx,xy->bty", (getattr(self, 'spk{}_rec_tensor'.format(i-1))*self.dropout_mask_list[i-1], getattr(self,'w{}'.format(i))))
            
            for t in range(self.num_step):
                
                thresh = self.b0_adptive_thresh + self.beta_adaptive_thresh*getattr(self,'b{}'.format(i)) # calculate the adaptive threshold
                setattr(self, 'spk{}'.format(i), self.spike_fn(getattr(self,'mem{}'.format(i))-thresh))
                getattr(self, 'spk{}_rec_list'.format(i)).append(getattr(self, 'spk{}'.format(i)))
                setattr(self, 'mem{}'.format(i), self.beta*(getattr(self, 'mem{}'.format(i)) * (1.0-getattr(self, 'spk{}'.format(i)).detach())+getattr(self, 'syn{}'.format(i))*self.time_step))
                setattr(self, 'syn{}'.format(i), self.alpha*(getattr(self, 'syn{}'.format(i)) + syn_current[:,t,:]))
                setattr(self, 'b{}'.format(i), self.rho_adaptive_thresh*getattr(self, 'b{}'.format(i)) + (1-self.rho_adaptive_thresh)*getattr(self, 'spk{}'.format(i)))
            
            # make the spk output of current layer a tensor
            setattr(self, 'spk{}_rec_tensor'.format(i), torch.stack(getattr(self, 'spk{}_rec_list'.format(i)),dim=1)) # size: [batch_size, time_step, num_neuron]

        return getattr(self, 'spk{}_rec_tensor'.format(i))
    
class net_LIF_trainable_thresh_delay(net_LIF_delay):
        
    name = 'net_LIF_trainable_bias_delay'

    # network 
    def run_snn(self):
        '''run a SNN with a feed-forward structure defined by self.layer_struct
        syn{} the synaptic current of a layer, (batch_size, num_neuron)
        mem{} the membrane potential of a layer
        spk{} the spike output of a layer
        spk{}_rec_list, record the spk at different time
        syn_current, the synaptic current of a layer, (batch_size, time step, num_neuron)
        spk{}_rec_tensor, the spk output of a layer, (batch_size, time step, num_neuron)
        '''
        self.dropout_mask_list = []
        for i in range(len(self.layer_struct)-1):
            # if i==1:
            #     prob = torch.ones(1,1,self.layer_struct[i]).to(self.device)
            #     self.dropout_mask_list.append(prob)
            # else:
            prob = torch.zeros(1,1,self.layer_struct[i]).to(self.device)+1-self.dropout_rate
            dropout_mask = torch.bernoulli(prob)
            self.dropout_mask_list.append(dropout_mask/(1-self.dropout_rate))
            
        self.spk0_rec_tensor=self.x_data.to(self.device) # size: [batch_size, time_step, num_neuron]
        batch_size = self.x_data.shape[0]

        for i in range(1, len(self.layer_struct)):
            
            # initial value of syn and mem
            setattr(self, 'syn{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) )  # the synaptic input per layer
            setattr(self, 'mem{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) ) # the membrane potential per layer

            # create list to save the spk
            setattr(self, 'spk{}_rec_list'.format(i), [])

            # calculate the synaptic current input each layer from last layer
            syn_current = self.delay_propogation(i)
            
            for t in range(self.num_step):

                setattr(self, 'spk{}'.format(i), self.spike_fn(getattr(self,'mem{}'.format(i))-getattr(self,'b{}'.format(i))))
                getattr(self, 'spk{}_rec_list'.format(i)).append(getattr(self, 'spk{}'.format(i)))
                setattr(self, 'mem{}'.format(i), self.beta*(getattr(self, 'mem{}'.format(i)) * (1.0-getattr(self, 'spk{}'.format(i)).detach())+getattr(self, 'syn{}'.format(i))*self.time_step))
                setattr(self, 'syn{}'.format(i), self.alpha*(getattr(self, 'syn{}'.format(i)) + syn_current[:,t,:]))

            # make the spk output of current layer a tensor
            setattr(self, 'spk{}_rec_tensor'.format(i), torch.stack(getattr(self, 'spk{}_rec_list'.format(i)),dim=1)) # size: [batch_size, time_step, num_neuron]

        return getattr(self, 'spk{}_rec_tensor'.format(i))
    
    # training
    def init_plastic_paras(self, kwargs):
        super().init_plastic_paras(kwargs)

        # the bias are intialized as one
        for i in range(1, len(self.layer_struct)):
                setattr(self, 'b{}_temp'.format(i), torch.ones((self.sampling_size, self.layer_struct[i]), device=self.device))

        # keep the bias for a specifc sample_index
        for i in range(1, len(self.layer_struct)):

            setattr(self, 'b{}'.format(i), getattr(self,'b{}_temp'.format(i))[self.sample_index,:])
            getattr(self,'b{}'.format(i)).requires_grad=True

        # init clipper
        self.CLIP=kwargs.get('CLIP', False)
        if self.CLIP:
            self.b_range=kwargs.get('b_range')

    def init_optimizer(self, kwargs):
        self.params=[]

        for i in range(1, len(self.layer_struct)):
            self.params.append(getattr(self,'w{}'.format(i)))
            self.params.append(getattr(self,'b{}'.format(i)))
            
        self.optimizer = torch.optim.Adam(self.params, lr=kwargs.get('opt_lr'), betas=(0.9,0.999))
        if 'scheduler' in kwargs.keys():
            self.scheduler = globals()[kwargs['scheduler']](self.optimizer, **kwargs['scheduler_kwargs'])

    # load learned parameters
    def load(self, path_data, sample_index):
        super().load(path_data, sample_index)
        data=np.load(path_data)

        # load the weight
        with torch.no_grad():
            for i in range(1, len(self.layer_struct)):
                setattr(self, 'b{}'.format(i), torch.from_numpy(data[ 'b{}'.format(i)][sample_index]).to(self.device))
                getattr(self, 'b{}'.format(i)).requires_grad=True
                
    def clipping(self):
        '''clip the weight, make sure it's still biological'''
        with torch.no_grad():
            for i in range(1, len(self.layer_struct)):
                getattr(self,'w{}'.format(i)).clamp_(*self.w_range)
                getattr(self,'b{}'.format(i)).clamp_(*self.b_range)

class net_LIF_adaptive_thresh_delay(net_LIF_delay):
        
    name = 'net_LIF_adaptive_thresh_delay'

    def __init__(self, sample_index, kwargs):
        super().__init__(sample_index, kwargs)
        self.b0_adptive_thresh=kwargs['b0_adptive_thresh']
        self.beta_adaptive_thresh=kwargs['beta_adaptive_thresh']
        self.rho_adaptive_thresh = np.exp(-self.time_step/kwargs['tau_adaptive_thresh'])

    # network 
    def run_snn(self):
        '''run a SNN with a feed-forward structure defined by self.layer_struct
        syn{} the synaptic current of a layer, (batch_size, num_neuron)
        mem{} the membrane potential of a layer
        spk{} the spike output of a layer
        spk{}_rec_list, record the spk at different time
        syn_current, the synaptic current of a layer, (batch_size, time step, num_neuron)
        spk{}_rec_tensor, the spk output of a layer, (batch_size, time step, num_neuron)
        '''
        self.dropout_mask_list = []
        for i in range(len(self.layer_struct)-1):
            # if i==1:
            #     prob = torch.ones(1,1,self.layer_struct[i]).to(self.device)
            #     self.dropout_mask_list.append(prob)
            # else:
            prob = torch.zeros(1,1,self.layer_struct[i]).to(self.device)+1-self.dropout_rate
            dropout_mask = torch.bernoulli(prob)
            self.dropout_mask_list.append(dropout_mask/(1-self.dropout_rate))

        self.spk0_rec_tensor=self.x_data.to(self.device) # size: [batch_size, time_step, num_neuron]
        batch_size = self.x_data.shape[0]

        for i in range(1, len(self.layer_struct)):
            
            # initial value of syn, mem, and thresh
            setattr(self, 'syn{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) )  # the synaptic input per layer
            setattr(self, 'mem{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) ) # the membrane potential per layer
            setattr(self, 'b{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=False) ) # the b from the threshold of neurons per layer, which are different per batch

            # create list to save the spk
            setattr(self, 'spk{}_rec_list'.format(i), [])

            # calculate the synaptic current input each layer from last layer
            syn_current = self.delay_propogation(i)
            
            for t in range(self.num_step):
                
                thresh = self.b0_adptive_thresh + self.beta_adaptive_thresh*getattr(self,'b{}'.format(i)) # calculate the adaptive threshold
                setattr(self, 'spk{}'.format(i), self.spike_fn(getattr(self,'mem{}'.format(i))-thresh))
                getattr(self, 'spk{}_rec_list'.format(i)).append(getattr(self, 'spk{}'.format(i)))
                setattr(self, 'mem{}'.format(i), self.beta*(getattr(self, 'mem{}'.format(i)) * (1.0-getattr(self, 'spk{}'.format(i)).detach())+getattr(self, 'syn{}'.format(i))*self.time_step))
                setattr(self, 'syn{}'.format(i), self.alpha*(getattr(self, 'syn{}'.format(i)) + syn_current[:,t,:]))
                setattr(self, 'b{}'.format(i), self.rho_adaptive_thresh*getattr(self, 'b{}'.format(i)) + (1-self.rho_adaptive_thresh)*getattr(self, 'spk{}'.format(i)))
            
            # make the spk output of current layer a tensor
            setattr(self, 'spk{}_rec_tensor'.format(i), torch.stack(getattr(self, 'spk{}_rec_list'.format(i)),dim=1)) # size: [batch_size, time_step, num_neuron]

        return getattr(self, 'spk{}_rec_tensor'.format(i))
    
class net_LIF_BN_delay(net_LIF_delay):
        
    name = 'net_LIF_BN_delay'

    # network 
    def run_snn(self):
        '''run a SNN with a feed-forward structure defined by self.layer_struct
        syn{} the synaptic current of a layer, (batch_size, num_neuron)
        mem{} the membrane potential of a layer
        spk{} the spike output of a layer
        spk{}_rec_list, record the spk at different time
        syn_current, the synaptic current of a layer, (batch_size, time step, num_neuron)
        spk{}_rec_tensor, the spk output of a layer, (batch_size, time step, num_neuron)
        '''
        self.dropout_mask_list = []
        for i in range(len(self.layer_struct)-1):
            prob = torch.zeros(1,1,self.layer_struct[i]).to(self.device)+1-self.dropout_rate
            dropout_mask = torch.bernoulli(prob)
            self.dropout_mask_list.append(dropout_mask/(1-self.dropout_rate))
            
        self.spk0_rec_tensor=self.x_data.to(self.device) # size: [batch_size, time_step, num_neuron]
        batch_size = self.x_data.shape[0]

        for i in range(1, len(self.layer_struct)):
            
            # initial value of syn and mem
            setattr(self, 'syn{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) )  # the synaptic input per layer
            setattr(self, 'mem{}'.format(i),torch.zeros((batch_size,self.layer_struct[i]), device=self.device, requires_grad=True) ) # the membrane potential per layer

            # create list to save the spk
            setattr(self, 'spk{}_rec_list'.format(i), [])

            # calculate the synaptic current input each layer from last layer
            syn_current = self.delay_propogation(i)
            
            for t in range(self.num_step):

                setattr(self, 'spk{}'.format(i), self.spike_fn(getattr(self,'mem{}'.format(i))-1.0))
                getattr(self, 'spk{}_rec_list'.format(i)).append(getattr(self, 'spk{}'.format(i)))
                setattr(self, 'mem{}'.format(i), self.beta*(getattr(self, 'mem{}'.format(i)) * (1.0-getattr(self, 'spk{}'.format(i)).detach())+getattr(self, 'syn{}'.format(i))*self.time_step))
                setattr(self, 'syn{}'.format(i), self.alpha*(getattr(self, 'syn{}'.format(i)) + getattr(self, 'BN{}'.format(i))[t](syn_current[:,t,:]))) # batch normalize the synaptic current per time step

            # make the spk output of current layer a tensor
            setattr(self, 'spk{}_rec_tensor'.format(i), torch.stack(getattr(self, 'spk{}_rec_list'.format(i)),dim=1)) # size: [batch_size, time_step, num_neuron]

        return getattr(self, 'spk{}_rec_tensor'.format(i))
    
    # training
    def init_plastic_paras(self, kwargs):
        super().init_plastic_paras(kwargs)

        # intialize the batch_normalization 
        for i in range(1, len(self.layer_struct)):
            setattr(self, 'BN{}'.format(i), nn.ModuleList([nn.BatchNorm1d(self.layer_struct[i], eps=1e-4, momentum=0.1, affine=True) for j in range(self.num_step)]).to(self.device))

    def init_optimizer(self, kwargs):
        self.params=[]

        for i in range(1, len(self.layer_struct)):
            self.params.append(getattr(self,'w{}'.format(i)))
            self.params+=list(getattr(self,'BN{}'.format(i)).parameters())
            
        self.optimizer = torch.optim.Adam(self.params, lr=kwargs.get('opt_lr'), betas=(0.9,0.999))
        if 'scheduler' in kwargs.keys():
            self.scheduler = globals()[kwargs['scheduler']](self.optimizer, **kwargs['scheduler_kwargs'])

    def train(self,):
        super().train()
        for i in range(1, len(self.layer_struct)):
            getattr(self, 'BN{}'.format(i)).train()
    
    def eval(self,):
        super().eval()
        for i in range(1, len(self.layer_struct)):
            getattr(self, 'BN{}'.format(i)).eval()
            
    # load learned parameters
    def load(self, path_data, sample_index):
        super().load(path_data, sample_index)
        data=np.load(path_data)

        # load the weight
        with torch.no_grad():
            for i in range(1, len(self.layer_struct)):
                BN_numpy = data['BN{}'.format(i)][self.sample_index]
                BN_var = data['BN{}_var'.format(i)][self.sample_index]
                BN_mean = data['BN{}_mean'.format(i)][self.sample_index]
                for tensor_index, BN in enumerate(getattr(self, 'BN{}'.format(i))):
                    BN.running_mean = torch.from_numpy(BN_mean[tensor_index]).to(self.device)
                    BN.running_var = torch.from_numpy(BN_var[tensor_index]).to(self.device)
                    BN._parameters['weight'] = torch.from_numpy(BN_numpy[tensor_index*2]).to(self.device)
                    BN._parameters['bias'] = torch.from_numpy(BN_numpy[tensor_index*2+1]).to(self.device)
