import sys
import os
import random
import argparse
import yaml
import multiprocessing as mp
import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from datetime import datetime
from net_class import *
from datasets import *
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.stdout.reconfigure(line_buffering=True, write_through=True)

class simulation_shell:
    def __init__(self, config_file_path=None):
        
        # load the path of config file
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_path",type=str, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--device", type=str, default=None)

        # save the settings as attributes
        args, unknown = parser.parse_known_args()
        if args.config_path != None:
            config_file_path = args.config_path
        with open(config_file_path, "r") as config_file:
            kwargs = yaml.full_load(config_file)
            for key in kwargs.keys():
                setattr(self,key, kwargs[key])

        # load the setting from command line
        if args.seed != None:
            self.net_setting['seed'] = args.seed
        if args.device != None:
            self.net_setting['device']= args.device

        # complete the setting
        self.trail_number += self.net_setting['seed']
        
        if self.net_setting['device'] == 'cpu':
            self.net_setting['device'] = torch.device("cpu")
        else:
            self.net_setting['device'] = torch.device("cuda:0")
        self.dataset_setting['time_step'] = self.net_setting['time_step']
        self.dataset_setting['num_in_neuron'] = self.net_setting['layer_struct'][0]
        self.dataset_setting['device'] = self.net_setting['device']
        self.continue_learning_path = self.data_path+'array_{}.npz'.format(self.trail_number)
        if 'scheduler' in self.training_setting.keys():
            if self.training_setting['scheduler'] == 'OneCycleLR':
                self.training_setting['scheduler_kwargs'] = {'max_lr':self.training_setting['opt_lr'], 'steps_per_epoch':self.dataset_setting['num_example']//self.dataset_setting['batch_size'], 'epochs': self.training_setting['num_epoch']}

        if not os.path.exists(self.dataset_setting['data_path']):
            # generate the dataset if it dosen't exist
            os.makedirs(self.dataset_setting['data_path'])
            dataset_name = os.path.basename(os.path.normpath(self.dataset_setting['data_path']))
            
            if 'Sequential' in dataset_name:
                create_sequential_encoded_dataset(dataset_name[:-11], self.dataset_setting['num_in_neuron'], self.dataset_setting['duration'])
            elif 'Sliced' in dataset_name:
                time_interval=self.dataset_setting['duration']/28
                thresh=50
                create_slicing_encoded_dataset(dataset_name[:-7], thresh, time_interval)
            elif 'SHD' in dataset_name:
                create_SHD_dataset() 
  
        # create the class of network
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        self.net_class = globals()[self.net_setting['net_class']]

    def learn(self, sample_index, AMP=False):

        print('sample index :{} learning'.format(sample_index))

        # set random seeds 
        np.random.seed(self.net_setting['seed'])
        torch.manual_seed(self.net_setting['seed']+256)
        random.seed(self.net_setting['seed'])
        torch.cuda.manual_seed(self.net_setting['seed']+256)
        torch.cuda.manual_seed_all(self.net_setting['seed']+256)
        torch.use_deterministic_algorithms(True)
        # torch.backends.cudnn.deterministic = True

        # create the network
        self.net=self.net_class(sample_index, self.net_setting)
        self.net.AMP = AMP
        self.net.scaler = torch.cuda.amp.GradScaler(enabled=self.net.AMP)
        if 'cuda' in str(self.net.device):
            device_str = 'cuda'
        else:
            device_str = 'cpu'

        # load the dataset
        self.net.duration=self.dataset_setting['duration']
        self.net.num_step=int(self.dataset_setting['duration']/self.net.time_step)

        self.ds= globals()[self.dataset_setting['name']](self.dataset_setting, sample_index)
        self.dl= DataLoader(self.ds, batch_size=self.dataset_setting['batch_size'], shuffle=self.dataset_setting['shuffle'], drop_last=True)
        
        if self.VALIDATION:
            self.val_ds= globals()[self.dataset_setting['name']](self.dataset_setting, sample_index, True)
            self.val_dl= DataLoader(self.val_ds, batch_size=self.dataset_setting['batch_size'], shuffle=False, drop_last=True)

        # init the training 
        self.net.init_train(self.training_setting)
        self.net.init_plastic_paras(self.training_setting)
        if self.CONTINUE == True:
            self.net.load(self.continue_learning_path, sample_index)
        self.net.init_optimizer(self.training_setting)

        # training of the model
        acc_batches_hist = []
        val_acc_batches_hist = []
        loss_batches_hist = []
        val_loss_batches_hist = []
        for epoch in range(self.training_setting['num_epoch']):
            
            self.net.train()
            
            # train_model
            iteration=0
            acc_batches = []
            loss_batches = []
            print('Training epoch {} -------------------------'.format(epoch))
            for x_data, y_data in self.dl:

                self.net.x_data= x_data
                self.net.y_data= y_data.to(self.net.device)

                if device_str=='cuda':
                    with torch.autocast(device_type=device_str, dtype=torch.float16, enabled=self.net.AMP):
                        output=self.net.run_forward()
                else:
                    output=self.net.run_forward()
                    
                loss_batches.append(self.net.loss_value.detach().cpu().numpy())
                acc_batches.append(self.net.acc_value.detach().cpu().numpy())
                
                if not self.INFERENCE:
                    self.net.train_backward()
                    
                # priting
                if iteration%20==0:
                    
                    print('batch {} loss : {}'.format(iteration, self.net.loss_value))

                iteration=iteration+1
                
            output_last_batch = output
            acc_batches_hist.append(acc_batches)
            loss_batches_hist.append(loss_batches)
            print('acc: {}'.format(np.average(acc_batches)))
            
            # validation
            if self.VALIDATION:
                
                self.net.eval()
                
                acc_batches = []
                loss_batches = []
                print('Validation -------------------------')
                with torch.no_grad():
                    for x_data, y_data in self.val_dl:

                        self.net.x_data= x_data
                        self.net.y_data= y_data.to(self.net.device)

                        if device_str=='cuda':
                            with torch.autocast(device_type=device_str, dtype=torch.float16, enabled=self.net.AMP):
                                output=self.net.run_forward()
                        else:
                            output=self.net.run_forward()
                            
                        loss_batches.append(self.net.loss_value.detach().cpu().numpy())
                        acc_batches.append(self.net.acc_value.detach().cpu().numpy())
                
                val_acc_batches_hist.append(acc_batches)
                val_loss_batches_hist.append(loss_batches)
                print('acc: {}'.format(np.average(acc_batches)))
                    
                    
        # record spk_count
        spk_counts_iterations=[]
        for i in range(1, len(self.net.layer_struct)):
            spk_counts_iterations.append(getattr(self.net, 'spk{}_count_iterations'.format(i)))
        ## for rfr layers
        if hasattr(self.net, 'num_rfr_neuron_layers'):
            for i in range(0, len(self.net.layer_struct)-1):
                if self.net.num_rfr_neuron_layers[i]>0:
                    spk_counts_iterations.append(getattr(self.net, 'rfr_spk{}_count_iterations'.format(i)))
        
        states_dict={}
        for i in range(1,len(self.net_setting['layer_struct'])):
            states_dict['w{}'.format(i)]=getattr(self.net, 'w{}'.format(i)).detach().cpu().numpy().copy()
            if 'bias' in self.net_class.name:
                states_dict['b{}'.format(i)]=getattr(self.net, 'b{}'.format(i)).detach().cpu().numpy().copy()
            if 'delay' in self.net_class.name:
                states_dict['d{}'.format(i)]=getattr(self.net, 'd{}'.format(i)).detach().cpu().numpy().copy()
            if 'rec' in self.net_class.name:
                if self.net.rec_bool_layers[i-1]:
                    states_dict['v{}'.format(i)]=getattr(self.net, 'v{}'.format(i)).detach().cpu().numpy().copy()
            if 'BN' in self.net_class.name:
                BN_parameters = torch.stack([parameter_tensor.detach().clone() for parameter_tensor in list(getattr(self.net, 'BN{}'.format(i)).parameters())],dim=0)
                states_dict['BN{}'.format(i)]=BN_parameters.detach().cpu().numpy().copy()
                
                BN_mean = torch.stack([BN.running_mean for BN in getattr(self.net, 'BN{}'.format(i))],dim=0)
                BN_var = torch.stack([BN.running_var for BN in getattr(self.net, 'BN{}'.format(i))],dim=0)
                states_dict['BN{}_mean'.format(i)]=BN_mean.detach().cpu().numpy().copy()
                states_dict['BN{}_var'.format(i)]=BN_var.detach().cpu().numpy().copy()
            
            if hasattr(self.net, 'num_rfr_neuron_layers'):
                if self.net.num_rfr_neuron_layers[i-1]>0:
                    states_dict['rfr{}'.format(i-1)]=getattr(self.net, 'rfr{}'.format(i-1)).cpu().detach().numpy().copy()
                    states_dict['w_rfr{}'.format(i)]=getattr(self.net, 'w_rfr{}'.format(i)).cpu().detach().numpy().copy()

        return states_dict, loss_batches_hist, acc_batches_hist, output_last_batch.detach(), spk_counts_iterations, val_loss_batches_hist, val_acc_batches_hist

    def run(self,):
        # print time
        current_time = datetime.now().strftime("%H:%M:%S")
        print("Start Time =", current_time)
        past = datetime.now()

        #run simulations paralelly in a pool of cpu
        if self.net_setting['device']  == torch.device("cpu"):
            sample_index_list=range(self.net_setting['sampling_size'])
            p=mp.Pool(self.num_workers)
            result_samples= p.map(self.learn, sample_index_list)
        else:
            result_samples= [self.learn(0, False)]   

        ## sort and collect the output from the poll
        self.states_dict_samples=[]
        self.loss_batches_hist_samples=[]
        self.acc_batches_hist_samples=[]
        output_last_batch_samples = []
        self.spk_counts_iterations_samples = []
        self.val_loss_batches_hist_samples=[]
        self.val_acc_batches_hist_samples=[]

        for sample_index_here in range(self.net_setting['sampling_size']):
            self.states_dict_samples+=[result_samples[sample_index_here][0]]
            self.loss_batches_hist_samples+=[result_samples[sample_index_here][1]]
            self.acc_batches_hist_samples+=[result_samples[sample_index_here][2]]
            output_last_batch_samples+=[result_samples[sample_index_here][3]]
            self.spk_counts_iterations_samples += [result_samples[sample_index_here][4]]
            self.val_loss_batches_hist_samples+=[result_samples[sample_index_here][5]]
            self.val_acc_batches_hist_samples+=[result_samples[sample_index_here][6]]
            
        self.output_last_batch_samples=torch.stack(output_last_batch_samples, axis=0)
        self.output_last_batch_samples=self.output_last_batch_samples.cpu().numpy()
        self.loss_batches_hist_samples=np.array(self.loss_batches_hist_samples)
        self.acc_batches_hist_samples=np.array(self.acc_batches_hist_samples)
        self.val_loss_batches_hist_samples=np.array(self.val_loss_batches_hist_samples)
        self.val_acc_batches_hist_samples=np.array(self.val_acc_batches_hist_samples)
        
        # print the ending
        now = datetime.now()
        current_time = datetime.now().strftime("%H:%M:%S")
        print("End Time =", current_time)
        print('the passing time is:')
        print(now-past)

        self.train_acc=float(np.average(np.average(self.acc_batches_hist_samples[:,-1,:])))
        if self.VALIDATION:
            self.val_acc=float(np.average(np.average(self.val_acc_batches_hist_samples[:,-1,:])))
        if self.SAVE:
            self.save(self.data_path, self.trail_number)

    def objective_adapt_tresh(self,trial):
        
        # modify the setting by optuna    
        b0_adptive_thresh = trial.suggest_float("b0_adptive_thresh", 0, 2, log=False)  
        beta_adaptive_thresh = trial.suggest_float("beta_adaptive_thresh", 0, 10, log=False) 
        tau_adaptive_thresh = trial.suggest_float("tau_adaptive_thresh", 1, 200, log=True) 
        
        self.net_setting['b0_adptive_thresh']=b0_adptive_thresh
        self.net_setting['beta_adaptive_thresh']=beta_adaptive_thresh
        self.net_setting['tau_adaptive_thresh']=tau_adaptive_thresh

        # run simulation
        self.run()

        return self.val_acc # return the test acc
    
    def gridsearch_adapt_tresh(self,b0_adptive_thresh,beta_list, tau_list):

        # beta_list = np.linspace(-4,0,10)
        # tau_list = np.linspace(0,2,5)
        beta, tau = np.meshgrid(beta_list, tau_list)
        paras = np.array([beta.reshape(-1), tau.reshape(-1)])

        # self.training_setting['num_epoch']=3
        val_accs = []
        for index_search in range(paras.shape[1]):
            self.net_setting['b0_adptive_thresh']= b0_adptive_thresh
            self.net_setting['beta_adaptive_thresh']= 10**(paras[0,index_search])
            self.net_setting['tau_adaptive_thresh']= 10**(paras[1,index_search])

            # run simulation
            self.run()

            val_accs.append(self.val_acc)

        return val_accs # return the test acc

    def save(self, data_path, seed_index):
        '''save the outcome'''
        saving={}
        for i in range(1, len(self.net_setting['layer_struct'])):
            saving['w{}'.format(i)]=np.stack([states_dict['w{}'.format(i)] for states_dict in self.states_dict_samples], axis=0)
            if 'bias' in self.net_class.name:
                saving['b{}'.format(i)]=np.stack([states_dict['b{}'.format(i)] for states_dict in self.states_dict_samples], axis=0)
            if 'delay' in self.net_class.name:
                saving['d{}'.format(i)]=np.stack([states_dict['d{}'.format(i)] for states_dict in self.states_dict_samples], axis=0)
            if 'rec' in self.net_class.name:
                if self.net.rec_bool_layers[i-1]:
                    saving['v{}'.format(i)]=np.stack([states_dict['v{}'.format(i)] for states_dict in self.states_dict_samples], axis=0)
            if 'BN' in self.net_class.name:
                saving['BN{}'.format(i)]=np.stack([states_dict['BN{}'.format(i)] for states_dict in self.states_dict_samples], axis=0)
                saving['BN{}_mean'.format(i)]=np.stack([states_dict['BN{}_mean'.format(i)] for states_dict in self.states_dict_samples], axis=0)
                saving['BN{}_var'.format(i)]=np.stack([states_dict['BN{}_var'.format(i)] for states_dict in self.states_dict_samples], axis=0)

        if 'num_rfr_neuron_layers' in self.training_setting.keys():
            for i in range(1, len(self.net_setting['layer_struct'])):
                if self.net_setting['num_rfr_neuron_layers'][i-1]>0:
                    saving['rfr{}'.format(i-1)]=np.stack([states_dict['rfr{}'.format(i-1)] for states_dict in self.states_dict_samples], axis=0)
                    saving['w_rfr{}'.format(i)]=np.stack([states_dict['w_rfr{}'.format(i)] for states_dict in self.states_dict_samples], axis=0)
    
            saving['num_rfr_neuron_layers']=self.net_setting['num_rfr_neuron_layers']

        saving['loss_batches_hist_samples']=self.loss_batches_hist_samples
        saving['acc_batches_hist_samples']=self.acc_batches_hist_samples
        saving['val_loss_batches_hist_samples']=self.val_loss_batches_hist_samples
        saving['val_acc_batches_hist_samples']=self.val_acc_batches_hist_samples
        saving['spk_counts_iterations_samples']=self.spk_counts_iterations_samples
        print('save the result at')
        print(data_path+'array_{}'.format(seed_index))
        np.savez(data_path+'array_{}'.format(seed_index), **saving, dtype='object')
        
        del saving

if __name__ == "__main__":
    # run the simulaiton
    time_start=time.time()
    simulation=simulation_shell()
    simulation.run()
    time_end=time.time()

    # save the train
    if simulation.CONTINUE==True:
        test_accs=np.load(simulation.data_path+'result_{}.npz'.format(simulation.trail_number))['test_accs'].tolist()
        train_accs=np.load(simulation.data_path+'result_{}.npz'.format(simulation.trail_number))['train_accs'].tolist()
    else:
        test_accs=[]
        train_accs=[]

    test_accs+=np.average(np.average(simulation.val_acc_batches_hist_samples[:,:,:], axis=0), axis=-1).tolist()
    train_accs+=np.average(np.average(simulation.acc_batches_hist_samples[:,:,:], axis=0), axis=-1).tolist()

    # print and save
    print('*****************')
    print('loss value:')
    print(float(np.average(simulation.loss_batches_hist_samples[:,-1,:])))
    print('the average acc of the test dataset({}) is:'.format(simulation.dataset_setting['test_num']))
    print(simulation.val_acc) #there is no validation set, or the validation set is the testset
    print('the time you spent is: ')
    print((time_end-time_start)/60)
    print('*****************')
    np.savez(simulation.data_path+'result_{}.npz'.format(simulation.trail_number), test_accs=test_accs, train_accs=train_accs)
    
    # save the result in txt file
    file_path = simulation.data_path+"result_{}.txt".format(simulation.trail_number)
    print('A text file is generated, {}'.format(file_path))
    with open(file_path, "w") as text_file:
        for i in range(len(train_accs)):
            text_file.write('epoch {} \n'.format(i))
            text_file.write('train acc: {}, test acc: {}\n'.format(train_accs[i], test_accs[i]))