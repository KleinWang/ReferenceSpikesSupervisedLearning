import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
import os
plt.rcdefaults()
plt.rcParams["figure.facecolor"]= 'white'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['font.size'] = 12
# plt.rcParams["figure.autolayout"]=True # IT WILL FAIL EPS GENERATION
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Times",
    "mathtext.fontset":"stix"}) # stix fits for the times for mathtext render
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# latex in matlibplot only support a few fonts, check the web below
# https://matplotlib.org/stable/tutorials/text/usetex.html

# https://matplotlib.org/stable/users/explain/text/mathtext.html , check how to change the font

SAVE = int(sys.argv[1])
format = sys.argv[2]
def get_result(path_name,index_list):
    seed_list = [1,2,3,4,5]
    
    train_acc_max_avg_list =[]
    test_acc_max_avg_list =[]
    test_acc_max_std_list =[]
    for index in index_list:
        train_acc_maxs = []
        test_acc_maxs = []
        for seed in seed_list:
            data = np.load(path_name+'result_{}.npz'.format(index+seed))
            train_accs = data['train_accs']*100
            test_accs = data['test_accs']*100
            test_acc_maxs.append(max(test_accs))
            train_acc_maxs.append(max(train_accs))
        train_acc_max_avg = np.mean(train_acc_maxs)
        test_acc_max_avg = np.mean(test_acc_maxs)
        train_acc_max_avg_list.append(train_acc_max_avg)
        test_acc_max_avg_list.append(test_acc_max_avg)
        test_acc_max_std_list.append(np.std(test_acc_maxs))

    return test_acc_max_avg_list, test_acc_max_std_list

if not os.path.exists('./plots'):
    os.mkdir('./plots')

#*# Manuscript -- temporal
if SAVE == True:
    test_accs_plots = []
    test_acc_stds_plots =[]
    dataset_names = ['MNIST_Sliced', 'MNIST_Sequential', 'FashionMNIST_Sliced','FashionMNIST_Sequential' ,'SHD']

    for dataset in dataset_names:
        path_name = './result_{}/'.format(dataset)
        if dataset == 'MNIST_Sliced':
            index_list = [3000, 3200]
        else:
            index_list = [3020, 3220]
        test_accs, test_acc_stds = get_result(path_name, index_list)
        test_accs_plots+=test_accs
        test_acc_stds_plots+=test_acc_stds
    test_accs_plots=np.array(test_accs_plots).reshape(-1,2)
    test_acc_stds_plots=np.array(test_acc_stds_plots).reshape(-1,2)
    np.savez('./plots/revision_temporal', test_accs_plots=test_accs_plots, test_acc_stds_plots=test_acc_stds_plots)
else:
    data = np.load('./plots/revision_temporal.npz')
    test_accs_plots = data['test_accs_plots']
    test_acc_stds_plots = data['test_acc_stds_plots']

fig=plt.figure()
fig.set_size_inches(6, 4)

dataset_names = ['MNIST_Sli', 'MNIST_Seq', 'Fashion_Sli', 'Fashion_Seq', 'SHD']
method1_accuracies = test_accs_plots[:,0]  # Accuracies for method 1
method1_stds = test_acc_stds_plots[:,0]
method2_accuracies = test_accs_plots[:,1]  # Accuracies for method 2
method2_stds = test_acc_stds_plots[:,1]
custom_colors1 = ['cornflowerblue' for i in dataset_names]
custom_colors2 = ['orange' for i in dataset_names]

# Set the width of the bars
bar_width = 0.25

# Create positions for the bars on the x-axis
x = range(len(dataset_names))

# Plot the bars for both networks
plt.bar(x, method1_accuracies,  width=bar_width, label='None', color = custom_colors1)
plt.errorbar(x, method1_accuracies, yerr=method1_stds, fmt='none', ecolor='black', capsize=5, capthick=0.7, elinewidth=0.7)
for i, acc in enumerate(method1_accuracies):
    plt.text(i, acc + 1.5, f"{acc:.1f}", ha="center", va="bottom", fontsize=9)
plt.bar([i + bar_width for i in x], method2_accuracies, width=bar_width, label='Ref', color = custom_colors2)
plt.errorbar([i + bar_width for i in x], method2_accuracies, yerr=method2_stds, fmt='none', ecolor='black', capsize=5, capthick=0.7, elinewidth=0.7)
for i, acc in enumerate(method2_accuracies):
    plt.text(i+ bar_width, acc + 1.5, f"{acc:.1f}", ha="center", va="bottom", fontsize=9)

# Customize the plot
# plt.xlabel('Datasets')
plt.ylabel('Accuracy (\%)')
plt.xticks([i + bar_width/2 for i in x], dataset_names)
plt.legend(loc='upper right', ncol=2, fontsize=9)
plt.grid(axis="y", linestyle="--", alpha=0.7)


# Show the plot
plt.ylim(0,110)
# plt.title('Temporally encoded')
plt.tight_layout()
fig.savefig('./plots/revision_temporal.'+format, format=format)
# fig.savefig('./plots/Temporal.eps', format='eps')

#*# Suuplementary -- rate
if SAVE == True:
    test_accs_plots = []
    test_acc_stds_plots =[]
    dataset_names = ['MNIST_Sliced', 'MNIST_Sequential', 'FashionMNIST_Sliced','FashionMNIST_Sequential' ,'SHD']

    for dataset in dataset_names:
        path_name = './result_{}/'.format(dataset)
        if dataset == 'MNIST_Sliced':
            index_list = [2000, 2200]
        else:
            index_list = [2020, 2220]
        test_accs, test_acc_stds = get_result(path_name, index_list)
        test_accs_plots+=test_accs
        test_acc_stds_plots+=test_acc_stds
    test_accs_plots=np.array(test_accs_plots).reshape(-1,2)
    test_acc_stds_plots=np.array(test_acc_stds_plots).reshape(-1,2)
    np.savez('./plots/revision_rate', test_accs_plots=test_accs_plots, test_acc_stds_plots=test_acc_stds_plots)
else:
    data = np.load('./plots/revision_rate.npz')
    test_accs_plots = data['test_accs_plots']
    test_acc_stds_plots = data['test_acc_stds_plots']

fig=plt.figure()
fig.set_size_inches(6, 4)

dataset_names = ['MNIST_Sli', 'MNIST_Seq', 'Fashion_Sli', 'Fashion_Seq', 'SHD']
method1_accuracies = test_accs_plots[:,0]  # Accuracies for method 1
method1_stds = test_acc_stds_plots[:,0]
method2_accuracies = test_accs_plots[:,1]  # Accuracies for method 2
method2_stds = test_acc_stds_plots[:,1]
custom_colors1 = ['cornflowerblue' for i in dataset_names]
custom_colors2 = ['orange' for i in dataset_names]
# custom_colors1 = ['orange' for i in dataset_names]
# custom_colors2 = ['cornflowerblue' for i in dataset_names]

# Set the width of the bars
bar_width = 0.25

# Create positions for the bars on the x-axis
x = range(len(dataset_names))

# Plot the bars for both networks
plt.bar(x, method1_accuracies,  width=bar_width, label='None', color = custom_colors1)
plt.errorbar(x, method1_accuracies, yerr=method1_stds, fmt='none', ecolor='black', capsize=5, capthick=0.7, elinewidth=0.7)
for i, acc in enumerate(method1_accuracies):
    plt.text(i, acc + 1.5, f"{acc:.1f}", ha="center", va="bottom", fontsize=9)
plt.bar([i + bar_width for i in x], method2_accuracies, width=bar_width, label='Ref', color = custom_colors2)
plt.errorbar([i + bar_width for i in x], method2_accuracies, yerr=method2_stds, fmt='none', ecolor='black', capsize=5, capthick=0.7, elinewidth=0.7)
for i, acc in enumerate(method2_accuracies):
    plt.text(i+ bar_width, acc + 1.5, f"{acc:.1f}", ha="center", va="bottom", fontsize=9)

# Customize the plot
# plt.xlabel('Datasets')
plt.ylabel('Accuracy (\%)')
plt.xticks([i + bar_width/2 for i in x], dataset_names)
plt.legend(loc='upper right', ncol=2, fontsize=9)
plt.grid(axis="y", linestyle="--", alpha=0.7)


# Show the plot
plt.ylim(0,110)
# plt.title('Rate Decoding')
plt.tight_layout()
fig.savefig('./plots/revision_rate.'+format, format=format)
# fig.savefig('./plots/Rate.eps', format='eps')

#*# Supplementary -- other methods
if SAVE == True:
    test_accs_plots = []
    test_acc_stds_plots = []
    dataset_names = ['MNIST_Sliced', 'FashionMNIST_Sliced'] #

    for dataset in dataset_names:
        path_name = './result_{}/'.format(dataset)
        if dataset == 'MNIST_Sliced':
            index_list = [2000, 2200, 2100, 2300, 3000, 3200, 3100, 3300]
        else:
            index_list = [2020, 2220, 2120, 2320, 3020, 3220, 3120, 3320]

        test_accs, test_acc_stds = get_result(path_name, index_list)
        test_accs_plots+=test_accs
        test_acc_stds_plots+=test_acc_stds
    test_accs_plots=np.array(test_accs_plots).reshape(-1,4).T
    test_acc_stds_plots=np.array(test_acc_stds_plots).reshape(-1,4).T
    np.savez('./plots/revision_comparison', test_accs_plots=test_accs_plots, test_acc_stds_plots=test_acc_stds_plots)
else:
    data = np.load('./plots/revision_comparison.npz')
    test_accs_plots = data['test_accs_plots']
    test_acc_stds_plots = data['test_acc_stds_plots']

# Draw raster plot of the outputs
fig=plt.figure()
fig.set_size_inches(6, 4)

spec = gridspec.GridSpec(ncols=2, nrows=2) # wspace=0.5, hspace=0.8, height_ratios=[1.2,1]
# Set the width of the bars
bar_width = 0.5
inter_bar = 0.1
inter_group = 1.5

method_names = ['None', 'Ref', 'Trainable', 'Adapt'] #'Adapt'
custom_colors = ['cornflowerblue', 'orange', 'mediumseagreen','mediumorchid'] # ,  

#ax1
ax1 = fig.add_subplot(spec[0,1])
dataset1_accuracies = test_accs_plots[:,0]  # Accuracies for method 1
dataset1_stds = test_acc_stds_plots[:,0]  # Accuracies for method 1
# Create positions for the bars on the x-axis
x1 = np.arange(len(method_names))
# Plot the bars for both networks
plt.bar(x1, dataset1_accuracies,  width=bar_width, label='Rate', color=custom_colors)
plt.errorbar(x1, dataset1_accuracies, yerr=dataset1_stds, fmt='none', ecolor='black', capsize=5, capthick=0.7, elinewidth=0.7)
for i, acc in enumerate(dataset1_accuracies):
    plt.text(x1[i], acc + 1.5, f"{acc:.1f}", ha="center", va="bottom", fontsize=9)


# plt.ylabel('Accuracy (\%)')
plt.xticks([i for i in x1], method_names)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax1.get_yticklabels(), visible=False)
# plt.legend(loc='upper right', ncol=2, fontsize=9)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.ylim(0,110)

plt.title('(b) MNIST_Sli, rate')

# ax2
ax2 = fig.add_subplot(spec[0,0])
dataset2_accuracies = test_accs_plots[:,1]  # Accuracies for method 2
dataset2_stds = test_acc_stds_plots[:,1]  # Accuracies for method 2
x2 = np.arange(len(method_names))

plt.bar(x2, dataset2_accuracies, width=bar_width, label='Temporal', color=custom_colors)
plt.errorbar(x2, dataset2_accuracies, yerr=dataset2_stds, fmt='none', ecolor='black', capsize=5, capthick=0.7, elinewidth=0.7)
for i, acc in enumerate(dataset2_accuracies):
    plt.text(x2[i], acc + 1.5, f"{acc:.1f}", ha="center", va="bottom", fontsize=9)

plt.ylabel('Accuracy (\%)')
plt.xticks([i for i in x1], method_names)
plt.setp(ax2.get_xticklabels(), visible=False)
# plt.setp(ax2.get_yticklabels(), visible=False)
# plt.legend(loc='upper right', ncol=2, fontsize=9)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.ylim(0,110)
plt.title('(a) MNIST_Sli, temporal')

# ax3
ax3 = fig.add_subplot(spec[1,1])
dataset3_accuracies = test_accs_plots[:,2]  # Accuracies for method 2
dataset3_stds = test_acc_stds_plots[:,2]  # Accuracies for method 2
x3 = np.arange(len(method_names))

plt.bar(x3, dataset3_accuracies, width=bar_width, label='Temporal', color=custom_colors)
plt.errorbar(x3, dataset3_accuracies, yerr=dataset3_stds, fmt='none', ecolor='black', capsize=5, capthick=0.7, elinewidth=0.7)
for i, acc in enumerate(dataset3_accuracies):
    plt.text(x3[i], acc + 1.5, f"{acc:.1f}", ha="center", va="bottom", fontsize=9)

# plt.ylabel('Accuracy (\%)')
plt.xticks([i for i in x1], method_names)
plt.setp(ax3.get_yticklabels(), visible=False)
# plt.setp(ax3.get_yticklabels(), visible=False)
# plt.legend(loc='upper right', ncol=2, fontsize=9)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.ylim(0,110)
plt.title('(d) Fashion_Sli, rate')

# ax4
ax4 = fig.add_subplot(spec[1,0])
dataset4_accuracies = test_accs_plots[:,3]  # Accuracies for method 2
dataset4_stds = test_acc_stds_plots[:,3]  # Accuracies for method 2
x4 = np.arange(len(method_names))

plt.bar(x4, dataset4_accuracies, width=bar_width, label='Temporal', color=custom_colors)
plt.errorbar(x4, dataset4_accuracies, yerr=dataset4_stds, fmt='none', ecolor='black', capsize=5, capthick=0.7, elinewidth=0.7)
for i, acc in enumerate(dataset4_accuracies):
    plt.text(x4[i], acc + 1.5, f"{acc:.1f}", ha="center", va="bottom", fontsize=9)

plt.ylabel('Accuracy (\%)')
plt.xticks([i for i in x4], method_names)
# plt.setp(ax4.get_yticklabels(), visible=False)
# plt.legend(loc='upper right', ncol=2, fontsize=9)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.ylim(0,110)
plt.title('(c) Fashion_Sli, temporal')

# fig.suptitle('MNIST_Sliced')
plt.tight_layout()
fig.savefig('./plots/revision_comparison.'+format, format=format)
# fig.savefig('./plots/Comparison.eps', format='eps')

#*# Grid Search of Adpt_thresh
beta_list = np.linspace(-2,0,10)
tau_list = np.linspace(0,2,10)
beta, tau = np.meshgrid(beta_list, tau_list)
paras = np.array([beta.reshape(-1), tau.reshape(-1)])
if SAVE == True:
    # The grid search
    path = './result_grid_search/'
    seed_list = range(1,6)
    accs_list =[]
    for seed in seed_list:
        file_name = path + 'grid_search_{}.npz'.format(seed)
        data = np.load(file_name)
        accs_list.append(data['val_accs'])

    accs = np.mean(np.array(accs_list), axis=0)
    accs = np.array(accs).reshape(beta.shape)*100
    np.savez('./plots/revision_adpt_grid', accs=accs)
else:
    data = np.load('./plots/revision_adpt_grid.npz')
    accs = data['accs']

fig=plt.figure()
fig.set_size_inches(5, 4)
clev = np.linspace(accs.min(),accs.max(),10) #Adjust the .001 to get finer gradient
Cp=plt.contourf(10**beta, 10**tau, accs, clev, ) #cmap='cool'
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\beta_{th}$')
plt.ylabel(r'$\tau_{th}$ (ms)')
fig.colorbar(Cp, label='Train Accuracy (\%)', ticks = [50, 55, 60])

plt.tight_layout()
plt.savefig('./plots/revision_adpt_grid.'+format, format=format)
# plt.savefig('./plots/Adpt_thresh.eps', format='eps')

#*# the control group of LIF with more hidden neurons

test_accs_plots = []
test_acc_stds_plots =[]
dataset_names = ['MNIST_Sliced', 'FashionMNIST_Sliced',]
if SAVE == True:
    for dataset in dataset_names:
        path_name = './result_{}/'.format(dataset)
        if dataset == 'MNIST_Sliced':
            index_list = [3000, 3400, 3200]
        else:
            index_list = [3020, 3420, 3220]
        test_accs, test_acc_stds = get_result(path_name, index_list)
        test_accs_plots+=test_accs
        test_acc_stds_plots+=test_acc_stds
    test_accs_plots=np.array(test_accs_plots).reshape(-1,3).T
    test_acc_stds_plots=np.array(test_acc_stds_plots).reshape(-1,3).T
    np.savez('./plots/revision_enlarged', test_accs_plots=test_accs_plots, test_acc_stds_plots=test_acc_stds_plots)
else:
    data = np.load('./plots/revision_enlarged.npz')
    test_accs_plots = data['test_accs_plots']
    test_acc_stds_plots = data['test_acc_stds_plots']

# Draw raster plot of the outputs
fig=plt.figure()
fig.set_size_inches(6, 3)

spec = gridspec.GridSpec(ncols=2, nrows=1) # wspace=0.5, hspace=0.8, height_ratios=[1.2,1]
# Set the width of the bars
bar_width = 0.5
inter_bar = 0.1
inter_group = 1.5

method_names = ['None', 'Enlarged', 'Ref'] #'Adapt'
custom_colors = ['cornflowerblue', 'pink', 'orange'] #

#ax0
ax0 = fig.add_subplot(spec[0,0])
dataset0_accuracies = test_accs_plots[:,0]  # Accuracies for method 1
dataset0_stds = test_acc_stds_plots[:,0]  # Accuracies for method 1
# Create positions for the bars on the x-axis
x0 = np.arange(len(method_names))
# Plot the bars for both networks
plt.bar(x0, dataset0_accuracies,  width=bar_width, label='Rate', color=custom_colors)
plt.errorbar(x0, dataset0_accuracies, yerr=dataset0_stds, fmt='none', ecolor='black', capsize=5, capthick=0.7, elinewidth=0.7)
for i, acc in enumerate(dataset0_accuracies):
    plt.text(x0[i], acc + 1.5, f"{acc:.1f}", ha="center", va="bottom", fontsize=9)


plt.ylabel('Accuracy (\%)')
plt.xticks([i for i in x0], method_names)
# plt.setp(ax0.get_xticklabels(), visible=False)
# plt.legend(loc='upper right', ncol=2, fontsize=9)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.ylim(0,110)

plt.title('(a) MNIST_Sli')

#ax1
ax1 = fig.add_subplot(spec[0,1])
dataset1_accuracies = test_accs_plots[:,1]  # Accuracies for method 1
dataset1_stds = test_acc_stds_plots[:,1]  # Accuracies for method 1
# Create positions for the bars on the x-axis
x1 = np.arange(len(method_names))
# Plot the bars for both networks
plt.bar(x1, dataset1_accuracies,  width=bar_width, label='Rate', color=custom_colors)
plt.errorbar(x1, dataset1_accuracies, yerr=dataset1_stds, fmt='none', ecolor='black', capsize=5, capthick=0.7, elinewidth=0.7)
for i, acc in enumerate(dataset1_accuracies):
    plt.text(x1[i], acc + 1.5, f"{acc:.1f}", ha="center", va="bottom", fontsize=9)

# plt.ylabel('Accuracy (\%)')
plt.xticks([i for i in x1], method_names)
plt.setp(ax1.get_yticklabels(), visible=False)
# plt.legend(loc='upper right', ncol=2, fontsize=9)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.ylim(0,110)

plt.title('(b) Fashion_Sli')
plt.savefig('./plots/revision_enlarged.'+format, format=format)

#*# the plot for saturation
def get_result_here(path_name,index_list):

    seed_list = [1,]
    
    train_acc_max_avg_list =[]
    test_acc_max_avg_list =[]
    test_acc_max_std_list =[]
    for index in index_list:
        train_acc_maxs = []
        test_acc_maxs = []
        for seed in seed_list:
            data = np.load(path_name+'result_{}.npz'.format(index+seed))
            train_accs = data['train_accs']
            test_accs = data['test_accs']
            test_acc_maxs.append(max(test_accs))
            train_acc_maxs.append(max(train_accs))
        train_acc_max_avg = np.mean(train_acc_maxs)
        test_acc_max_avg = np.mean(test_acc_maxs)
        train_acc_max_avg_list.append(train_acc_max_avg)
        test_acc_max_avg_list.append(test_acc_max_avg)
        test_acc_max_std_list.append(np.std(test_acc_maxs))
        # print(test_acc_maxs)
                        
    # print(train_acc_max_avg_list)
    # print(test_acc_max_avg_list)
    return test_acc_max_avg_list, test_acc_max_std_list
if SAVE == True:
    test_accs_plots = []
    test_acc_stds_plots =[]
    names = ['Saturation']

    for name in names:
        path_name = '/ifs/groups/cruzGrp/NNN_data/map_number_final/result_{}/'.format(name)
        index_list = [ 32005, 3210, 3220, 3240, 3260,3280, 32100]
        
        test_accs, test_acc_stds = get_result_here(path_name, index_list)
        test_accs_plots+=test_accs
        test_acc_stds_plots+=test_acc_stds
    np.savez('./plots/revision_saturation', test_accs_plots=test_accs_plots, test_acc_stds_plots=test_acc_stds_plots)
else:
    data = np.load('./plots/revision_saturation.npz')
    test_accs_plots = data['test_accs_plots']
    test_acc_stds_plots = data['test_acc_stds_plots']

fig=plt.figure()
fig.set_size_inches(5, 3)
test_accs_plots=np.array(test_accs_plots)
test_acc_stds_plots=np.array(test_acc_stds_plots)

plt.plot([5,10,20,40,60,80,100],test_accs_plots*100,'-o')
plt.yticks([92, 93, 94])
plt.xticks([20, 40, 60, 80,100])
plt.xlabel('$N_{ref}$')
plt.ylabel('Acurracy (\%)')
plt.tight_layout()
plt.savefig('./plots/revision_saturation.'+format, format=format)
