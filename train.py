import numpy as np
import sys, os, time
import torch 
import torch.nn as nn
import data
import architecture
import torch.backends.cudnn as cudnn
import optuna
import utils as U

class Objective(object):
    def __init__(self, device, seed, f_maps, f_params, batch_size, splits, f_maps_norm, 
                 arch, min_lr, beta1, beta2, epochs, root_out, fields, monopole, sname, 
                 label, num_workers, params, rot_flip_in_mem, sim, smoothing):

        self.device          = device
        self.seed            = seed
        self.f_maps          = f_maps
        self.f_maps_norm     = f_maps_norm
        self.f_params        = f_params
        self.batch_size      = batch_size
        self.splits          = splits
        self.arch            = arch
        self.min_lr          = min_lr
        self.beta1           = beta1
        self.beta2           = beta2
        self.epochs          = epochs
        self.root_out        = root_out
        self.fields          = fields
        self.monopole        = monopole
        self.label           = label
        self.num_workers     = num_workers
        self.params          = params
        self.rot_flip_in_mem = rot_flip_in_mem
        self.sim             = sim
        self.smoothing       = smoothing
        self.sname           = sname

    def __call__(self, trial):

        # get the number of channels in the maps
        channels  = len(self.fields)
        criterion = nn.MSELoss() 

        # tuple with the indexes of the parameters to train
        g = self.params      #posterior mean
        #h = [1+i for i in g] #posterior variance
        h = [6+i for i in g]

        # get the value of the hyperparameters
        max_lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        wd     = trial.suggest_float("wd", 1e-8, 1e-1, log=True)
        dr     = trial.suggest_float("dr", 0.0,  0.9)
        hidden = trial.suggest_int("hidden", 6, 12)

        # optimum hyperparams found by Jonah with Optuna
        #max_lr = 0.003081406540122628
        #wd     = 0.0012352939003943669
        #dr     = 0.4044701879352993
        #hidden = 11


        # some verbose
        print('\nTrial number: {}'.format(trial.number))
        print('Fields: {}'.format(self.fields))
        print('lr: {}'.format(max_lr))
        print('wd: {}'.format(wd))
        print('dr: {}'.format(dr))
        print('hidden: {}'.format(hidden))


        # get the name of the files
        if self.monopole:             suffix = '%d'%trial.number
        else:                         suffix = 'no_monopole_%d'%trial.number
        if self.label is not None:    suffix = '%s_%s_%s'%(suffix,label,self.sname)

        fout   = '%s/losses_%s_%s/loss'%(self.root_out, self.sim, suite)
        fmodel = '%s/models_%s_%s/model'%(self.root_out, self.sim, suite)
        for field in self.fields:
            fout   = '%s_%s'%(fout,   field)
            fmodel = '%s_%s'%(fmodel, field)
        
        fout   = '%s_%s.txt'%(fout, suffix)
        fmodel = '%s_%s.pt'%(fmodel, suffix)

        print('fmodel:', fmodel)

        # get train, validation and test sets
        print('\nPreparing datasets...')
        train_loader = data.create_dataset_multifield('train', self.seed, self.f_maps, 
                    self.f_params, self.batch_size, self.splits, self.f_maps_norm, 
                    num_workers=self.num_workers, monopole=self.monopole, 
                    rot_flip_in_mem=self.rot_flip_in_mem, smoothing=self.smoothing, 
                    verbose=True)
        valid_loader = data.create_dataset_multifield('valid', self.seed, self.f_maps, 
                    self.f_params, self.batch_size, self.splits, self.f_maps_norm, 
                    num_workers=self.num_workers, monopole=self.monopole, 
                    rot_flip_in_mem=True, smoothing=self.smoothing, verbose=True)
        test_loader  = data.create_dataset_multifield('test',  self.seed, self.f_maps, 
                    self.f_params, self.batch_size, self.splits, self.f_maps_norm, 
                    num_workers=self.num_workers, monopole=self.monopole, 
                    rot_flip_in_mem=True, smoothing=self.smoothing, verbose=True)
    

        # define architecture
        model = architecture.get_architecture(self.arch+'_err', hidden, dr, channels)
        if torch.cuda.device_count() > 1:
            print("Using %d GPUs"%(torch.cuda.device_count()))
            model = nn.DataParallel(model)

        model.to(device)
        network_total_params = sum(p.numel() for p in model.parameters())
        print('total number of parameters in the model = %d'%network_total_params)

        # define optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=wd,
                                      betas=(self.beta1, self.beta2))
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.min_lr, 
                                max_lr=max_lr, cycle_momentum=False, step_size_up=500,
                                                      step_size_down=500)

        # get validation loss
        print('Computing initial validation loss')
        model.eval()
        valid_loss1, valid_loss2 = torch.zeros(len(g)).to(device), torch.zeros(len(g)).to(device)
        min_valid_loss, points = 0.0, 0
        for x, y in valid_loader:
            with torch.no_grad():
                bs   = x.shape[0]                #batch size
                x    = x.to(device=device)       #maps
                y    = y.to(device=device)[:,g]  #parameters
                p    = model(x)                  #NN output
                y_NN = p[:,g]                    #posterior mean
                e_NN = p[:,h]                    #posterior std
                loss1 = torch.mean((y_NN - y)**2,                axis=0)
                loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
                loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
                valid_loss1 += loss1*bs
                valid_loss2 += loss2*bs
                points += bs
        min_valid_loss = torch.log(valid_loss1/points) + torch.log(valid_loss2/points)
        min_valid_loss = torch.mean(min_valid_loss).item()
        print('Initial valid loss = %.3e'%min_valid_loss)

        old_p = []

        # do a loop over all epochs
        start = time.time()
        for epoch in range(self.epochs):
        
            # do training
            train_loss1, train_loss2 = torch.zeros(len(g)).to(device), torch.zeros(len(g)).to(device)
            train_loss, points = 0.0, 0
            model.train()
            for x, y in train_loader:
                bs   = x.shape[0]         #batch size
                x    = x.to(device)       #maps
                y    = y.to(device)[:,g]  #parameters
                p    = model(x)           #NN output
                y_NN = p[:,g]             #posterior mean
                e_NN = p[:,h]             #posterior std
                loss1 = torch.mean((y_NN - y)**2,                axis=0)
                loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
                loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
                train_loss1 += loss1*bs
                train_loss2 += loss2*bs
                points      += bs
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if points>18000:  break
            train_loss = torch.log(train_loss1/points) + torch.log(train_loss2/points)
            train_loss = torch.mean(train_loss).item()

            # do validation: cosmo alone & all params
            valid_loss1, valid_loss2 = torch.zeros(len(g)).to(device), torch.zeros(len(g)).to(device)
            valid_loss, points = 0.0, 0
            model.eval()
            for x, y in valid_loader:
                with torch.no_grad():
                    bs    = x.shape[0]        #batch size
                    x     = x.to(device)       #maps
                    y     = y.to(device)[:,g]  #parameters
                    p     = model(x)           #NN output
                    y_NN  = p[:,g]             #posterior mean
                    e_NN  = p[:,h]             #posterior std
                    loss1 = torch.mean((y_NN - y)**2,                axis=0)
                    loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
                    loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
                    valid_loss1 += loss1*bs
                    valid_loss2 += loss2*bs
                    points     += bs
            valid_loss = torch.log(valid_loss1/points) + torch.log(valid_loss2/points)
            valid_loss = torch.mean(valid_loss).item()

            # do testing
            test_loss1, test_loss2 = torch.zeros(len(g)).to(device), torch.zeros(len(g)).to(device)
            test_loss, points = 0.0, 0
            model.eval()
            for x, y in test_loader:
                with torch.no_grad():
                    bs    = x.shape[0]         #batch size
                    x     = x.to(device)       #maps
                    y     = y.to(device)[:,g]  #parameters
                    p     = model(x)           #NN output
                    y_NN  = p[:,g]             #posterior mean
                    e_NN  = p[:,h]             #posterior std
                    loss1 = torch.mean((y_NN - y)**2,                axis=0)
                    loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
                    loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
                    test_loss1 += loss1*bs
                    test_loss2 += loss2*bs
                    points    += bs
            test_loss = torch.log(test_loss1/points) + torch.log(test_loss2/points)
            test_loss = torch.mean(test_loss).item()

            # verbose
            print('%03d %.3e %.3e %.3e '%(epoch, train_loss, valid_loss, 
                                          test_loss), end='')

            # save model if it is better
            if valid_loss<min_valid_loss:
                torch.save(model.state_dict(), fmodel)
                min_valid_loss = valid_loss
                print('(C) ', end='')
            print('')

            # save losses to file
            f = open(fout, 'a')
            f.write('%d %.5e %.5e %.5e\n'%(epoch, train_loss, valid_loss, test_loss))
            f.close()
    
            # Handle pruning based on the intermediate value
            # comment out these lines if using prunning
            #trial.report(valid_loss, epoch)
            #if trial.should_prune():  raise optuna.exceptions.TrialPruned()

        stop = time.time()
        print('Time take (h):', "{:.4f}".format((stop-start)/3600.0))

        return min_valid_loss


##################################### INPUT ##########################################
# architecture parameters

# simulation suite
sim = 'WDM'
suite = 'TNG'

arch  = 'o3'
beta1 = 0.5
beta2 = 0.999

# data parameters
root_out        = f'results/' #output folder
root_maps       = f'images/' #folder with the maps
root_storage    = 'sqlite:///databases_%s_%s/%s'%(sim,suite,arch)
f_params        = f'WDM_TNG_params_1000.txt'
seed            = 1               #random seed to initially mix the maps
splits          = 15 #51 #15             #number of maps per simulation
z               = 0.00            #redshift
monopole        = True  #keep the monopole of the maps (True) or remove it (False)
#fields          = ['Nbody']  #nbody, om_Nbody
#fields          = ['Mgas', 'Mstar', 'T', 'Z', 'P', 'HI', 'ne', 'MgFe'] #hydro
#fields          = ['Nbody','Mgas', 'Mstar', 'T', 'Z', 'P', 'HI', 'ne', 'MgFe'] #All
#fields           = ['Vgas', 'Vcdm', 'Mtot'] #hydro_other
#fields          = ['Nbody', 'Mtot', 'Mgas', 'P', 'HI', 'ne'] #good, good_long
#fields          = ['Mstar', 'T', 'Z', 'MgFe', 'Vgas', 'Vcdm'] #bad 
fields          = ['B']
params          = [5]
rot_flip_in_mem = False  #whether rotations and flipings are kept in memory
smoothing       = 0 #Gaussian smoothing in pixels units
label           = f'all_steps_500_500_o3'

# training parameters
batch_size  = 128 #128
min_lr      = 1e-9
epochs      = 200 # increase for grokking? Ask Paco?
num_workers = 4  #number of workers to load data

# optuna parameters
#nbody, hydro, all
study_name = f'wd_dr_hidden_lr_{arch}_smoothing{smoothing}_z{z}_B'
n_trials   = 50
######################################################################################

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')
cudnn.benchmark = True      #May train faster but cost more memory


# get the name of files containing the maps and the databases
f_maps, f_maps_norm, suffix = U.fname_maps(root_maps, fields, sim, z, suite=suite)
storage_m = U.fname_storage(root_storage, fields, label, monopole=monopole)

print("Storage:", storage_m)
print(f_maps, f_maps_norm)

# train networks with bayesian optimization
objective = Objective(device, seed, f_maps, f_params, batch_size, splits, f_maps_norm, 
                      arch, min_lr, beta1, beta2, epochs, root_out, fields, monopole, study_name, 
                      label, num_workers, params, rot_flip_in_mem, sim, smoothing)

sampler = optuna.samplers.TPESampler(n_startup_trials=20)
study = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage_m,
                            load_if_exists=True)
study.optimize(objective, n_trials, gc_after_trial=False)

