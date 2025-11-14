import numpy as np
import sys, os, time
import torch 
import torch.nn as nn
import data
import architecture
import optuna
import utils as U
from collections import OrderedDict

def get_pixels(res):
    if res <= 128:
        return 128
    if res <= 256:
        return 256
    if res <=512:
        return 512
    return None

def get_results(study_name, storage, arch, channels, device, root_files, test_loader,
                suite, suffix_train, label_train, subfields_train, monopole_train,
                suffix_test, label_test, subfields_test, monopole_test):

    ####### load best model ######## Training
    # get the parameters of the best model for the mean
    fmodel, num, dr, hidden, lr, wd = U.best_model_params(study_name, storage, 
                                                          subfields_train, root_files, 
                                                          label_train, 'new', 
                                                          sim_train, monopole_train, suite=suite)

    # get the model
    model = architecture.get_architecture(arch+'_err', hidden, dr, channels)
    model = nn.DataParallel(model)
    model.to(device=device)
    network_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters in the model = %d'%network_total_params)

    print(fmodel)

    # load best-model, if it exists
    if os.path.exists(fmodel):  
        print('Loading model...')

        state_dict = torch.load(fmodel) #, map_location=torch.device(device))
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.'+k
            #else:
            #    k = k.replace('features.module.', 'module.features.')
            new_state_dict[k]=v

        model.load_state_dict(new_state_dict)
    else:
        raise Exception('model doesnt exists!!!')
    ################################

    # get the name of the output files
    suffix1 = 'train_%s%s_%s'%(sim_train, suffix_train, label_train)
    if not(monopole_train):  suffix1 = '%s_no_monopole'%suffix1
    suffix2 = 'test_%s%s_%s'%(sim_test, suffix_test, label_test)
    if not(monopole_test):   suffix2 = '%s_no_monopole'%suffix2
    suffix = '%s_%s_%s.txt'%(suffix1, suffix2, suite)
    fresults  = f'{root_files}/results_{suffix}'
    fresults1 = f'{root_files}/Normalized_errors_{suffix}'

    # get the number of maps in the test set
    num_maps = U.dataloader_elements(test_loader)
    print('\nNumber of maps in the test set: %d'%num_maps)

    # define the arrays containing the value of the parameters
    #params_true = np.zeros((num_maps,6), dtype=np.float32)
    #params_NN   = np.zeros((num_maps,6), dtype=np.float32)
    #errors_NN   = np.zeros((num_maps,6), dtype=np.float32)
    #params_true = np.zeros((num_maps,1), dtype=np.float32)
    #params_NN   = np.zeros((num_maps), dtype=np.float32)
    #errors_NN   = np.zeros((num_maps), dtype=np.float32)
    params_true = np.zeros((num_maps,6), dtype=np.float32)
    params_NN   = np.zeros((num_maps,6), dtype=np.float32)
    errors_NN   = np.zeros((num_maps,6), dtype=np.float32)

    # get test loss
    g = [0, 1, 2, 3, 4, 5]
    g = [0, 1]
    g = [5] #5 = WDM
    test_loss1, test_loss2 = torch.zeros(len(g)).to(device), torch.zeros(len(g)).to(device)
    test_loss, points = 0.0, 0
    model.eval()
    for x, y in test_loader:
        #print(x.numpy().shape, [np.sum(el<1) for el in x.numpy()])
        with torch.no_grad():
            bs    = x.shape[0]    #batch size
            x     = x.to(device)  #send data to device
            y     = y.to(device)  #send data to device
            p     = model(x)      #prediction for mean and variance
            y_NN  = p[:,:6]       #prediction for mean
            e_NN  = p[:,6:]       #prediction for error
            #y_NN  = p[:,1::2]       #prediction for mean
            #e_NN  = p[:,::2]       #prediction for error
            loss1 = torch.mean((y_NN[:,5] - y[:,5])**2,                     axis=0)
            loss2 = torch.mean(((y_NN[:,5] - y[:,5])**2 - e_NN[:,5]**2)**2, axis=0)
            #loss2 = torch.mean(((y_NN[:,5] - y[:,5])**2 - e_NN[:,0]**2)**2, axis=0)
            #loss1 = torch.mean((y_NN[:] - y[:])**2)
            #loss2 = torch.mean(((y_NN[:] - y[:])**2 - e_NN[:]**2)**2)
            test_loss1 += loss1*bs
            test_loss2 += loss2*bs

            # save results to their corresponding arrays
            params_true[points:points+x.shape[0]] = y.cpu().numpy() 
            params_NN[points:points+x.shape[0]]   = y_NN.cpu().numpy()
            errors_NN[points:points+x.shape[0]]   = e_NN.cpu().numpy()
            points    += x.shape[0]
    test_loss = torch.log(test_loss1/points) + torch.log(test_loss2/points)
    test_loss = torch.mean(test_loss).item()
    print('Test loss = %.3e\n'%test_loss)

    Norm_error = np.sqrt(np.mean((params_true - params_NN)**2, axis=0))
    #print('Normalized Error WDM_m = %.3f'%Norm_error[0])
    #print('Normalized Error Omega_m = %.3f'%Norm_error[0])
    #print('Normalized Error sigma_8 = %.3f'%Norm_error[1])
    #print('Normalized Error A_SN1   = %.3f'%Norm_error[2])
    #print('Normalized Error A_AGN1  = %.3f'%Norm_error[3])
    #print('Normalized Error A_SN2   = %.3f'%Norm_error[4])
    print('Normalized Error WDM  = %.3f\n'%Norm_error[5])

    # de-normalize
    #minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
    #maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
    #minimum = np.array([0.0333333])
    #maximum = np.array([0.4])
    minimum = np.array([0.1,  0.6,  0.25,   0.5,    0.25, 0.0625])
    maximum = np.array([0.5,  1.0,  4.00,   2.00,   4.00, 0.5555])
    #params_true = params_true*(maximum - minimum) + minimum
    #params_NN   = params_NN*(maximum - minimum) + minimum
    #errors_NN   = errors_NN*(maximum - minimum)

    error = np.sqrt(np.mean((params_true - params_NN)**2, axis=0))
    #print('Error WDM_m = %.3f'%error[0])
    #print('Error Omega_m = %.3f'%error[0])
    #print('Error sigma_8 = %.3f'%error[1])
    #print('Error A_SN1   = %.3f'%error[2])
    #print('Error A_AGN1  = %.3f'%error[3])
    #print('Error A_SN2   = %.3f'%error[4])
    print('Error WDM  = %.3f\n'%error[5])

    mean_error = np.absolute(np.mean(errors_NN, axis=0))
    #print('Bayesian error WDM_m = %.3f'%mean_error)
    #print('Bayesian error Omega_m = %.3f'%mean_error[0])
    #print('Bayesian error sigma_8 = %.3f'%mean_error[1])
    #print('Bayesian error A_SN1   = %.3f'%mean_error[2])
    #print('Bayesian error A_AGN1  = %.3f'%mean_error[3])
    #print('Bayesian error A_SN2   = %.3f'%mean_error[4])
    print('Bayesian error WDM  = %.3f\n'%mean_error[5])

    rel_error = np.sqrt(np.mean((params_true - params_NN)**2/params_true**2, axis=0))
    #print('Relative error WDM_m = %.3f'%rel_error[0])
    #print('Relative error Omega_m = %.3f'%rel_error[0])
    #print('Relative error sigma_8 = %.3f'%rel_error[1])
    #print('Relative error A_SN1   = %.3f'%rel_error[2])
    #print('Relative error A_AGN1  = %.3f'%rel_error[3])
    #print('Relative error A_SN2   = %.3f'%rel_error[4])
    print('Relative error WDM  = %.3f\n'%rel_error[5])

    # save results to file
    #dataset = np.zeros((num_maps,18), dtype=np.float32)
    #dataset[:,:6]   = params_true
    #dataset[:,6:12] = params_NN
    #dataset[:,12:]  = errors_NN
    dataset = np.zeros((num_maps,18), dtype=np.float32)
    dataset[:,:6]   = params_true
    dataset[:,6:12]   = params_NN
    dataset[:,12:]   = errors_NN
    np.savetxt(fresults,  dataset)
    np.savetxt(fresults1, Norm_error)

    print(fresults)


##################################### INPUT ##########################################
# architecture parameters

suite = 'TNG'
arch  = 'o3' #'o3_512' # 'o3'

# properties of the network/maps used for training
#sim_train       = 'WDM_Full'
sim_train       = 'WDM'

fields_train    = ['HI'] #, 'Vcdm']
#fields_train    = ['Nbody', 'Mgas', 'Mstar', 'T', 'Z', 'P', 'HI', 'ne', 'MgFe']
#fields_train    = ['Mgas', 'Mstar', 'T', 'Z', 'P', 'HI', 'ne', 'MgFe']
#fields_train    = ['Vgas', 'Vcdm', 'Mtot'] #hydro_other
#fields_train    = ['Nbody', 'Mtot', 'Mgas', 'P', 'HI', 'ne'] #good
#fields_train    = ['Mstar', 'T', 'Z', 'MgFe', 'Vgas', 'Vcdm'] #bad 

monopole_train  = True
smoothing_train = 0
z_train = 0.00

# properties of the maps for testing
#sim_test       = 'WDM_Full' 
sim_test       = 'WDM' #'WDM-test'  

#fields_test    = ['Nbody'] #, 'Vcdm']
#fields_test    = ['Nbody', 'Mgas', 'Mstar', 'T', 'Z', 'P', 'HI', 'ne', 'MgFe']
#fields_test     = ['Mgas', 'Mstar', 'T', 'Z', 'P', 'HI', 'ne', 'MgFe']
#fields_test    = ['Vgas', 'Vcdm', 'Mtot'] #hydro_other
fields_test = fields_train

monopole_test  = True
smoothing_test = 0
mode           = 'test' # 'all' # 'test'

# other parameters (usually no need to modify)
root_files      = f'results/' #output folder
root_maps     = f'images/' #folder with the maps
batch_size    = 128 #32
root_storage  = 'sqlite:///databases_%s_%s/%s'%(sim_train,suite,arch)
z_test        = 0.00  #redshift
seed          = 1               #random seed to initially mix the maps
splits        = 15 #51 #15              #number of maps per simulation
just_monopole = False #whether to create the images with just the monopole
sim_norm      = sim_train
label_test     = f'all_steps_500_500_{arch}_z{z_test}'


label_train   = f'all_steps_500_500_o3'
study_name = f'wd_dr_hidden_lr_{arch}_smoothing{smoothing_train}_z{z_train}_HI'


# data parameters
#['Comb', 'Mgas', 'Mcdm', 'Mtot', 'Mstar', 'Vgas', 'Vcdm',
#                'T', 'Z', 'HI', 'P', 'ne', 'B', 'MgFe',
#                'Nbody', 'Comb']
######################################################################################

#if smoothing_train>0:  label_train = '%s_smoothing_%d'%(label_train,smoothing_train)
#if smoothing_test>0:   label_test  = '%s_smoothing_%d'%(label_test, smoothing_train)

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# define loss function
criterion = nn.MSELoss() 

# get the test mode and the file with the parameters
#if suite=='LH':    
#    f_params = '/mnt/ceph/users/camels/Software/%s/latin_hypercube_params.txt'%sim_test
if suite=='CV': 
    f_params = '/mnt/ceph/users/camels/Software/%s/CV_params.txt'%sim_test
else:
    f_params = 'WDM_TNG_params_1000.txt'

#else:          raise Exception('wrong suite value')

# do a loop over the different fields
for field_train, field_test in zip(fields_train, fields_test):

    print('\n############# Trained on %s %s ##################'%(field_train,sim_train))
    print('############# Testing on %s %s ##################'%(field_test,sim_test))

    ############ Training files #############
    # get the subfields of the considered field
    if field_train=='Comb':  subfields_train = ['Mgas', 'Mtot', 'Mstar', 'Vgas', 'T',
                                                'Z', 'P', 'HI', 'ne', 'B', 'MgFe']
    elif len(fields_train) > 1:
        #subfields_train = ['Nbody', 'Vcdm'] #JONAH EDIT?
        subfields_train = fields_train
    else:                    subfields_train = [field_train]

    # get the suffix
    suffix_train = ''
    for x in subfields_train:  suffix_train = '%s_%s'%(suffix_train,x)

    # get the storage name to use best-model (training)
    storage = U.fname_storage(root_storage, subfields_train, label_train, 
                              monopole=monopole_train)
    print("Storage", storage)
    #########################################

    ########## Testing files ################
    # get the subfields of the considered field
    if field_test=='Comb':   subfields_test = ['Mgas', 'Mtot', 'Mstar', 'Vgas', 'T', 
                                               'Z', 'P', 'HI', 'ne', 'B', 'MgFe']
    if len(fields_test) > 1:
        #subfields_test = ['Nbody','Vcdm']
        subfields_test = fields_test
    else:                    subfields_test = [field_test]

    print(subfields_train, subfields_test)

    # get the name of the maps to test the model
    channels = len(subfields_test)
    f_maps, f_maps_norm, suffix_test = U.fname_maps(root_maps, subfields_test, sim_test,
                                                    z_test, suite, sim_norm)

    print(f_maps_norm, f_maps)

    #f_maps = [f"{root_maps}/Images_Nbody_WDM_LH_low{low}_lownorm_z=0.00.npy"]
    #f_maps = [f"{root_maps}/Images_Nbody_WDM_LH_high{high}_highnorm_z=0.00.npy"]
    #f_maps = [f"{root_maps}/Images_Nbody_WDM_LH_low{low}_high{high}_midnorm_z=0.00.npy"]
    #f_maps = [f"{root_maps}/Images_Nbody_WDM_LH_ngb{ngb}_z=0.00.npy"] #JONAH EDIT
    #f_maps = [f"{root_maps}/Images_Nbody_WDM_LH_test_z=0.00.npy"]
    #f_maps = [f"{root_maps}/Images_Nbody_WDM_LH_test_voxngb{ngb}_z=0.00.npy"]
    print(f_maps)
    # get the test data
    test_loader = data.create_dataset_multifield(mode, seed, f_maps, f_params, 
                            batch_size, splits, f_maps_norm, monopole=monopole_test, 
                            monopole_norm=monopole_train, rot_flip_in_mem=True, 
                            shuffle=False, just_monopole=just_monopole, 
                            smoothing=smoothing_test, smoothing_norm=smoothing_train, 
                            verbose=True)
    #########################################

    # get the results
    
    get_results(study_name, storage, arch, channels, device, root_files, test_loader,
                suite, suffix_train, label_train, subfields_train, monopole_train,
                suffix_test, label_test, subfields_test, monopole_test)
                
    print('############################################')











