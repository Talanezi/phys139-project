import numpy as np
import sys, os, time
import torch 
import torch.nn as nn
import data
import architecture
import optuna
import utils as U
from collections import OrderedDict
from captum.attr import Saliency,IntegratedGradients,GradientShap
import matplotlib.pyplot as plt
import shared_data as sd

def main():

    idx_im = 1200

    arch = 'o3' 
    suite = 'TNG' 
    sim_train       = 'WDM'

    # properties of the network/maps used for training
    fields_train    = ['Nbody'] #nbody
    #fields_train    = ['Vgas', 'Vcdm', 'Mtot'] #hydro_other
    fields_train    = ['Nbody', 'Mgas', 'Mstar', 'T', 'Z', 'P', 'HI', 'ne', 'MgFe'] #all
    #fields_train    = ['Mgas', 'Mstar', 'T', 'Z', 'P', 'HI', 'ne', 'MgFe'] #hydro
    #fields_train    = ['Nbody', 'Mtot', 'Mgas', 'P', 'HI', 'ne'] #good

    monopole_train  = True
    smoothing_train = 0
    z_train = 0.00

    # properties of the maps for testing
    sim_test       = sim_train  
    fields_test    = fields_train
    monopole_test  = True
    smoothing_test = 0
    label_test     = 'all_steps_500_500_{arch}_z{z_test}'
    mode           = 'test'

    # other parameters (usually no need to modify)
    root_files    = f'results/'
    root_maps     = f'images/' #folder containing the maps
    batch_size    = 128
    root_storage  = 'sqlite:///databases_%s_%s/%s'%(sim_train,suite,arch)
    z_test        = 0.00  #redshift
    seed          = 1               #random seed to initially mix the maps
    splits        = 15 #15              #number of maps per simulation
    just_monopole = False #whether to create the images with just the monopole
    sim_norm      = sim_train

    study_name  = f'wd_dr_hidden_lr_{arch}_smoothing{smoothing_train}_z{z_train}_all'
    label_train = f'all_steps_500_500_o3' 

    subfields_train = fields_train
    storage = U.fname_storage(root_storage, subfields_train, label_train, 
                              monopole=monopole_train)
    print("Found Storage")

    fmodel, num, dr, hidden, lr, wd = U.best_model_params(study_name, storage, 
                                                          subfields_train, root_files, 
                                                          label_train, 'new', 
                                                          sim_train, monopole_train, suite)

    print("Found Model")

    channels = len(fields_test)
    device = torch.device('cuda')
    model = architecture.get_architecture(arch+'_err', hidden, dr, channels)
    model = nn.DataParallel(model)
    model.to(device=device)

    print(fmodel)
    if os.path.exists(fmodel):  
        print('Loading model...')

        state_dict = torch.load(fmodel)
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.'+k
            new_state_dict[k]=v

        model.load_state_dict(new_state_dict)
    else:
        raise FileNotFoundError("Did not find model")

    f_params = 'WDM_TNG_params_1000.txt'
    params = np.loadtxt(f_params)
    minimum = np.array([0.1,  0.6,  0.25,   0.5,    0.25, 0.0625])
    maximum = np.array([0.5,  1.0,  4.00,   2.00,   4.00, 0.5555])
    params_maps = (params * (maximum - minimum)) + minimum

    print("Loading images...")   

    #normalize data
    im = np.zeros((len(fields_train), 256, 256))
    for i,field in enumerate(fields_train):

        print(f"    Loading {field}")

        f_maps = f"{root_maps}/Images_{field}_WDM_{suite}_z=0.00.npy"

        images = np.load(f_maps)

        if field == 'Mstar':
            images = np.log10(1 + images) 
        else:
            images = np.log10(images) 

        mean = np.mean(images)
        std = np.std(images)
        images = (images - mean) / std 

        im[i] = images[idx_im]

    print("Evaluating model...")

    results = params_maps[idx_im // splits]

    im = torch.tensor(im,dtype=torch.float).to(device=device) #.unsqueeze(0).to(device=device)
    input = im.unsqueeze(0)
    im = np.transpose(im.cpu().numpy(), (1,2,0))

    input.requires_grad = True

    model.eval()
    pred = model(input)
    
    predicted_value = pred.cpu().detach().numpy()[0]
    predicted_results = (predicted_value[:6]* (maximum - minimum)) + minimum
    predicted_errors = predicted_value[6:] * (maximum - minimum)

    t = 1/results[5]
    p = 1/predicted_results[5]
    e = predicted_errors[5] * p * p #0
    print("True", t, "Predicted", p, "Error", e)

    print("Calculating Saliencies")

    plots = []
    for mode in ['Saliency']: # ['Image', 'Saliency', 'Gradients', 'Shap']:

        if mode == 'Saliency':
            saliency = Saliency(model.forward)
            grads = saliency.attribute(input, target=5) 
        elif mode == 'Gradients':
            saliency = IntegratedGradients(model)
            grads = saliency.attribute(input, target=5) 
        elif mode == 'Shap':
            saliency = GradientShap(model.forward)
            baselines = torch.randn(20,len(fields_train),256,256)
            grads = saliency.attribute(input, target=0, baselines=baselines.cuda())
        else:
            plots.append([im[:,:,j] for j in range(len(fields_train))])
            continue

        grads = np.abs(grads.squeeze(0).cpu().detach().numpy())
        grads = np.transpose(grads, (1,2,0))

        mi = np.min(grads)
        ma = np.max(grads)
        grads -= mi 
        grads /=  ma - mi
        #grads = grads[:,:,0]
        grads /= np.max(grads)

        gcut = grads[:,:,:] > np.percentile(grads[:,:,:], 95)
        grads[~gcut] = 0
        grads[gcut] = 1

        #bounds on the image 
        #origin is top left corner
        #x is vertical, y is horizontal
        xl, xh, yl, yh = [0, 256, 0, 256]

        tmp_plots = []
        for j,field in enumerate(fields_train):

            fig, ax = plt.subplots()
            image = ax.imshow(im[xl:xh,yl:yh,j])
            im_data = image.cmap(image.norm(im[xl:xh,yl:yh,j]))
            plt.close(fig)

            gpixels = np.zeros(im_data.shape)
            gpixels[:,:,0] = grads[xl:xh,yl:yh,j]
            gpixels[:,:,3] = 1
            im_data[gcut[:,:,j]] =  gpixels[gcut[:,:,j]]

            tmp_plots.append(im_data)

        plots.append(tmp_plots)

    print("Ploting...")

    fig, ax = sd.set_plot_params(ncols=1, nrows=len(fields_train))
    if len(fields_train) == 1:
        ax = [ax]
    ax = [[axi] for axi in ax]

    for i in range(1):
        for j,field in enumerate(fields_train):
            ax[j][i].imshow(plots[i][j])
            ax[j][i].axis("off")
            ax[j][0].set_title(field)

    #ax[0][0].set_title(f"{fields_train[0]}; True: {t:.2f} Pred: {p:.2f} $\pm$ {e:.2f}")
    #ax[0][1].set_title("Saliency")
    #ax[0][2].set_title("Integrated Gradients")
    #ax[0][3].set_title("Shap")

    fig.savefig(f"../analysis/plots/Saliencies_good_{idx_im}.pdf", bbox_inches='tight')

    plt.close(fig)

    return

if __name__=='__main__':
    main()

