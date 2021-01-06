# import libraries
import os
import numpy as np
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_bvalue', type=int, default=4)
    parser.add_argument('--SNR', type=int, default=100)
    parser.add_argument('--data_path', type=str, default='/mnt/data003/wonil/IVIM_github/')

    args = parser.parse_args()
    path = args.data_path

    print('data_path:'+path)
    print('number of bvalue: '+str(args.num_bvalue))
    print('SNR: '+str(args.SNR))

    # parameters
    num_samples = 1000000
    num_samples_val = 10000
    fmin=0;     fmax=0.2
    Dmin=0;     Dmax=1.8e-3
    Dpmin=0;    Dpmax=20e-3
    S0min=200;    S0max=1700
    S0=300

    noise_std = S0/args.SNR
    num_bvalue = args.num_bvalue

    data_save_path = path+'bvalue_'+str(num_bvalue)+'_SNR_'+str(args.SNR)+'/'

    try:
        os.mkdir(data_save_path)
    except OSError:
        print ("Creation of the directory %s failed " % data_save_path)
    else:
        print ("Successfully created the directory %s " % data_save_path)

    # datagenerator
    Dp_train = np.zeros((num_samples))
    Dt_train = np.zeros((num_samples))
    Fp_train = np.zeros((num_samples))
    S0_train = np.zeros((num_samples))

    Dp_validation = np.zeros((num_samples_val))
    Dt_validation = np.zeros((num_samples_val))
    Fp_validation = np.zeros((num_samples_val))
    S0_validation = np.zeros((num_samples_val))

    for i in range(num_samples):
        Dp_train[i] = np.random.uniform(0, 1) * (Dpmax - Dpmin) + Dpmin
        Dt_train[i] = np.random.uniform(0, 1) * (Dmax - Dmin) + Dmin
        Fp_train[i] = np.random.uniform(0, 1) * (fmax - fmin) + fmin
        S0_train[i] = np.random.uniform(0, 1) * (S0max - S0min) + S0min

    for i in range(num_samples_val):
        Dp_validation[i] = np.random.uniform(0, 1) * (Dpmax - Dpmin) + Dpmin
        Dt_validation[i] = np.random.uniform(0, 1) * (Dmax - Dmin) + Dmin
        Fp_validation[i] = np.random.uniform(0, 1) * (fmax - fmin) + fmin
        S0_validation[i] = np.random.uniform(0, 1) * (S0max - S0min) + S0min

    noise_train_real = torch.normal(torch.tensor(np.zeros((num_samples, num_bvalue))), std=noise_std).float()
    noise_train_imag = torch.normal(torch.tensor(np.zeros((num_samples, num_bvalue))), std=noise_std).float()

    noise_validation_real = torch.normal(torch.tensor(np.zeros((num_samples_val, num_bvalue))), std=noise_std).float()
    noise_validation_imag = torch.normal(torch.tensor(np.zeros((num_samples_val, num_bvalue))), std=noise_std).float()

    inputdata = np.transpose(np.concatenate(([Fp_train], [Dt_train], [Dp_train], [S0_train]), axis=0))
    inputdata_val = np.transpose(
    np.concatenate(([Fp_validation], [Dt_validation], [Dp_validation], [S0_validation]), axis=0))


    ##Save noise & IVIM parameters
    torch.save(noise_train_real,data_save_path+'noise_train_real.pth')
    torch.save(noise_train_imag,data_save_path+'noise_train_imag.pth')

    torch.save(noise_validation_real,data_save_path+'noise_validation_real.pth')
    torch.save(noise_validation_imag, data_save_path + 'noise_validation_imag.pth')

    torch.save(inputdata,data_save_path+'inputdata.pth')
    torch.save(inputdata_val,data_save_path+'inputdata_val.pth')