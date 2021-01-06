import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
import torch
from torch.autograd import Variable
from model_ivim import Net
import argparse

# parameters
num_samples = 1000000
num_samples_val = 10000
fmin = 0;fmax = 0.2
Dmin = 0;Dmax = 1.8e-3
Dpmin = 0;Dpmax = 20e-3
S0min = 200;S0max = 1700

batch_size = 1000
num_epoch = 200
best_loss = 10 ** 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_bvalue', type=int, default=4)
    parser.add_argument('--SNR', type=int, default=100)
    parser.add_argument('--num_trials', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=1.0)
    parser.add_argument('--lambda3', type=float, default=1.0)
    parser.add_argument('--data_path', type=str, default='/home/user/')
    args = parser.parse_args()

    lambda1 = args.lambda1
    lambda2 = args.lambda2
    lambda3 = args.lambda3

    # GPU
    GPU_NUM = args.gpu # GPU number
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    print('data path: '+args.data_path)
    print('number of bvalue: '+str(args.num_bvalue))
    print('SNR: '+str(args.SNR))
    print ('Current cuda device ', torch.cuda.current_device()) #

    num_bvalue = args.num_bvalue
    num_trials = args.num_trials

    load_data_path = args.data_path + '/bvalue_' + str(num_bvalue) + '_SNR_' + str(args.SNR) + '/'
    net_save_path = load_data_path +'l1_'+str(lambda1)+'_l2_'+str(lambda2)+'_l3_'+str(lambda3) +'/'

    noise_train_real = torch.load(load_data_path + 'noise_train_real.pth')
    noise_train_imag = torch.load(load_data_path + 'noise_train_imag.pth')
    noise_validation_real = torch.load(load_data_path + 'noise_validation_real.pth')
    noise_validation_imag = torch.load(load_data_path + 'noise_validation_imag.pth')
    inputdata = torch.load(load_data_path + 'inputdata.pth')
    inputdata_val = torch.load(load_data_path + 'inputdata_val.pth')

    try:
        os.mkdir(net_save_path)
    except OSError:
        print ("Creation of the directory %s failed " % net_save_path)
    else:
        print ("Successfully created the directory %s " % net_save_path)

    # dataloader
    num_batches = len(inputdata) // batch_size
    trainloader = utils.DataLoader(torch.from_numpy(inputdata.astype(np.float32)),
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True)
    validationloader = utils.DataLoader(torch.from_numpy(inputdata_val.astype(np.float32)),
                                        batch_size=100,
                                        shuffle=True,
                                        drop_last=True)

    # training DNN
    for trials in range(num_trials):
        net = Net(num_bvalue)
        net = net.to(device)
        criterion = nn.MSELoss()
        initial_bvalue_sigmoid = torch.rand(num_bvalue)                                 # random initialization
        initial_bvalue = np.log(initial_bvalue_sigmoid / (1 - initial_bvalue_sigmoid))      # inverse of sigmoid
        bvalue = initial_bvalue.type(torch.FloatTensor).to(device)
        bvalue = Variable(bvalue, requires_grad=True)                                   # trainable b-values

        train_loss = []
        validation_loss = []
        bvalue_learning = []

        optimizer_net = optim.Adam(net.parameters(), lr=0.003)
        optimizer_bvalues = optim.Adam([bvalue], lr=0.003)

        for epoch in range(num_epoch):
            print("Trials: {}".format(trials))
            print("-----------------------------------------------------------------")
            print("Epoch: {}".format(epoch))
            running_loss = 0.
            train_batch_loss = []
            val_batch_loss = []

            for i, X_batch in enumerate((trainloader), 0):
                # zero the parameter gradients
                optimizer_net.zero_grad()
                optimizer_bvalues.zero_grad()
                X_batch = X_batch.to(device)
                # network input
                noise_real = noise_train_real[i * X_batch.size(0):(i + 1) * X_batch.size(0)].to(device)
                noise_imag = noise_train_imag[i * X_batch.size(0):(i + 1) * X_batch.size(0)].to(device)
                Signal, b, Output, f, D, Dp, Signal_pred = net(X_batch, bvalue, 0, 'train', noise_real, noise_imag)

                loss1 = criterion(Output[:, 0], (X_batch[:, 0] - fmin) / (fmax - fmin))
                loss2 = criterion(Output[:, 1], (X_batch[:, 1] - Dmin) / (Dmax - Dmin))
                loss3 = criterion(Output[:, 2], (X_batch[:, 2] - Dpmin) / (Dpmax - Dpmin))

                loss = lambda1*loss1 + lambda2*loss2 + lambda3*loss3
                loss.backward()

                optimizer_net.step()
                optimizer_bvalues.step()
                running_loss += loss.item()
                train_batch_loss.append(loss.item())

            train_loss.append(np.mean(train_batch_loss))
            print(b)
            print("Loss: {}".format(train_loss[-1]))
            with torch.no_grad():
                for i, X_batch_val in enumerate((validationloader), 0):
                    # forward + backward + optimize
                    X_batch_val = X_batch_val.to(device)
                    noise_real = noise_validation_real[i * X_batch_val.size(0):(i + 1) * X_batch_val.size(0)].to(device)
                    noise_imag = noise_validation_imag[i * X_batch_val.size(0):(i + 1) * X_batch_val.size(0)].to(device)
                    Signal, b, Output_val, f, D, Dp, Signal_pred = net(X_batch_val, bvalue, 0, 'validation', noise_real, noise_imag)

                    loss1 = criterion(Output_val[:, 0], (X_batch_val[:, 0] - fmin) / (fmax - fmin))
                    loss2 = criterion(Output_val[:, 1], (X_batch_val[:, 1] - Dmin) / (Dmax - Dmin))
                    loss3 = criterion(Output_val[:, 2], (X_batch_val[:, 2] - Dpmin) / (Dpmax - Dpmin))

                    loss = lambda1*loss1 + lambda2*loss2 + lambda3*loss3
                    val_batch_loss.append(loss.item())

                    if (epoch == 0) & (i == 0):  # save initial b-value
                        bvalue_learning = np.array(b.detach().cpu())

                validation_loss.append(np.mean(val_batch_loss))

            print("Validation Loss: {}".format(validation_loss[-1]))
            rmse_f = np.sqrt(np.mean((Output_val.detach().cpu().numpy()[:, 0] * (
                        fmax - fmin) + fmin - X_batch_val.detach().cpu().numpy()[:, 0]) ** 2))
            avg_f = np.mean(X_batch_val.detach().cpu().numpy()[:, 0])
            rmse_D = np.sqrt(np.mean((Output_val.detach().cpu().numpy()[:, 1] * (
                        Dmax - Dmin) + Dmin - X_batch_val.detach().cpu().numpy()[:, 1]) ** 2))
            avg_D = np.mean(X_batch_val.detach().cpu().numpy()[:, 1])
            rmse_Dp = np.sqrt(np.mean((Output_val.detach().cpu().numpy()[:, 2] * (
                        Dpmax - Dpmin) + Dpmin - X_batch_val.detach().cpu().numpy()[:, 2]) ** 2))
            avg_Dp = np.mean(X_batch_val.detach().cpu().numpy()[:, 2])

            print('NRMSE of f: ' + str(rmse_f / avg_f))
            print('NRMSE of D: ' + str(rmse_D / avg_D))
            print('NRMSE of Dp: ' + str(rmse_Dp / avg_Dp))
            print('number of bvalue: ' + str(args.num_bvalue))
            print('SNR: ' + str(args.SNR))
            print('Current cuda device ', torch.cuda.current_device())
            bvalue_learning = np.concatenate((bvalue_learning, np.array(b.detach().cpu())), 0)

            #====================SAVE=======================
            if validation_loss[epoch] < best_loss:
                best_trials = trials
                best_loss = validation_loss[epoch]
                final_b = b
                if os.path.isfile(net_save_path + 'validation_loss.pth'):
                    os.unlink(net_save_path + 'validation_loss.pth')
                if os.path.isfile(net_save_path + 'train_loss.pth'):
                    os.unlink(net_save_path + 'train_loss.pth')
                if os.path.isfile(net_save_path + 'final_net.pth'):
                    os.unlink(net_save_path + 'final_net.pth')
                if os.path.isfile(net_save_path + 'final_b.pth'):
                    os.unlink(net_save_path + 'final_b.pth')
                if os.path.isfile(net_save_path + 'bvalue_learning.pth'):
                    os.unlink(net_save_path + 'bvalue_learning.pth')
                torch.save(validation_loss, net_save_path + 'validation_loss.pth')
                torch.save(train_loss, net_save_path + 'train_loss.pth')
                torch.save(net, net_save_path +'final_net.pth')
                torch.save(final_b, net_save_path +'final_b.pth')
                torch.save(bvalue_learning, net_save_path + 'bvalue_learning.pth')
                print('network saved!')

        if best_trials == trials:
            final_b = b
            if os.path.isfile(net_save_path + 'validation_loss.pth'):
                os.unlink(net_save_path + 'validation_loss.pth')
            if os.path.isfile(net_save_path + 'train_loss.pth'):
                os.unlink(net_save_path + 'train_loss.pth')
            if os.path.isfile(net_save_path + 'final_net.pth'):
                os.unlink(net_save_path + 'final_net.pth')
            if os.path.isfile(net_save_path + 'final_b.pth'):
                os.unlink(net_save_path + 'final_b.pth')
            if os.path.isfile(net_save_path + 'bvalue_learning.pth'):
                os.unlink(net_save_path + 'bvalue_learning.pth')
            torch.save(validation_loss, net_save_path + 'validation_loss.pth')
            torch.save(train_loss, net_save_path + 'train_loss.pth')
            torch.save(net, net_save_path + 'final_net.pth')
            torch.save(final_b, net_save_path + 'final_b.pth')
            torch.save(bvalue_learning, net_save_path + 'bvalue_learning.pth')

        #=======================================================
        print('=============FINAL rounded b-value===========')
        print(torch.round(final_b))
    print("Done")
