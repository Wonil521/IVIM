import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
fmin=0;     fmax=0.2
Dmin=0;     Dmax=1.8e-3
Dpmin=0;    Dpmax=20e-3
S0min=200;    S0max=1700

step_n = 20

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_num_bvalue', type=int, default=4)
    parser.add_argument('--SNR', type=int, default=100)
    parser.add_argument('--test_trials', type=int, default=50)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=1.0)
    parser.add_argument('--lambda3', type=float, default=1.0)
    parser.add_argument('--data_path', type=str, default='/home/user/')
    args = parser.parse_args()

    test_num_bvalue = args.test_num_bvalue
    test_SNR = args.SNR
    trials = args.test_trials
    l1 = args.lambda1
    l2 = args.lambda2
    l3 = args.lambda3
    GPU_NUM = args.gpu

    for num_bvalue in ([test_num_bvalue]):
        for SNR in ([test_SNR]):
            data_path = args.data_path+'bvalue_'+str(num_bvalue)+'_SNR_'+str(SNR)+'/'+'l1_'+str(l1)+'_l2_'+str(l2)+'_l3_'+str(l3)+'/'

            S0_test = 300
            Dp_test = np.linspace(Dpmin, Dpmax, step_n)
            D_test = np.linspace(Dmin, Dmax, step_n)
            f_test = np.linspace(fmin, fmax, step_n)

            net = torch.load(data_path+'final_net.pth', map_location='cpu')
            b = np.round(np.array(torch.load(data_path+'final_b.pth', map_location='cpu')))
            bvalue_learning = torch.load(data_path+'bvalue_learning.pth', map_location='cpu')

            f_nrmse = np.zeros(trials)
            Dp_nrmse = np.zeros(trials)
            D_nrmse = np.zeros(trials)

            TPE_f = np.zeros(trials)
            TPE_D = np.zeros(trials)
            TPE_Dp = np.zeros(trials)

            device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
            torch.cuda.set_device(device)  # change allocation of current GPU
            net = net.to(device)

            # Generate test data
            num_samples_test=step_n**3
            S_test = np.zeros((step_n**3,num_bvalue))
            f_test_ref= np.zeros((step_n**3))
            D_test_ref= np.zeros((step_n**3))
            Dp_test_ref= np.zeros((step_n**3))
            for i1 in range(step_n):
                for i2 in range(step_n):
                    for i3 in range(step_n):
                        S_test[i1*step_n*step_n+i2*step_n+i3][:] = S0_test*(f_test[i1]*np.exp(-b*Dp_test[i2])+(1-f_test[i1])*np.exp(-b*D_test[i3]))
                        f_test_ref[i1 * step_n * step_n + i2 * step_n + i3] = f_test[i1]
                        D_test_ref[i1 * step_n * step_n + i2 * step_n + i3] = D_test[i3]
                        Dp_test_ref[i1 * step_n * step_n + i2 * step_n + i3] = Dp_test[i2]

            S_test=torch.tensor(S_test).float()
            for t in range(trials):
                # Generate noise
                noise_real=torch.normal(torch.tensor(np.zeros((num_samples_test, num_bvalue))),std=3).float()
                noise_imag = torch.normal(torch.tensor(np.zeros((num_samples_test, num_bvalue))), std=3).float()

                MR_signals_real = S_test + noise_real
                MR_signals = torch.sqrt(MR_signals_real**2+noise_imag**2).to(device)

                Signal, b_out, Output_test, f, D, Dp, Signal_pred = net(0, torch.tensor(b).to(device), MR_signals, 'test', noise_real, noise_imag)

                f_est=Output_test.detach().cpu().numpy()[:,0]*(fmax-fmin)+fmin
                D_est=Output_test.detach().cpu().numpy()[:,1]*(Dmax-Dmin)+Dmin
                Dp_est=Output_test.detach().cpu().numpy()[:,2]*(Dpmax-Dpmin)+Dpmin

                f_nrmse[t]  = np.square(np.subtract(f_test_ref,f_est)).mean()**0.5/(np.mean(f_test_ref))
                D_nrmse[t]  = np.square(np.subtract(D_test_ref, D_est)).mean() ** 0.5 / (np.mean(D_test_ref))
                Dp_nrmse[t] = np.square(np.subtract(Dp_test_ref,Dp_est)).mean()**0.5/(np.mean(Dp_test_ref))

            print('nrmse of f: %.4f +/- %.4f' % (np.mean(f_nrmse), np.std(f_nrmse)))
            print('nrmse of D: %.4f +/- %.4f' % (np.mean(D_nrmse), np.std(D_nrmse)))
            print('nrmse of Dp: %.4f +/- %.4f' % (np.mean(Dp_nrmse), np.std(Dp_nrmse)))

            print('optimized b-values = ', np.sort(b))

            plt.figure()
            plt.plot(bvalue_learning)
            plt.xlabel('epochs')
            plt.ylabel('b-values $[s/mm^2]$')
            plt.show()

            train_loss = torch.load(data_path + 'train_loss.pth', map_location='cpu')
            validation_loss = torch.load(data_path + 'validation_loss.pth', map_location='cpu')
            plt.figure()
            plt.plot(train_loss)
            plt.plot(validation_loss)
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend(['train_loss','validation_loss'])
            plt.axis([0, len(train_loss), 0, 10e-1])
            plt.show()

