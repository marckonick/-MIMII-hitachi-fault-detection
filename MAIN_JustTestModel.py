import os
import torch 
import numpy as np
import TestFunctions as test_f
import matplotlib.pyplot as plt
import TorchModels as t_model
import TorchDataStructures as TDS
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import copy 
import argparse
import time
import logging
from datetime import datetime

parser = argparse.ArgumentParser()

####### DATA/FEATURES ############
parser.add_argument("--selected_machine_type",
                    type = str, default = "pump_0dB") #"fan_-6dB" # -6_dB_fan, pump_0dB, fan_-6dB 
parser.add_argument("--selected_feature",
                    type = str, default = "MEL_ENERGY") # FFT, STFT, MEL_ENERGY, MelLog
parser.add_argument("--feature_save_folder", 
                    type = str, default = "../SavedFeatures_NPY/")  # where the features are saved 
parser.add_argument('--selected_machine_ids', nargs='+', action = 'append', default = [0,2,4,6])
parser.add_argument("--apply_scaler",
                    type = bool, default = False) # apply standard scaler

######### DIMENSION REDUCTION #########
parser.add_argument("--apply_dim_red",
                    type = bool, default = False)
parser.add_argument("--dr_type",
                    type = str, default = 'TakeSamples' ) # TakeSamples, PCA,
parser.add_argument("--embeded_dim",
                    type = int, default = 20000)


#### MODEL ######
parser.add_argument("--model_type",
                    type = str, default = "DNN") # DNN, CNN 
parser.add_argument("--use_cuda",
                    type = bool, default = False)
parser.add_argument("--model_name_2_save",
                    type = str, default = "default_name")

###### LOGGING ###########
parser.add_argument("--log_folder",
                    type = str, default = "Logs/")


def main():
 #################3 PARAMS ##############################################
 args = parser.parse_args()
 
 
 selected_feature = args.selected_feature  #FFT, STFT, MelLog, MEL_ENERGY
 selected_machine_type = args.selected_machine_type # pump_-6dB, pump_0dB
 apply_scaler =  args.apply_scaler
 device = 'cpu'
 if args.use_cuda == True:
     device =  'cuda'
 model_type = args.model_type
 save_folder = args.feature_save_folder  
 selected_machine_ids = args.selected_machine_ids 


 model_name = "PyTorch_AutoEnc_" + selected_machine_type + "_" +  selected_feature + "_" + model_type + ".pt"
 test_normal_pc = 0.1

 apply_dim_red = args.apply_dim_red
 dimred_kwarg = {'dr_type':args.dr_type, 'embeded_dim':args.embeded_dim} 
 dr = None
  

 ######## LOGGER PARAMS ###########
 log_folder = args.log_folder

 current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
 log_file_name = os.path.join(log_folder, f"Test_Model_log_{current_date}.log")
 #logging.basicConfig(filename=log_file_name, level=logging.INFO, format='%(asctime)s - %(message)s') 
 logging.basicConfig(filename=log_file_name, level=logging.INFO, format='%(message)s')   
##########################LOAD DATA###################
 print("Loading Data ...\n")
  
 X_features_all, Y_features_all = test_f.load_data(save_folder, selected_machine_type, selected_feature, selected_machine_ids, "normal")

 if selected_feature == "MEL_ENERGY":
    xt_x, xt_y = X_features_all.shape[1], X_features_all.shape[2]
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_features_all, Y_features_all, test_size=test_normal_pc, random_state=42)
    
    X_train = np.reshape(X_train, (X_train.shape[0]*xt_x, xt_y))
    Y_train = np.repeat(Y_train, xt_x)
    
    X_valid = np.reshape(X_valid, (X_valid.shape[0]*xt_x, xt_y))
 else:
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_features_all, Y_features_all, test_size=test_normal_pc, random_state=42)
  

 X_abnormal, Y_abnormal = test_f.load_data(save_folder, selected_machine_type, selected_feature, selected_machine_ids, "abnormal")
 X_abnormal = np.reshape(X_abnormal, (X_abnormal.shape[0]*X_abnormal.shape[1], X_abnormal.shape[2]))    
 x_shape = X_valid.shape[1]
 ############# DATA ALTER #############
 if apply_scaler:
    X_train, X_valid = test_f.apply_normalization_func("min_max", X_train, selected_feature, apply_scaler, X_valid) # min_max, standard
    X_abnormal, _ = test_f.apply_normalization_func("min_max", X_train, selected_feature, apply_scaler, None)
    
    
 if apply_dim_red:
     dr, X_train = test_f.Make_DimensionReduction(X_train, **dimred_kwarg) 
     X_valid = test_f.Apply_DimensionReduction(X_valid, dr, **dimred_kwarg)
     X_abnormal = test_f.Apply_DimensionReduction(X_abnormal, dr, **dimred_kwarg)
    
     

 X_train = TDS.labeled_dataset(X_train, X_train)
 X_train = data.DataLoader(X_train, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

 if selected_feature != "MEL_ENERGY":
    X_valid = TDS.labeled_dataset(X_valid, X_valid)
    X_valid = data.DataLoader(X_valid, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
 else:
    X_valid = TDS.labeled_dataset(X_valid, X_valid)
    X_valid = data.DataLoader(X_valid, batch_size=309, shuffle=True, num_workers=0, drop_last=False)


 X_abnormal = TDS.labeled_dataset(X_abnormal, X_abnormal)

 if selected_feature != "MEL_ENERGY":
    X_abnormal = data.DataLoader(X_abnormal, batch_size=32, shuffle=False, num_workers=0, drop_last=False)
 else:
    X_abnormal = data.DataLoader(X_abnormal, batch_size=309, shuffle=False, num_workers=0, drop_last=False)
 ################# LOAD MODEL ######################


 if selected_feature == "FFT":
    model_ae = t_model.DNN_AE(x_shape, [128,128,8])
 elif selected_feature == "STFT":
    model_ae = t_model.CNN_AE(n_chans1=[16,16,16])
 elif selected_feature == 'MelLog':
    model_ae = t_model.CNN_AE_MEL(k_size = [6,4,3], n_chans1=[16,16,16])
 elif selected_feature == "MEL_ENERGY":
    if model_type == "VAE":
        model_ae =  t_model.VariationalAutoencoder(N_input = x_shape, latent_dims = 12, hidden_dim = 64, device='cpu')
    elif  model_type == "DNN":
        model_ae = t_model.DNN_AE_ME(x_shape, n_layers = [128,128,128,128,8])
    else:
        model_ae = None
        

 if args.model_name_2_save == "model":
    if args.model_name_2_save == "default_name":
          model_name = "PyTorch_AutoEnc_" + selected_machine_type + "_" +  selected_feature + "_" + model_type + ".pt"
    else:
          model_name = args.model_name_2_save
          
      
 model_ae.load_state_dict(torch.load('saved_models/'  + model_name))
 model_ae.to(device)
 model_ae.eval()
 ############ COMPUTE LOSS #############
 if selected_feature == "STFT" or selected_feature == "MelLog": 
    test_score_normal = test_f.mse_loss_test_stft(model_ae, X_valid, device, selected_feature)
    test_score_abnormal = test_f.mse_loss_test_stft(model_ae, X_abnormal, device, selected_feature)
 else:
    test_score_normal = test_f.mse_loss_test(model_ae, X_valid, device, selected_feature)
    test_score_abnormal = test_f.mse_loss_test(model_ae, X_abnormal, device, selected_feature)



 #plt.plot(test_score_normal)
 #plt.plot(test_score_abnormal)
 #plt.legend(['Normal loss', 'Abnormal loss'])
 logging.info(f"Date - {current_date}\n")
 logging.info(f"\nLogging test results for the feature type {selected_feature}, model type: {model_type}\n")
 for machine_idx in selected_machine_ids:

  machine_idx /= 2
  m_idx_sel_normal = np.where(Y_valid==machine_idx)[0]
  m_idx_sel_abnormal = np.where(Y_abnormal==machine_idx)[0]


  labels_4_test = np.concatenate((np.zeros_like(test_score_normal[m_idx_sel_normal]), np.ones_like(test_score_abnormal[m_idx_sel_abnormal])))
  preds_4_test = np.concatenate((test_score_normal[m_idx_sel_normal], test_score_abnormal[m_idx_sel_abnormal]))

  ROCAUC_test_score = roc_auc_score(labels_4_test, preds_4_test)

  print("Machine ID", machine_idx*2, ", ROC AUC Score: ", ROCAUC_test_score)
  logging.info(f"Machine ID  {machine_idx*2} ROC AUC Score: {ROCAUC_test_score}")


if __name__ == '__main__':
    main()

