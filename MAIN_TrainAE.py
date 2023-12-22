import numpy as np
import torch
import TorchDataStructures as TDS
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import TorchModels as t_model
import torch.optim as optim
import TestFunctions as test_f
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import argparse

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
parser.add_argument("--save_model",
                    type = bool, default = True)
parser.add_argument("--model_name_2_save",
                    type = str, default = "default_name")

####### Training params #################
parser.add_argument("--test_percent",
                    type = float, default = 0.1)
parser.add_argument("--n_epochs",
                    type = int, default = 2)
parser.add_argument("--lr",
                    type = float, default = 0.0003)
parser.add_argument("--batch_size",
                    type = int, default = 64)


parser.add_argument("--to_test_model",
                    type = bool, default = True)


def main():
    
  args = parser.parse_args()
    

  ################## PARAMS ##############################################
  selected_feature = args.selected_feature  #FFT, STFT, MelLog, MEL_ENERGY
  selected_machine_type = args.selected_machine_type # pump_-6dB, pump_0dB
  apply_scaler =  args.apply_scaler
  device = 'cpu'
  if args.use_cuda == True:
     device =  'cuda'
  model_type = args.model_type
  save_folder = args.feature_save_folder  
  selected_machine_ids = args.selected_machine_ids 
      
  test_normal_pc = args.test_percent 
  save_model = args.save_model 
  batch_size = args.batch_size
  
  apply_dim_red = args.apply_dim_red
  dimred_kwarg = {'dr_type':args.dr_type, 'embeded_dim':args.embeded_dim} 
  dr = None
  
  to_test_model = args.to_test_model
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
      
    
  ############# DATA ALTER #############
  if apply_scaler:
    X_train, X_valid = test_f.apply_normalization_func("min_max", X_train, X_valid, selected_feature, apply_scaler) # min_max, standard
  
  if apply_dim_red:
     dr, X_train = test_f.Make_DimensionReduction(X_train, **dimred_kwarg) 
     X_valid = test_f.Apply_DimensionReduction(X_valid, dr, **dimred_kwarg)

  ############# Torch Data ############# 
  X_train = TDS.labeled_dataset(X_train, X_train)
  X_train = data.DataLoader(X_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
  
  if selected_feature == "MEL_ENERGY":
    X_valid = TDS.labeled_dataset(X_valid, X_valid)
    X_valid = data.DataLoader(X_valid, batch_size=xt_x, shuffle=True, num_workers=0, drop_last=False)
  else:
    X_valid = TDS.labeled_dataset(X_valid, X_valid)
    X_valid = data.DataLoader(X_valid, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)




###################  TRAIN MODEL #########################

  model_ae = None
  x,y = next(iter(X_train))
  if selected_feature == "FFT":
    if model_type == "DNN":
        model_ae = t_model.DNN_AE(x.size()[1], [128,128,8])       
  elif selected_feature == "STFT":
    if model_type == "CNN":  
        model_ae = t_model.CNN_AE(n_chans1=[16,16,16])
  elif selected_feature == 'MelLog':
    if model_type == "CNN":  
        model_ae = t_model.CNN_AE_MEL(k_size = [6,4,3], n_chans1=[16,16,16])
  elif selected_feature == "MEL_ENERGY":
    if model_type == "VAE":
        model_ae =  t_model.VariationalAutoencoder(N_input = x.size()[1], latent_dims = 12, hidden_dim = 64, device='cpu')
    elif  model_type == "DNN":
        model_ae = t_model.DNN_AE_ME(x.size()[1], n_layers = [128,128,128,128,8])
    
  if model_ae is None:
      print("Invalid model type!!! ")
      return -1
    
  if device == 'cuda':
    model_ae = model_ae.cuda()


  ################# TRAINING #########################
  #optimizer = optim.AdamW(model_ae.parameters(), lr=0.0001, weight_decay = 5e-4)
  optimizer = optim.Adam(model_ae.parameters(), lr=args.lr) # weight_decay = 5e-4
  loss_fn = torch.nn.MSELoss()
  #print(model_ae.number_of_params())
  n_epochs = args.n_epochs
  

  model_ae.train()
  print("Starting model training ...\n")
  for epoch in range(1,n_epochs+1):
    loss_train = 0.0
    for x, y in X_train:
               batch_size = x.shape[0]

               x = x.to(device=device)
               y = y.to(device=device)

               outputs = model_ae(x.float())
               
               if model_type == 'VAE':
                   loss = loss_fn(outputs, y.float()) + model_ae.encoder.kl # torch.argmax(y, axis=1)
               else:
                   loss = loss_fn(outputs, y.float())
                   
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
               loss_train += loss.item()

    #if epoch % 20==0:
    #  for param_group in optimizer.param_groups:
    #     param_group['lr'] /= 2
    #  print("New lr: ", param_group['lr'])
      #test_model(model, X_test, num_classes)
    if epoch % 1==0:
      print(" Epoch ", epoch, "/",n_epochs, " Train loss = ", float(loss_train/len(X_train)))
      vloss = test_f.valid_loss(X_valid, model_ae, loss_fn, model_type, device)
      print(" Epoch ", epoch, "/",n_epochs, " Valid loss = ", vloss)
      
      
      
  if save_model:
    if args.model_name_2_save == "default_name":
          model_name = "PyTorch_AutoEnc_" + selected_machine_type + "_" +  selected_feature + "_" + model_type + ".pt"
    else:
          model_name = args.model_name_2_save
         
          
    torch.save(model_ae.state_dict(), 'saved_models/' + model_name)   
    print(f"Model {model_name} saved - congratulations :)")


 
  ############ COMPUTE TEST SCORE #############
  
  if to_test_model:
      print("Testing model now ...")
      
      X_abnormal, Y_abnormal = test_f.load_data(save_folder, selected_machine_type, selected_feature, selected_machine_ids, "abnormal")
      if selected_feature == "MEL_ENERGY": 
          X_abnormal = np.reshape(X_abnormal, (X_abnormal.shape[0]*X_abnormal.shape[1], X_abnormal.shape[2]))

      if apply_dim_red:
          X_abnormal = test_f.Apply_DimensionReduction(X_abnormal, dr, **dimred_kwarg)
     

      X_abnormal = TDS.labeled_dataset(X_abnormal, X_abnormal)

      if selected_feature == "MEL_ENERGY":
          X_abnormal = data.DataLoader(X_abnormal, batch_size=xt_x, shuffle=False, num_workers=0, drop_last=False)
      else: 
          X_abnormal = data.DataLoader(X_abnormal, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    
    
    
      if selected_feature == "STFT" or selected_feature == "MelLog": 
         test_score_normal = test_f.mse_loss_test_stft(model_ae, X_valid, device, selected_feature)
         test_score_abnormal = test_f.mse_loss_test_stft(model_ae, X_abnormal, device, selected_feature)
      else:
          test_score_normal = test_f.mse_loss_test(model_ae, X_valid, device, selected_feature)
          test_score_abnormal = test_f.mse_loss_test(model_ae, X_abnormal, device, selected_feature)


      #plt.plot(test_score_normal)
      #plt.plot(test_score_abnormal)
      #plt.legend(['Normal loss', 'Abnormal loss'])


      for machine_idx in selected_machine_ids:

          machine_idx /= 2
          m_idx_sel_normal = np.where(Y_valid==machine_idx)[0]
          m_idx_sel_abnormal = np.where(Y_abnormal==machine_idx)[0]


          labels_4_test = np.concatenate((np.zeros_like(test_score_normal[m_idx_sel_normal]), np.ones_like(test_score_abnormal[m_idx_sel_abnormal])))
          preds_4_test = np.concatenate((test_score_normal[m_idx_sel_normal], test_score_abnormal[m_idx_sel_abnormal]))

          ROCAUC_test_score = roc_auc_score(labels_4_test, preds_4_test)
     
          print("Machine ID", machine_idx*2, ", ROC AUC Score: ", ROCAUC_test_score)


if __name__ == '__main__':
    main()










    