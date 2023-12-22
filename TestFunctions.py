import torch 
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def mse_loss_test(model_ae, X_fdload, dvc = 'cpu', selected_feature = 'None'):
    
  
   device = dvc
   model_ae.to(device)
   
   if selected_feature != "MEL_ENERGY":
       loss_test = torch.nn.MSELoss(reduction='none')
   else:
       loss_test = torch.nn.MSELoss(reduction='mean')
   #####
   mse_loss = []
   model_ae.eval()
   with torch.no_grad():
     for xt, _ in X_fdload:
         xt = xt.to(device).float()
         #x_pred = x_pred.to(device).float()
         x_pred = model_ae(xt)
         
         if selected_feature == "MEL_ENERGY":                     
             loss_t = loss_test(xt, x_pred)
         else:
             loss_t = loss_test(xt, x_pred).mean(axis=1)
             
         mse_loss.append(loss_t.detach().cpu().numpy())        
         
   if selected_feature == "MEL_ENERGY":      
       return np.array(mse_loss)
   else:
       return np.concatenate(mse_loss) 


def mse_loss_test_stft(model_ae, X_fdload, dvc = 'cpu', selected_feature = 'None'):

   device = dvc
   model_ae.to(device)
   loss_test = torch.nn.MSELoss(reduction='none')
 
   #####
   mse_loss = []
   model_ae.eval()
   with torch.no_grad():
     for xt, _ in X_fdload:
         xt = xt.to(device).float()
         #x_pred = x_pred.to(device).float()
         x_pred = model_ae(xt)
         
         loss_t = loss_test(xt, x_pred)
         loss_t = loss_t[:,0,:,:]
         loss_t = loss_t.mean(axis=1).mean(axis=1)
         mse_loss.append(loss_t.detach().cpu().numpy())        
         
   return np.concatenate(mse_loss) 


def valid_loss(X_valid, model_ae, loss_fn, model_type, device = 'cpu'):
  loss_valid = 0
  model_ae.eval()
  with torch.no_grad():
       for x, y in X_valid:
               batch_size = x.shape[0]

               x = x.to(device=device)
               y = y.to(device=device)

               outputs = model_ae(x.float())
               if model_type == 'VAE':
                  lossv = loss_fn(outputs, y.float()) + model_ae.encoder.kl 
               else:
                   lossv = loss_fn(outputs, y.float())
               loss_valid += lossv.item()
               
  return loss_valid/len(X_valid)



def apply_normalization_func(scaler_type, X_train, X_valid, selected_feature, apply_scaler):
    
    
    if (selected_feature == 'FFT' or selected_feature == 'MelEnergy') and apply_scaler:
       if scaler_type == "min_max":
           scaler = MinMaxScaler()
       elif scaler_type == "standard":
           scaler = StandardScaler()
           
       X_train = scaler.fit_transform(X_train)
       X_valid = scaler.transform(X_valid)

    if (selected_feature == 'STFT' or selected_feature == "MelLog") and apply_scaler:
       if scaler_type == "min_max": 
           X_train = np.array([min_max_normalize_matrix(matrix) for matrix in X_train])
           X_valid = np.array([min_max_normalize_matrix(matrix) for matrix in X_valid])
       elif scaler_type == "standard":
           X_train = np.array([z_score_normalize_matrix(matrix) for matrix in X_train])
           X_valid = np.array([z_score_normalize_matrix(matrix) for matrix in X_valid])   
           
    return X_train, X_valid  

def Make_DimensionReduction(X, **kwargs):

    DR_type = kwargs['dr_type']
    embeded_dim = kwargs['embeded_dim']
    
    if DR_type == 'TakeSamples':
       X_red = X[:, 0:embeded_dim]
       dr = None        
    elif DR_type == 'PCA':
      dr = PCA(n_components=embeded_dim)
      X_red = dr.fit_transform(X)    
      
    return dr, X_red
    
def Apply_DimensionReduction(X, dr, **kwargs):

    DR_type = kwargs['dr_type']
    embeded_dim = kwargs['embeded_dim']
    
    if DR_type == 'TakeSamples':
       X_red = X[:, 0:embeded_dim]
    elif DR_type == 'PCA':
      X_red = dr.transfrom(X)    
      
    return X_red

def min_max_normalize_matrix(matrix):
    return (matrix - matrix.min()) / (matrix.max() - matrix.min())

def z_score_normalize_matrix(matrix):
    mean = matrix.mean()
    std = matrix.std()
    return (matrix - mean) / std


def load_desired_data(save_folder, selected_machine_type, selected_feature, selected_machine_id, condition_type):
    
    save_folder_npy = save_folder + selected_machine_type + '_npy/' + selected_feature + "/"
    ids_string = "_id" + str(selected_machine_id)
    x_f_name = save_folder_npy + "X_features_" + selected_machine_type + "_" + selected_feature + ids_string + "_" + condition_type  + ".npy"
    y_f_name = save_folder_npy + "Y_" + selected_machine_type + "_" + selected_feature + ids_string + "_" + condition_type  + ".npy"

    X = np.load(x_f_name)
    Y = np.load(y_f_name)
    return X,Y

def load_data(save_folder, selected_machine_type, selected_feature, selected_machine_ids, condition_type):
    
    
    X_features_all, Y_features_all = [], []
    for mid in selected_machine_ids:
        Xt, Yt = load_desired_data(save_folder, selected_machine_type, selected_feature, mid, condition_type)
        X_features_all.append(Xt)
        Y_features_all.append(Yt)
        
    X_features_all = np.concatenate(X_features_all)    
    Y_features_all = np.concatenate(Y_features_all)    
    
    if selected_feature == 'STFT' or selected_feature == 'MelLog':
        X_features_all = X_features_all[:,None,:,:]
    
    return X_features_all, Y_features_all


