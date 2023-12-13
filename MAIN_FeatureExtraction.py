import numpy as np
import os
import FeatureExtractionFunctions as FEF
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--selected_machine_type",
                    type = str, default = "fan_-6dB") #"fan_-6dB" # -6_dB_fan, pump_0dB, fan_-6dB
parser.add_argument("--selected_feature",
                    type = str, default = "FFT") # FFT, STFT, MelEnergy, MelLog
parser.add_argument("--condition_type",
                    type = str, default = "normal") # normal, abnormal
parser.add_argument("--N_samples_2_extract",
                    type = int, default = -1)  #-1 # 100
parser.add_argument("--apply_filter",
                    type = str, default = "False")
parser.add_argument("--f_type",
                    type = str, default = "median")
parser.add_argument("--filter_order",
                    type = int, default = 11)
parser.add_argument("--cutoff",
                    type = int, default = 11)
parser.add_argument("--win_len",
                    type = int, default = 1024)
parser.add_argument("--overlap_l",
                    type = int, default = 256)
parser.add_argument("--hop_l",
                    type = int, default = 512)
parser.add_argument("--n_mels",
                    type = int, default = 128)
parser.add_argument("--frames_mel",
                    type = int, default = 5)
parser.add_argument("--power_mel",
                    type = float, default = 2.0)
parser.add_argument("--save_2_npy",
                    type = bool, default = False)
parser.add_argument("--file_folder",
                    type = str, default = "../")
parser.add_argument("--save_folder",
                    type = str, default = "../SavedFeatures_NPY/")


parser.add_argument('--selected_machine_ids', nargs='+', action = 'append', default = [0,2,4,6])




def main():
    
 args = parser.parse_args()

 
 selected_machine_type = args.selected_machine_type 
 selected_feature = args.selected_feature
 condition_type = args.condition_type
 N_samples_2_extract = args.N_samples_2_extract


 filter_kwarg = {'apply_filter':args.apply_filter, 'f_type':args.f_type, 'filter_order':args.filter_order,'cutoff':args.cutoff} 
 kwarg_args = { 'win_len':args.win_len, 'overlap_l':args.overlap_l, 'hop_l':args.hop_l, "n_mels":args.n_mels, 
                 'frames_mel':args.frames_mel, 'power_mel':args.power_mel}
 kwarg_args.update(filter_kwarg)


 save_2_npy = args.save_2_npy
 save_folder = args.save_folder 

 selected_machine_ids = args.selected_machine_ids

 curr_label = 0


 for selected_machine_id in selected_machine_ids:

    X_features_all = []
    Y_features_all = []
    
    folder_name = args.file_folder +  selected_machine_type + "/id_0" + str(selected_machine_id) + "/" + condition_type + "/"
    recording_names = os.listdir(folder_name) 
    temp_kwargs = {'folder_name':folder_name}
    kwarg_args.update(temp_kwargs)

    print(f"\nExtracting {selected_feature} features for machine {selected_machine_id} ... ")
    t0 = time.time()
    X_features_t, N_samples = FEF.ExtractSelectedFeatures(N_samples_2_extract, recording_names, selected_feature, **kwarg_args)
    t1 = time.time()
    print(f"Feature extraction done! Elapsed time: {t1-t0:.2f} seconds\n")
    
    X_features_all.append(X_features_t)
    Y_features_all.append(np.ones(X_features_t.shape[0])*curr_label)
    curr_label += 1

    X_features_all = np.concatenate(X_features_all)
    Y_features_all = np.concatenate(Y_features_all)

    del X_features_t

    if save_2_npy:
        save_folder_npy = save_folder + selected_machine_type + '_npy/' + selected_feature + "/"
        ids_string = ""
  
        smi = selected_machine_id
        
        #for smi in selected_machine_ids:
        #    ids_string += "_id" + str(smi)
         
        ids_string += "_id" + str(smi)
        x_f_name = save_folder_npy + "X_features_" + selected_machine_type + "_" + selected_feature + ids_string + "_" + condition_type  + ".npy"
        y_f_name = save_folder_npy + "Y_" + selected_machine_type + "_" + selected_feature + ids_string + "_" + condition_type  + ".npy"
    
        isExist = os.path.exists(save_folder_npy)
        if not isExist:
            os.makedirs(save_folder_npy) 
    
    
        print(f"Saving features to : \n X - {x_f_name} \n Y - {y_f_name} " )
        np.save(x_f_name, X_features_all) 
        np.save(y_f_name, Y_features_all) 
    
    
    
if __name__ == '__main__':
    main()    