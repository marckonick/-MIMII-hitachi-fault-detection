# MIMII-hitachi-fault-detection
Fault detection methods for the MIMII Hitachi dataset

## Description 
  MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection
  Dataset available @ https://zenodo.org/records/3384388

## File Description
MAIN_FeatureExtraction.py  - extracting deatures
MAIN_TrainAE.py            - training model 
MAIN_JustTestModel.py      - script for testing saved models 


Current feature types:
- FFT - Fast Fourier Transform
- STFT - Short-Time Fourier Transform
- MelLog -   Mel-log spectrogram
- MEL_ENERGY -  Mel-log spectrogram energies - features taken from the baseline model repository 
              - https://github.com/y-kawagu/dcase2020_task2_baseline

## Example results

### pump_0dB
- machine type: pump
- SNR: 0dB 

MEL_ENER + DNN 
- Machine ID 0.0 , ROC AUC Score:  0.9249171880750828 
- Machine ID 2.0 , ROC AUC Score:  0.7633087633087633
- Machine ID 4.0 , ROC AUC Score:  0.9384210526315788
- Machine ID 6.0 , ROC AUC Score:  0.9985294117647059

STFT + CNN
- Machine ID 0 , ROC AUC Score:  0.752005347593583
- Machine ID 2 , ROC AUC Score:  0.9014436122869858
- Machine ID 4 , ROC AUC Score:  0.2892814371257485
- Machine ID 6 , ROC AUC Score:  0.8340000000000001


## Notes 
- Changing configuration of the STFT or MelLog data will require 
  modifying network architecture 

## Fututre work 

- Logger 
- Make models more configurable
- Add more advanced models/features
- Comapre results across different machines 

