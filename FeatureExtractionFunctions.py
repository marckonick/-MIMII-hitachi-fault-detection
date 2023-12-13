import numpy as np
from scipy.fft import fft, fftfreq
from scipy import signal 
from scipy.signal import butter, lfilter
import librosa 
import sys


def perform_stft(X_ex, Fs, win_len, overlap_l):
 
    # noverlap - Number of points to overlap between segments.
    _,_, Xxx = signal.stft(X_ex, Fs, nperseg=win_len, noverlap=overlap_l)

    return np.abs(Xxx)

def perform_fft(X_ex, Fs):

    T = 1.0/Fs
    N = len(X_ex)
    
    yf = fft(X_ex)
    xf = fftfreq(N, T)[:N//2]

    return xf, 2.0/N*np.abs(yf[0:N//2])

def perform_mellog(X_ex, Fs, win_len, hop_l):
    mel_signal = librosa.feature.melspectrogram(y = np.array(X_ex, dtype=np.float32)  , sr = Fs, hop_length = hop_l, n_fft = win_len)
    spectrogram = np.abs(mel_signal)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    
    return spectrogram

def perform_mel_ener(X_ex, Fs, **kwargs):
    
    n_mels = kwargs['n_mels']
    power = kwargs['power_mel']
    frames = kwargs['frames_mel']
    win_len = kwargs['win_len']
    hop_l = kwargs['hop_l']
    
    dims = n_mels * frames

    
    mel_spectrogram = librosa.feature.melspectrogram(y=np.array(X_ex, dtype=np.float32),
                                                     sr=Fs,
                                                     n_fft= win_len,
                                                     hop_length = hop_l,
                                                     n_mels=n_mels,
                                                     power=power)
    
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1    

    # 05 skip too short clips
    if vector_array_size < 1:
        return np.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = np.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T
        
    return vector_array



def preprocess_filter_signal(X_ex, filter_type = 'median', Fs=16000, filter_order = 17, cutoff = 1000):
    
    if filter_type == 'median':
        return signal.medfilt(X_ex, filter_order)
    elif filter_type == 'lowpass':
         b, a = butter(filter_order, cutoff, fs=Fs, btype='low', analog=False)
         return lfilter(b, a, X_ex)
     
        
def extract_stft_features(N_samples, recording_names,  **kwargs):
    
    if N_samples == -1 or N_samples >len(recording_names):
        N_samples = len(recording_names)
        
    apply_filter = kwargs['apply_filter']        
    win_len = kwargs['win_len'] 
    overlap_l = kwargs['overlap_l']
    folder_name = kwargs['folder_name']
    idxs = np.random.choice(len(recording_names),N_samples)
    
    X_features_fft = []
    for idx in idxs:
        file_2_read = folder_name + recording_names[idx]
        
        #Fs, X_ex = wavfile.read(file_2_read)
        X_ex, Fs = librosa.load(file_2_read, sr=None, mono=True) # ovooo
        #X_ex = X_ex[:,0]        
        if apply_filter:       
           X_ex = preprocess_filter_signal(X_ex, filter_type = kwargs['f_type'], Fs=16000, filter_order = kwargs['filter_order'], cutoff = kwargs['cutoff'])
        
        xstft_temp = perform_stft(X_ex, Fs, win_len, overlap_l)
        X_features_fft.append(xstft_temp)


    return np.array(X_features_fft), N_samples



def extract_fft_features(N_samples, recording_names, **kwargs):
    
    if N_samples == -1 or N_samples >len(recording_names):
        N_samples = len(recording_names)
     
    apply_filter = kwargs['apply_filter']   
    folder_name = kwargs['folder_name']
    idxs = np.random.choice(len(recording_names),N_samples)
    
    
    X_features_fft = []
    for idx in idxs: 
        file_2_read = folder_name + recording_names[idx]
        X_ex, Fs = librosa.load(file_2_read, sr=None, mono=True) # ovooo
        if apply_filter:
           X_ex = preprocess_filter_signal(X_ex, filter_type = kwargs['f_type'], Fs=16000, filter_order = kwargs['filter_order'], cutoff = kwargs['cutoff'])
           
           
        _, xfft_temp = perform_fft(X_ex, Fs)
        if apply_filter:
            xfft_temp = xfft_temp[0:int(kwargs['cutoff']*10)]
            
        X_features_fft.append(xfft_temp)


    return np.array(X_features_fft), N_samples
   
def extract_mellog_features(N_samples, recording_names,  **kwargs):
    
    if N_samples == -1 or N_samples >len(recording_names):
        N_samples = len(recording_names)
        
    apply_filter = kwargs['apply_filter']        
    win_len = kwargs['win_len'] 
    hop_l = kwargs['hop_l']
    folder_name = kwargs['folder_name']
    
    idxs = np.random.choice(len(recording_names),N_samples)
    
    X_features_ml = []
    for idx in idxs:
        file_2_read = folder_name + recording_names[idx]
        #Fs, X_ex = wavfile.read(file_2_read)
        X_ex, Fs = librosa.load(file_2_read, sr=None, mono=True) # ovooo
        #X_ex = X_ex[:,0] # vec smo se sveli na jedan kanal         
        if apply_filter:    
           X_ex = preprocess_filter_signal(X_ex, filter_type = kwargs['f_type'], Fs=16000, filter_order = kwargs['filter_order'], cutoff = kwargs['cutoff'])
        
        xmell_temp = perform_mellog(X_ex, Fs, win_len, hop_l)
        X_features_ml.append(xmell_temp)


    return np.array(X_features_ml), N_samples
 
def extract_mel_energ_features(N_samples, recording_names,  **kwargs):
    
    if N_samples == -1 or N_samples >len(recording_names):
        N_samples = len(recording_names)
        
    apply_filter = kwargs['apply_filter']        
    folder_name = kwargs['folder_name']
    
    X_features_ml = []
    for idx in range(N_samples):
        file_2_read = folder_name + recording_names[idx]
        #Fs, X_ex = wavfile.read(file_2_read)     
        X_ex, Fs = librosa.load(file_2_read, sr=None, mono=True) # ovooo
        #X_ex = X_ex[:,0] # vec smo se sveli na jedan kanal         
        if apply_filter:    
           X_ex = preprocess_filter_signal(X_ex, filter_type = kwargs['f_type'], Fs=16000, filter_order = kwargs['filter_order'], cutoff = kwargs['cutoff'])
        
        xmell_temp = perform_mel_ener(X_ex, Fs, **kwargs)
        X_features_ml.append(xmell_temp)


    return np.concatenate(X_features_ml, 0), N_samples


def ExtractSelectedFeatures(N_samples, recording_names, selected_feature, **kwargs):
    
    if selected_feature == "FFT":
       X_features_extracted, N_samples = extract_fft_features(N_samples, recording_names, **kwargs)
    elif selected_feature == "STFT":
       X_features_extracted, N_samples = extract_stft_features(N_samples, recording_names, **kwargs)
    elif selected_feature == "MelLog":
          X_features_extracted, N_samples = extract_mellog_features(N_samples, recording_names, **kwargs)
    elif selected_feature == "MEL_ENERGY":
       X_features_extracted, N_samples = extract_mel_energ_features(N_samples, recording_names, **kwargs)          
    else:
        print(f" {selected_feature} - nNot a regualr feature type !!!")
        return -1, 0
        
    return X_features_extracted, N_samples 

def get_mel_en_test_idx(idx_anom, mpltkl):
    
    idxn = np.zeros((len(idx_anom), mpltkl), dtype=np.int32)
    cntr = 0
    
    for idxt in idx_anom:
        idxn[cntr,:] = np.linspace(idxt*mpltkl, (idxt+1)*mpltkl-1, mpltkl, dtype=np.int32)
        cntr +=1
    
    return idxn.flatten()
   