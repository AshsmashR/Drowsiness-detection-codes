import os
import matplotlib.pyplot as plt
import numpy as np
import mne
#
n_channels = 31        
sfreq = 512          
dtype = 'int16'       
raw_data = np.fromfile(r"D:\v13_session1_BL_8.11.2024.eeg", dtype=dtype)
raw_data = raw_data.reshape((n_channels, -1))
ch_names = [f"EEG{i+1}" for i in range(n_channels)]
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(raw_data, info)
raw.save("subject_raw.fif", overwrite=True)
raw.plot(n_channels=n_channels, scalings='auto')
raw = mne.io.RawArray(raw_data, info)
print(raw.info['ch_names'])
#########

import mne

data_dir = r"F:\EEGFILES"
vhdr_file = data_dir + r"\v13_session1_BL_8.11.2024.vhdr"
raw = mne.io.read_raw_brainvision(
    vhdr_file,
    preload=True
)
print(raw)
raw.plot()
events, event_id = mne.events_from_annotations(raw)
print("Event IDs:", event_id)
print("Number of events:", len(events))
raw.annotations
##selecting few channels to plot
print(raw.ch_names)
channels = ['P7', 'P8', 'O1', 'O2']
raw_pick = raw.copy().pick_channels(channels)
raw_pick.plot(
    scalings='auto'
)
psd = raw_pick.compute_psd(
    method='welch',
    fmin=1,
    fmax=40,
    n_fft=2048
)
psd.plot()
psds, freqs = psd.get_data(return_freqs=True)
# psds shape: (n_channels, n_freqs)
print("PSD shape:", psds.shape)
print("Frequencies shape:", freqs.shape)
#######eeg preprocessing##
print(raw.info["bads"])
n_time_samps = raw.n_times
time_secs = raw.times
ch_names = raw.ch_names
n_chan = len(ch_names)  # note: there is no raw.n_channels attribute
print(
    f"the (cropped) sample data object has {n_time_samps} time samples and "
    f"{n_chan} channels."
)
print(f"The last time sample is at {time_secs[-1]} seconds.")
print("The first few channel names are {}.".format(", ".join(ch_names[:3])))
print()  # insert a blank line in the output

# some examples of raw.info:
print("bad channels:", raw.info["bads"])  # chs marked "bad" during acquisition
print(raw.info["sfreq"], "Hz")  # sampling frequency
print(raw.info["description"], "\n")  # miscellaneous acquisition info
print(raw.info)
# Filtering the data
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import mne


print("Bad channels:", raw.info["bads"])   # channels marked bad
print(raw.info["sfreq"], "Hz")              # sampling frequency
print(raw.info["description"], "\n")        # acquisition info
print(raw.info)

##FILTERING###
sfreq = raw.info["sfreq"]
nyq = sfreq / 2.0
x = raw.get_data()          
lp_cut = 140.0  # Hz

sos_lp = signal.iirfilter(
    N=4,
    Wn=lp_cut / nyq,
    btype="lowpass",
    ftype="butter",
    output="sos"
)

x_lp = signal.sosfiltfilt(sos_lp, x, axis=1)
del sos_lp

powerline = 50.0     
Q = 30.0             
harmonics = [powerline, 2 * powerline]
x_clean = x_lp.copy()

for f in harmonics:
    if f < nyq:
        b, a = signal.iirnotch(w0=f / nyq, Q=Q)
        x_clean = signal.filtfilt(b, a, x_clean, axis=1)

hp_cut = 0.1  # Hz
sos_hp = signal.iirfilter(
    N=4,
    Wn=hp_cut / nyq,
    btype="highpass",
    ftype="butter",
    output="sos"
)

x_clean = signal.sosfiltfilt(sos_hp, x_clean, axis=1)
del sos_hp
raw_filt = raw.copy()
raw_filt._data = x_clean
print("\nFiltering complete.")
print("Low-pass:", lp_cut, "Hz")
print("Notch:", harmonics, "Hz")
print("High-pass:", hp_cut, "Hz")
##
import matplotlib.pyplot as plt

ch = 4  
t = np.arange(x.shape[1]) / sfreq
plt.figure(figsize=(10, 4))
plt.plot(t, x[ch], label="Original", alpha=0.6)
plt.plot(t, x_clean[ch], label="Filtered", linewidth=1)
plt.xlim(0, 5)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.title("EEG Before vs After Filtering")
plt.legend()
plt.tight_layout()
plt.show()


plt.plot(t, x[ch] * 1e6, label="Original")
plt.plot(t, x_clean[ch] * 1e6, label="Filtered")
plt.ylabel("Amplitude (µV)")
plt.xlabel("Time (s)")
plt.xlim(0, 5)
plt.title("EEG Before vs After Filtering (µV)")
plt.legend()
plt.tight_layout()
plt.show()
montage = mne.channels.make_standard_montage('standard_1020')
raw_filt.pick_channels(raw_filt.ch_names)  # basically keeps all your current channels
raw_filt.set_montage(montage)
print(raw_filt.get_montage())

montage = mne.channels.make_standard_montage('standard_1020')
raw_filt.set_montage(montage)
raw_filt.plot_sensors(kind='topomap', show_names=True)


from mne.preprocessing import ICA
#ica filtering
raw_ica = raw_filt.copy()
raw_ica.filter(
    l_freq=1.0,
    h_freq=None,
    method="iir",
    verbose=True
)
ica = ICA(
    n_components=0.95,      
    method="infomax",
    random_state=97,
    max_iter="auto"
)
ica.fit(raw_ica)

ica.plot_components()
ica.plot_sources(raw_filt)
ica.plot_properties(raw_filt, picks=range(20))
eog_inds, eog_scores = ica.find_bads_eog(
    raw_filt,
    ch_name="Fp2",   # 👈 EOG proxy
    threshold=3.0
)
ica.plot_scores(eog_scores, exclude=eog_inds)
ica.plot_components(picks=eog_inds)
ica.exclude = eog_inds
print("ICA components marked for exclusion:", ica.exclude)
