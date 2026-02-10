import os
import matplotlib.pyplot as plt
import numpy as np
import mne
#
#ignore this
#########
data_dir = r"F:\EEGFILES"
vhdr_file = data_dir + r"\v13_session1_AML_8.11.2024.vhdr"
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
'''
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
print("Frequencies shape:", freqs.shape)'''

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



###start here###
####using excel file ###
import pandas as pd
eeg_data_type  = "EO_AML"

filebasedir = rf"F:\eegcode\files to test"
filename = os.path.join(filebasedir, r"V14_APL2_EyeOpen.xlsx")
data = pd.read_excel(filename)
eeg_data_raw = data.values.T
n_channels = eeg_data_raw.shape[0]
channel_names = data.columns.tolist()
print("Channel names:", channel_names)
print("EEG data shape:", eeg_data_raw.shape)
print(eeg_data_raw[:, :5])
sfreq = 500
info = mne.create_info(channel_names, sfreq, ch_types='eeg')        
if np.max( np.abs(eeg_data_raw) ) > 1:
    raw = mne.io.RawArray(eeg_data_raw , info)
else:
    raw = mne.io.RawArray(eeg_data_raw* 1e+6, info)
print(info)
''''
raw.set_channel_types({"Fp2": "eog"})
eog_events = mne.preprocessing.find_eog_events(raw)
print(eog_events[:5])
'''


# Filtering the data USING MNE
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import mne
print("Bad channels:", raw.info["bads"])   # channels marked bad
print(raw.info["sfreq"], "Hz")              # sampling frequency
print(raw.info["description"], "\n")        # acquisition info
print(raw.info)

raw_filt = raw.copy()
raw_filt.notch_filter(
    freqs=[50],  
    picks="data",
    method="iir",  
    filter_length="auto",
    phase="zero"
)

raw_filt.filter(
    l_freq=0.1,
    h_freq=35.0,
    picks="data",
    method="iir",
    phase="zero", 
    iir_params=dict(
        order=4,
        ftype="butter"
    )
)


'''
##FILTERING###
sfreq = raw.info["sfreq"]
nyq = sfreq / 2.0
x = raw.get_data()
powerline = 50.0     
Q = 30.0             
harmonics = [powerline, 2 * powerline]

for f in harmonics:
    if f < nyq:
        b, a = signal.iirnotch(w0=f / nyq, Q=Q)
        x_clean = signal.filtfilt(b, a, x, axis=1)
        
sfreq = raw.info["sfreq"]
nyq = sfreq / 2.0
lp_cut = 35.0  # Hz

sos_lp = signal.iirfilter(
    N=4,
    Wn=lp_cut / nyq,
    btype="lowpass",
    ftype="butter",
    output="sos"
)
x_lp = signal.sosfiltfilt(sos_lp, x_clean, axis=1)
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
'''
##ignore plots
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
raw_ica.set_eeg_reference('average', projection=False)
raw_ica.filter(
    l_freq=1.0,
    h_freq=None,
    verbose=True
)

ica = ICA(
    n_components=15,      
    method="infomax",
    random_state=42,
    max_iter=1000
)
ica.fit(raw_ica)
ica.plot_sources(raw_ica)

muscle_idx_auto, scores = ica.find_bads_muscle(raw)
ica.plot_scores(scores, exclude=muscle_idx_auto)
print(
    f"Automatically found muscle artifact ICA components: {muscle_idx_auto}"
)

muscle_idx = [0, 14]
raw_ica_clean = raw_filt.copy()
ica.plot_properties(raw_ica_clean, picks=muscle_idx, log_scale=True)

blink_idx = [0]
heartbeat_idx = [5]
ica.apply(raw_ica_clean, exclude=blink_idx + heartbeat_idx + muscle_idx)
ica.plot_overlay(raw_ica_clean, exclude=muscle_idx)


#raw_ica.set_eeg_reference('average', projection=False)  # for fitting

ica.plot_components()
ica.plot_sources(raw_filt)
ica.plot_properties(raw_filt, picks=range(20))
eog_inds, eog_scores = ica.find_bads_eog(
    raw_filt,
    ch_name="Fp2",   
    threshold=3.0
)
ica.plot_scores(eog_scores, exclude=eog_inds)
ica.plot_components(picks=eog_inds)
ica.exclude = eog_inds
print("ICA components marked for exclusion:", ica.exclude)
ica.apply(raw_ica_clean)
#raw_ica_clean.set_eeg_reference('average', projection=False)  # the data we'll apply ICA to
print(ica.n_iter_)



###plots i used to see the effect of ICA on the data no need to run all the time
ch_name = "Fp2"
ch_idx = raw_filt.ch_names.index(ch_name)
sfreq = raw_filt.info["sfreq"]
t = np.arange(raw_filt.n_times) / sfreq

x_before = raw_filt.get_data(picks=[ch_idx])[0]
x_after  = raw_ica_clean.get_data(picks=[ch_idx])[0]

plt.figure(figsize=(12, 4))
plt.plot(t, x_before * 1e6, label="Before ICA", alpha=0.6)
plt.plot(t, x_after  * 1e6, label="After ICA", linewidth=1)
plt.xlim(0, 10)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.title("Blink Artifact Removal (Fp2)")
plt.legend()
plt.tight_layout()
plt.show()
ica.plot_components(picks=eog_inds)

ch_name = "Fp1"
ch_idx = raw_filt.ch_names.index(ch_name)
sfreq = raw_filt.info["sfreq"]
t = np.arange(raw_filt.n_times) / sfreq

# Before & after ICA
x_before = raw_filt.get_data(picks=[ch_idx])[0]
x_after  = raw_ica_clean.get_data(picks=[ch_idx])[0]

plt.figure(figsize=(12, 4))
plt.plot(t, x_before * 1e6, label="Before ICA", alpha=0.6)
plt.plot(t, x_after  * 1e6, label="After ICA", linewidth=1)
plt.xlim(0, 10)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.title("Effect of Blink Removal on Fp1")
plt.legend()
plt.tight_layout()
plt.show()
#ignore
#Epoching the data
#Event IDs: {'Stimulus/S 30': 30, 'Stimulus/S 31': 31, 'Stimulus/S 32': 32, 'Stimulus/S 33': 33}
#>>> print("Number of events:", len(events))
#Number of events: 4
# choose the two events you want
'''
selected_event_ids = {
    'Stimulus/S 31': 31,
    'Stimulus/S 32': 32
}
events_sel = events[np.isin(events[:, 2], list(selected_event_ids.values()))]
epoch_len = 5.0
overlap = 0.5
step = epoch_len * (1 - overlap)   # 2.5 s

tmin = 0.0
tmax = epoch_len
sfreq = raw_ica_clean.info["sfreq"]

overlap_events = []

for ev in events_sel:
    onset = ev[0]          # sample index
    ev_id = ev[2]
    start = onset
    stop = onset + int(epoch_len * sfreq)

    while stop <= raw_ica_clean.n_times:
        overlap_events.append([start, 0, ev_id])
        start += int(step * sfreq)
        stop = start + int(epoch_len * sfreq)

overlap_events = np.array(overlap_events, dtype=int)
print("Number of overlapping epochs:", len(overlap_events))
'''
#epoching start
epoch_len = 5.0
overlap = 0.5
step = epoch_len * (1 - overlap)   # 2.5 s
tmin = 0.0
tmax = epoch_len
sfreq = raw_ica_clean.info["sfreq"]
epochs = mne.make_fixed_length_epochs(
    raw_ica_clean,
    duration=epoch_len,
    overlap=step,
    preload=True,
    reject_by_annotation=False
)
print(epochs)
print(epochs.event_id)
print("Number of epochs:", len(epochs))
print(overlap)

psds = epochs.compute_psd(
    method="welch",
    fmin=0.5,
    fmax=35,
    n_fft=1000,
    picks= ["O1", "O2", "P7", "P8"],
    n_overlap=500,
    verbose=True,
    window="hann",
    average="mean"
)#Mean over frequencies in band


####
import numpy as np
import pandas as pd

# Compute PSD
psds = epochs.compute_psd(
    method="welch",
    fmin=0.5,
    fmax=35,
    n_fft=1000,
    picks=["O1", "O2", "P7", "P8"],
    n_overlap=500,
    window="hann",
    average="mean",
    verbose=True
)

# Get PSD data and freqs
psd_data = psds.get_data()   # shape: (n_epochs, n_channels, n_freqs)
freqs = psds.freqs

lo, hi = 0.5, 4.0
freq_mask = (freqs >= lo) & (freqs <= hi)

assert freq_mask.any(), "No frequency bins found in the delta band"

channels = ["O1", "O2", "P7", "P8"]
n_epochs = psd_data.shape[0]

delta_power = []

for ep in range(n_epochs):
    row = {"epoch": ep}

    for ch_idx, ch_name in enumerate(channels):
        psd_curve = psd_data[ep, ch_idx, freq_mask]   # PSD vs freq
        band_power = np.trapz(psd_curve, freqs[freq_mask])

        row[f"{ch_name}_delta_Power"] = band_power

    delta_power.append(row)

df_delta = pd.DataFrame(delta_power)
df_delta.describe()
print(df_delta)
df_delta.to_csv("V13_delta_power.csv", index=False)

#ALL POWERS 
##
bands = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}

psd_data = psds.get_data()   # (n_epochs, n_channels, n_freqs)
freqs = psds.freqs

channels = ["O1", "O2", "P7", "P8"]
n_epochs = psd_data.shape[0]

rows = []

for ep in range(n_epochs):
    row = {"epoch": ep}

    for ch_idx, ch_name in enumerate(channels):
        for band_name, (lo, hi) in bands.items():
            freq_mask = (freqs >= lo) & (freqs <= hi)
            assert freq_mask.any(), f"No freq bins for {band_name}"

            psd_curve = psd_data[ep, ch_idx, freq_mask]
            band_power = np.trapz(psd_curve, freqs[freq_mask])

            row[f"{ch_name}_{band_name}_Power"] = band_power

    rows.append(row)

df_bands = pd.DataFrame(rows)
df_bands.to_csv("V14_band_powers_APL2.csv", index=False)

print(df_bands.describe())
####








bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta":  (13, 30),
   
}
# mean over epochs
psd_data = psds.get_data()   # (n_epochs, n_channels, n_freqs)
freqs = psds.freqs
results = {}
channels = ["O1", "O2", "P7", "P8"]
for ch_idx, ch in enumerate(channels):
    results[ch] = {}
    for band, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        
        # mean over epochs AND frequencies
        results[ch][band] = psd_data[:, ch_idx, idx].mean()
df_mean_psd = pd.DataFrame(results).T
print(df_mean_psd)

#PSD
####
bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta":  (13, 30),
}

psd_data = psds.get_data()   # (n_epochs, n_channels, n_freqs)
freqs = psds.freqs

channels = ["O1", "O2", "P7", "P8"]
results = {}

for ch_idx, ch in enumerate(channels):
    results[ch] = {}
    for band, (fmin, fmax) in bands.items():
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        assert freq_mask.any(), f"No freq bins for {band}"

        # mean over epochs AND frequencies
        results[ch][band] = psd_data[:, ch_idx, freq_mask].mean()

df_mean_psd = pd.DataFrame(results).T
df_mean_psd.to_csv("V14_mean_PSD_by_bandAPL1.csv")

print(df_mean_psd)
###






###ALL POWERS
import numpy as np
import pandas as pd

psd_data = psds.get_data()   # (n_epochs, n_channels, n_freqs)
freqs = psds.freqs

channels = ["O1", "O2", "P7", "P8"]

bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta":  (13, 30)
}

n_epochs = psd_data.shape[0]
band_power = []

for ep in range(n_epochs):
    row = {"epoch": ep}

    for band_name, (lo, hi) in bands.items():
        freq_mask = (freqs >= lo) & (freqs <= hi)

        # sanity check (same idea as your alpha code)
        assert freq_mask.any(), f"No frequency bins in {band_name} band"

        for ch_idx, ch_name in enumerate(channels):
            psd_curve = psd_data[ep, ch_idx, freq_mask]
            band_val = np.trapz(psd_curve, freqs[freq_mask])

            row[f"{ch_name}_{band_name}"] = band_val

    band_power.append(row)

df_bandpower = pd.DataFrame(band_power)

print(df_bandpower)
df_bandpower.describe()

df_bandpower.to_csv("V13_band_power_all_chann.csv", index=False)
