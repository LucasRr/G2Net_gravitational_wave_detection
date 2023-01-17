import os
os.environ["DISPLAY"] = "1.0"

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

import h5py

import pyfstat
from pyfstat.utils import get_sft_as_arrays
from PyFstat.examples.tutorials import tutorial_utils

logger = pyfstat.set_up_logger(outdir="pyfstat_log", label="1_generating_signals", log_level="WARNING")



def generate_noise_profile(signal_length):

    noise_profile = {}

    for detector in ["H1", "L1"]:

        carrier = np.zeros(signal_length)
        carrier[:signal_length] = np.repeat(np.random.randn(20), signal_length//20)

        noise_profile[detector] = 1 + 0.1*carrier + 0.1*np.random.randn(signal_length)

    return noise_profile



# Generate signals with parameters drawn from a specific population

num_signals = 1000

out_dir = "/Users/lucas/Documents/Data/G2Net/"
dataset_name = "data_synth_realistic_noise_1_5"

if not os.path.isdir(os.path.join(out_dir, dataset_name)):
    os.mkdir(os.path.join(out_dir, dataset_name))

# These parameters describe background noise and data format
writer_kwargs_noise_only = {
    "label": "single_detector_gaussian_noise",
    "outdir": "Generated_data",
    "tstart": 1238166018,  # Starting time of the observation [GPS time]
    "duration": 120 * 86400,  # Duration [seconds]
    "detectors": "H1,L1",  # Detector to simulate, in this case LIGO Hanford
    "F0": 400.0,  # Central frequency of the band to be generated [Hz]
    "Band": 0.2,  # Frequency band-width around F0 [Hz]
    "sqrtSX": 1e-23,  # Single-sided Amplitude Spectral Density of the noise
    "Tsft": 1800,  # Fourier transform time duration
    "SFTWindowType": "tukey",  # Window function to compute short Fourier transforms
    "SFTWindowBeta": 0.01,  # Parameter associated to the window function
}

writer_kwargs_noisy_signal = {
    "label": "single_detector_gaussian_noise",
    "outdir": "Generated_data",
    "tstart": 1238166018,  # Starting time of the observation [GPS time]
    "duration": 120 * 86400,  # Duration [seconds]
    "detectors": "H1,L1",  # Detector to simulate, in this case LIGO Hanford
    "F0": 400.0,  # Central frequency of the band to be generated [Hz]
    "Band": 0.2,  # Frequency band-width around F0 [Hz]
    "sqrtSX": 0,  # Single-sided Amplitude Spectral Density of the noise
    "Tsft": 1800,  # Fourier transform time duration
    "SFTWindowType": "tukey",  # Window function to compute short Fourier transforms
    "SFTWindowBeta": 0.01,  # Parameter associated to the window function
}

# Sample signal parameters from a specific population:
signal_parameters_generator = pyfstat.AllSkyInjectionParametersGenerator(
    priors={
        "tref": 1238166018,
        "F1": lambda: -10**stats.uniform(-12, 4).rvs(),
        "F2": 0,
        "h0": lambda: 1e-23/stats.uniform(1, 5).rvs(),
        **pyfstat.injection_parameters.isotropic_amplitude_distribution,
    },
)



columns = ['id', 'target']
df = pd.DataFrame(columns=columns, index=None)

i_signal = 0

while i_signal < num_signals:

    file_id = f"{i_signal:05}"

    ####### Generate noise only signal:

    # Draw signal parameters.
    writer_kwargs_noise_only["outdir"] = f"Generated_data/Noise_{i_signal}"
    writer = pyfstat.Writer(**writer_kwargs_noise_only)
    writer.make_data()

    frequency, timestamps, amplitudes_noise = pyfstat.utils.get_sft_as_arrays(writer.sftfilepath)

    ####### Generate clean signal:
    
    # Draw signal parameters.
    signal_params = signal_parameters_generator.draw()

    # Draw signal parameters.
    writer = pyfstat.Writer(**writer_kwargs_noisy_signal, **signal_params)
    try:
        writer.make_data()
    except:
        print(f"couldn't write {file_id}. Trying again with new set of parameters")
        continue  # do not increment iterator

    frequency, timestamps, amplitudes_signal = pyfstat.utils.get_sft_as_arrays(writer.sftfilepath)

    ###### generate realistic noise profile:

    noise_profile = generate_noise_profile(amplitudes_signal['L1'].shape[1])

    ###### sum up signal and noise:

    target = stats.bernoulli(0.5).rvs()  # 0 or 1

    amplitudes = {}
    for detector in ["H1", "L1"]:
        amplitudes[detector] = amplitudes_noise[detector] * noise_profile[detector]
        + amplitudes_signal[detector] * target


    df.loc[i_signal] = [file_id, target]
    print(f"generating {file_id}")
    
    plot = False
    save = True
    
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(16, 10), sharey=True)

        for i, detector in enumerate(("H1", "L1")):
            time_in_days = (timestamps[detector] - timestamps[detector][0]) / (24 * 3600)

            c = axs[i].pcolorfast(time_in_days, frequency, np.absolute(amplitudes[detector]))
            fig.colorbar(c, ax=axs[i], orientation="horizontal")
        
    if save:
        
        # save hdf5 files:
        out_file = os.path.join(out_dir, dataset_name, file_id+".hdf5")
        with h5py.File(out_file, "w") as f:

            H1 = f.create_group(file_id+'/H1')
            L1 = f.create_group(file_id+'/L1')

            f.create_dataset(file_id+'/frequency_Hz', data=frequency)

            H1.create_dataset('SFTs', data=amplitudes["H1"])
            H1.create_dataset('timestamps_GPS', data=timestamps["H1"])  

            L1.create_dataset('SFTs', data=amplitudes["L1"])
            L1.create_dataset('timestamps_GPS', data=timestamps["L1"])  
                        
    i_signal += 1
        
    
if save:
    # save csv files:
    df.to_csv(out_dir+dataset_name+"_labels.csv", index=False)

    print()
    print("Deleting data in Generated_data/")
    os.system("rm -r Generated_data/")


if plot:
    plt.show()


    