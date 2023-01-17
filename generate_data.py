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



# Generate signals with parameters drawn from a specific population

sensitivity = 35.0  # try with 1, 20, 50

num_signals = 1000

out_dir = "/Users/lucas/Documents/Data/G2Net/"
dataset_name = "data_synth_sensitivity_35"

print(f"Generating dataset with sensitivity: {sensitivity}")

if not os.path.isdir(os.path.join(out_dir, dataset_name)):
    os.mkdir(os.path.join(out_dir, dataset_name))

# These parameters describe background noise and data format
writer_kwargs = {
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

# Sample signal parameters from a specific population:
signal_parameters_generator = pyfstat.AllSkyInjectionParametersGenerator(
    priors={
        "tref": writer_kwargs["tstart"],
        "F1": lambda: -10**stats.uniform(-12, 4).rvs(),
        "F2": 0,
#         "h0": lambda: writer_kwargs["sqrtSX"] / stats.uniform(1, 10).rvs(),
        "h0": writer_kwargs["sqrtSX"]/sensitivity,
#         "h0": lambda: (writer_kwargs["sqrtSX"]/sensitivity) * stats.bernoulli(0.5).rvs(),  # randomly zero
        **pyfstat.injection_parameters.isotropic_amplitude_distribution,
    },
)




columns = ['id', 'target', 'SNR', 'sqrtSX', 'F0', 'F1', 'h0', 'alpha', 'delta', 'psi', 'phi', 'cosi']
df = pd.DataFrame(columns=columns, index=None)

print(columns[:7])

i_signal = 0

while i_signal < num_signals:

    file_id = f"{i_signal:05}"
    
    # Draw signal parameters.
    params = signal_parameters_generator.draw()
    writer_kwargs["outdir"] = f"Generated_data/Signal_{i_signal}"
    writer_kwargs["label"] = f"Signal_{i_signal}"
    
    target = stats.bernoulli(0.5).rvs()  # 0 or 1
    params["h0"] *= target               # set h0 randomly to zero to generate only noise
    
    writer = pyfstat.Writer(**writer_kwargs, **params)
    try:
        writer.make_data()
    except:
        print(f"couldn't write {file_id}. Trying again with new set of parameters")
        continue  # do not increment iterator
    
    # SNR can be compute from a set of SFTs for a specific set
    # of parameters as follows:
    snr_ = pyfstat.SignalToNoiseRatio.from_sfts(F0=writer.F0, sftfilepath=writer.sftfilepath)
    
    squared_snr = snr_.compute_snr2(
        Alpha=writer.Alpha, 
        Delta=writer.Delta,
        psi=writer.psi,
        phi=writer.phi, 
        h0=writer.h0,
        cosi=writer.cosi
    )
    
    SNR = np.sqrt(squared_snr)
    
    # Data can be read as a numpy array using PyFstat
    frequency, timestamps, amplitudes = pyfstat.utils.get_sft_as_arrays(writer.sftfilepath)
    
#     columns = ['file_id', 'target', 'SNR', 'sqrtSX', 'F0', 'F1', 'h0', 'alpha', 'delta', 'psi', 'phi', 'cosi']
    vals = [file_id, target, SNR, writer.sqrtSX, writer.F0, writer.F1, writer.h0, writer.Alpha, writer.Delta, writer.psi, writer.phi, writer.cosi]
    df.loc[i_signal] = vals
    print(vals[:7])
    
    plot = False
    save = True
    
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(16, 10), sharey=True)
        fig.suptitle(f"SNR = {SNR:.2f}")

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
    print(f"Data saved to {out_dir+dataset_name}")

    print()
    print("Deleting data in Generated_data/")
    os.system("rm -r Generated_data/")
    