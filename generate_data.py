import os, sys
os.environ["DISPLAY"] = "1.0"

import h5py
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

import pyfstat
logger = pyfstat.set_up_logger(outdir="pyfstat_log", label="1_generating_signals", log_level="WARNING")


if __name__ == "__main__":

    # Read arguments:
    msg = "Generate dataset containing continuous gravitational wave data.\n"
    msg += "Example usage:\n"
    msg += "> python generate_data.py --sensitivity 10.0 --num_signals 1000"
    parser = argparse.ArgumentParser(description=msg, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--sensitivity', required=True, type=float)  
    parser.add_argument('--num_signals', default='1000', type=int)

    args = parser.parse_args()  

    sensitivity = args.sensitivity
    num_signals = args.num_signals

    out_dir = "./data/"
    dataset_name = f"data_sensitivity_{int(sensitivity)}"

    print("=============================================")
    print(f"Generating dataset {dataset_name} with sensitivity {sensitivity} and {num_signals} files.\n")

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    if not os.path.isdir(os.path.join(out_dir, dataset_name)):
        os.mkdir(os.path.join(out_dir, dataset_name))

    
    # Parameters for Pyfstat:

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
            "h0": writer_kwargs["sqrtSX"]/sensitivity,
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
        writer_kwargs["outdir"] = f"Generated_SFT_data_tmp/Signal_{i_signal}"
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
            
        
    # save labels as csv file:
    df.to_csv(out_dir+dataset_name+"_labels.csv", index=False)

    print()
    print(f"Data saved to {os.path.join(out_dir, dataset_name)}")

    print()
    print("Deleting temporary SFT data in Generated_SFT_data_tmp/")
    os.system("rm -r Generated_SFT_data_tmp/")
    