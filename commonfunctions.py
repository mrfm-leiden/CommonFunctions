import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import math
import datetime

from nptdms import TdmsFile
from tqdm.auto import trange

#------------------------------------------------------------------------------
"""Constants"""

pi = math.pi
mu_0 = 4*pi*10**-7 #
rho = 7600 #[kg/m^3]
g = 9.81 #[m/s^2]
Br = 0.7 #[T] remanent magnetisation
kb = 1.38*10**-23
T = 0.01
Phi_0 = 2E-15

# 99% of the time, this is how the 16 channel tdms files are set up.
guralplist = ["topx", "topy", "topz", 
              "celnx", "celny", "celnz", 
              "celex", "celey", "celez", 
              "1kx", "1ky", "1kz", 
              "ptsound", "magsig", "sq", "none"]

#------------------------------------------------------------------------------
"""Functions"""

def helpme():
    print("This file contains the functions we commonly use in our programs so we don't have to copy them in our notebooks")
    print("\nThe following functions can be used:")
    print("\tconvert(seconds)\t\t\t Convert seconds to <hours>:<minutes>:<seconds>")
    print("\treadtdms(filename, group, channel)\t Read tdms_files")
    print("\ttakefourier(data_tdms,props_tdms)\t Just calculate the fft of an array")
    print("\tsmooth(array,binsize)\t\t\t Smooth a 1D array by taking the average of every <binsize> entries")
    print("\tpropstort(data,props)\t\t\t Obtain the real time stamps from the data array")
    print("\nTo use these functions, type cf.<function>\nFor help, type help(cf.<function>)")
    

def convert(seconds: float) -> str: 
    """
    Function that converts seconds to <hours>:<minutes>:<seconds>
    """
    min, sec = divmod(seconds, 60) 
    hour, min = divmod(min, 60) 
    return "%d:%02d:%02d" % (hour, min, sec) 


def readtdms(filename: str, group: str = None, channel: str = None) -> tuple[dict, dict]:
    """
    Function that reads a tdms file and returns the data and properties.
    
    Parameters
    ----------
    filename : str
        The path + name of the file.
    group : str, optional
        The name of the group the data is requested from.
        The default is None, this means that the first group (and only group for 99.99999% of the time) in the file is chosen.
    channel : str, optional
        The name of the channel the data is requested from.
        The default is None, this means that all of the channels inside the given group will be returned.

    Returns
    -------
    data: dict
        If no channel was specified, the data will be a dictionary especially constructed for the 16 channel guralp data.
        If a channel was specified, the data will be a dict containing both the time and the data from that channel.
    props: dict
        The properties of the last read channel, will be the same for all channels inside a file.
    """
    tdms_file = TdmsFile(filename)
    if group == None:
        tdmsgroup = tdms_file.groups()[0]
    else:
        tdmsgroup = tdms_file[group]

    data = {}
    if channel == None:
        for _, tdmschannel in enumerate(tdmsgroup.channels()):
            if _ == 0:
                data['time'] = tdmschannel.time_track()
            data[guralplist[_]] = tdmschannel[:]
            props = tdmschannel.properties
    else:
        tdmschannel = tdmsgroup[channel]
        data['time'] = tdmschannel.time_track()
        data['data'] = tdmschannel[:]
        props = tdmschannel.properties

    return data, props


def takefourier(data: np.ndarray, props: dict, cutoff: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to take the fourier transform of tdms data after it is multiplied by a window-function.

    Parameters
    ----------
    data : np.ndarray
        The array containing the data the FFT needs to be calculated from.
    props : dict
        The properties dictionary containing the timesteps of <data>.
    cutoff : int, optional
        The cutoff point of the FFT in Hz.
        Default is 300 Hz since that was mainly used for the Levitated Zeppelin.

    Returns
    -------
    freqs: np.ndarray
        The array containing the frequencies (x-axis) for the FFT of <data>.
    fft: np.ndarray
        The array containing the FFT of <data>.
    """
    measure_freq = 1. / props["wf_increment"]
    
    freqs = np.fft.fftfreq(data.size, d=props["wf_increment"]) #x-as
    
    window = 2* np.pi*np.arange(len(data))/len(data)
    window = 0.5*(1-np.cos(window))

    data = data * window
 
    freq_res = measure_freq/len(data_tdms)
    
    #factor 1.63 from hanning window correction, y-axis  
    fft = np.fft.fft(data)[(freqs>=0)*(freqs<cutoff)]*1.63/(len(data)*np.sqrt(freq_res))

    # Frequencies corresponding to the fft data (between 0 and <cutoff> Hz). 
    freqs = freqs[(freqs>=0)*(freqs<cutoff)]
    return freqs, fft


def smooth(array: list, binsize: int) -> np.ndarray:
    """
    Function to smooth the input array by dividing it into bits of size <binsize> and
    returning and array containing the average of the bits.

    Warning: len(array) should be dividible by <binsize>

    Parameters
    ----------
    array : np.ndarray
        The array that needs to be smoothed.
    binsize : int
        The amount of data points the averages should be taking over.
    Returns
    -------
    np.ndarray
        The smoothed array of size len(array)/binsize
    """
    
    return np.mean(np.reshape(array, (int(len(array)/binsize), binsize)), axis = 1)


def propstort(data: np.ndarray, props: dict) -> list:   
    '''
    Function to obtain the time array using real time from data and props

    Parameters
    ----------
    data : np.ndarray
        The data array one wants the real time from.
    props : dict
        The properties dictionary that was made together with the data file.
    Returns
    -------
    np.ndarray
        The time array containing the time and date info
    '''
    dt64 = props['wf_start_time']
    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')
    seconds_since_epoch = (dt64 - unix_epoch) / one_second
    customdate = datetime.datetime.utcfromtimestamp(seconds_since_epoch)
    return [customdate + datetime.timedelta(seconds=i*props['wf_increment']) for i in range(len(data))]


if __name__ == '__main__':
    helpme()

