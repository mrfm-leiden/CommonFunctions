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


#------------------------------------------------------------------------------
"""Functions"""

def helpme():
    print("This file contains the functions we commonly use in our programs so we don't have to copy them in our notebooks")
    print("\nThe following functions can be used:")
    print("\tconvert(seconds)\t\t\t Convert seconds to <hours>:<minutes>:<seconds>")
    print("\tread_tdms(filename)\t\t\t Read tdms_files")
    print("\ttakefourier(data_tdms,props_tdms)\t Just calculate the fft of an array")
    print("\tsmooth(array,binsize)\t\t\t Smooth a 1D array by taking the average of every <binsize> entries")
    print("\tpropstort(data,props)\t\t\t Obtain the real time stamps from the data array")
    print("\nTo use these functions, type cf.<function>\nFor help, type help(cf.<function>)")
    

def convert(seconds): 
    """
    Function that converts seconds to <hours>:<minutes>:<seconds>
    """
    min, sec = divmod(seconds, 60) 
    hour, min = divmod(min, 60) 
    return "%d:%02d:%02d" % (hour, min, sec) 


def readtdms(tdms_file_loc, groupname = 'Group', channelname = 'Channel'):
	"""
	Function used by Jaimy to read a bunch of tdms files without getting an error in between
	"""
    tdms_file = TdmsFile.read(tdms_file_loc)
    try:
        group = tdms_file[groupname]
        channel = group[channelname]
        data = channel[:]
        props = channel.properties
        return data, props
    except:
        return np.nan, np.nan	
	

def read_tdms(filename, group_name='Group', channel_name='Channel'):
    """
    function made by Timon that reads tdms files and returns the data inside the file
    inputs:
        filename: Str that contains path + filename.type
        group_name: The name of the group inside the tdms that needs to be read
        channel_name: The name of the channel inside the tdms file that needs to be read
    ouputs:
        data: An array containing the data inside the chosen channel
        props: properties of the data inside the channel, includes the data+time the file was made and the step size (and more)
    """
    # See https://readthedocs.org/projects/nptdms/downloads/pdf/latest/
    # 
    # Code to find group and channel names:
    # tdms_file = TdmsFile.read("my_file.tdms")
    # all_groups = tdms_file.groups()
    
    # group = tdms_file["group name"]
    # all_group_channels = group.channels()
    
    # channel = group["channel name"]
    
    # To obtain time data
    # time = np.arange(0, props['wf_increment']*len(data), props['wf_increment'])
    
    # time = channel.time_track()
    
    # Read .tdms file
    tdms_file = TdmsFile(filename)
    
    if group_name == 'unknown':
        # If group and channel name are UNKNOWN (indicated by zero): 
        # Only take data from the first group and the first channel in this group!
        # SO BE AWARE THAT THIS FUNCTION ONLY READS THE FIRST GROUP AND CHANNEL. 
        group = tdms_file.groups()[0]
        group_name = group.name

        channel = group.channels()[0]
        channel_name = channel.name
            
    # Read data from the specified group and channel
    channel_object = tdms_file.object(group_name, channel_name)
    data = channel_object.data
    props = channel_object.properties
    
    return data, props


def takefourier(data: numpy.ndarray, props: dict, cutoff: int = 300) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Function to take the fourier transform of tdms data.
    inputs:
        data: 1D array to take the fourier transform of.
        props: properties of <data>, used to find the time increments of <data>.
    outputs:
        freqs: 1D array containing the frequencies of the fft (starts at 0 and ends before cutoff).
        fft: 1D array of the fft of <data>.
    """
    
    measure_freq = 1. / props["wf_increment"]
    
    freqs = np.fft.fftfreq(data.size, d=props["wf_increment"]) #x-as
    
    window = 2* np.pi*np.arange(len(data))/len(data)
    window = 0.5*(1-np.cos(window))

    data = data * window

    # Only save the spectrum between 0 and 300 Hz.  
    freq_res = measure_freq/len(data_tdms)
    
    #factor 1.63 from hanning window correction, y-as    
    fft = np.fft.fft(data)[(freqs>=0)*(freqs<cutoff)]*1.63/(len(data)*np.sqrt(freq_res))

    # Frequencies corresponding to the fft data (between 0 and <cutoff> Hz). 
    freqs = freqs[(freqs>=0)*(freqs<cutoff)]
    return freqs, fft


def smooth(array: list, binsize: int) -> numpy.ndarray:
    """
    Function to smooth the input array by dividing it into bits of size <binsize> and
    returning and array containing the average of the bits.
    inputs:
        array: 1D array to smooth
        binsize: int, the amount of data points the averages should be taking over
    outputs:
        The smoothed array of size len(array)/binsize 
    """
    
    return np.mean(np.reshape(array, (int(len(array)/binsize), binsize)), axis = 1)


def propstort(data: numpy.ndarray, props: dict) -> list:
    '''
    Function to obtain the time array using real time from data and props
    inputs:
        data: the data array from the tdms file
        props: the props array from the tdms file
    outputs: time array containing the time and date info
    '''
    dt64 = props['wf_start_time']
    unix_epoch = np.datetime64(0, 's')
    one_second = np.timedelta64(1, 's')
    seconds_since_epoch = (dt64 - unix_epoch) / one_second
    customdate = datetime.datetime.utcfromtimestamp(seconds_since_epoch)
    return [customdate + datetime.timedelta(seconds=i*props['wf_increment']) for i in range(len(data))]