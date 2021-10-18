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
    print("\treadcalcfft(tdms_files,div)\t\t Uses read_tdms and calculates the fft")
    print("\ttakefourier(data_tdms,props_tdms)\t Just calculate the fft of an array")
    print("\tcalcenergy(freqs,fft,mode,df)\t\t Calculate the energy of a mode")
    print("\treadblocks(tdms_block)\t\t\t Reads the directory in a block, used by calcspectrablocks")
    print("\tcalcspectrablocks(tdms_files,size)\t Read a lot of files in blocks of <size> and calculate correlation")
    print("\tdftoa(f,df,magnet)\t\t\t Calculate the amplitude of the mode from the frequencyshift")
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


def readcalcfft(tdms_files, div):
    """
    function that uses the read_tdms to read out a file, split the data in <div> 
    pieces and calculate the fft of each piece.
    inputs:
        tdms_files: Str that contains path + filename.type
        div: The amount of pieces the data array has to be split into
    outputs: 
        freqs_25min: 1D array containing the frequencies of the fft (starts at 
                     0 and ends before 300Hz)
        fft_25min: 2D array of size (len(tdms_files)*div, len(freqs_25min))
    """
    
    for i in trange(len(tdms_files), file=sys.stdout):
        # Read .tdms files 
        data_tdms, props_tdms = read_tdms(str(tdms_files[i]), 'Group', 'Channel')

        for k in range(div): 
            temparray = data_tdms[int(len(data_tdms)/div)*k:int(len(data_tdms)/div)*(k+1)]

            measure_freq = 1. / props_tdms["wf_increment"]

            # For the first iteration: calculate the array with frequencies that belongs to the FT of the
            # data. Also, create an array in which the FT between 0 and 300 Hz is stored for every dataset of 25 minutes
            if i==0:
                freqs_25min = np.fft.fftfreq(temparray.size, d=props_tdms["wf_increment"])
                fft_25min = np.zeros((len(tdms_files)*div, \
                                      len(temparray[(freqs_25min>=0)*(freqs_25min<300)])))
                window = 2* np.pi*np.arange(len(temparray))/len(temparray)                        # (only the first time in the for loop)  create the Hanning or Hamming window
                window = 0.5*(1-np.cos(window))
            freq_res = measure_freq/len(temparray)
            temparray = temparray * window
                    # here the data is multiplied with the window
            # Calculate the FFT for the 25 minutes datasets. Only save the spectrum between 0 and 300 Hz. 
            fft_25min[i*div+k, :] = np.abs(np.fft.fft(temparray))[(freqs_25min>=0)*(freqs_25min<300)]*1.63/(len(temparray)*np.sqrt(freq_res))


    # Frequencies corresponding to the fft data (between 0 and 300 Hz). 
    freqs_25min = freqs_25min[(freqs_25min>=0)*(freqs_25min<300)]
    return freqs_25min, fft_25min


def takefourier(data_tdms, props_tdms, cutoff = 300):
    """
    Function to take the fourier transform of data_tdms
    inputs:
        data_tdms: 1D array to take the fourier transform of
        props_tdms: properties of data_tdms, used to find the time increments of data_tdms
    outputs:
        freqs_25min: 1D array containing the frequencies of the fft (starts at 0 and ends before 300Hz)
        fft_25min: 1D array of the fft of data_tdms
    """
    
    measure_freq = 1. / props_tdms["wf_increment"]
    
    # For the first iteration: calculate the array with frequencies that belongs to the FT of the
    # data. Also, create an array in which the FT between 0 and 300 Hz is stored for every dataset of 25 minutes
    
    freqs_25min = np.fft.fftfreq(data_tdms.size, d=props_tdms["wf_increment"]) #x-as
    
    window = 2* np.pi*np.arange(len(data_tdms))/len(data_tdms)                        # (only the first time in the for loop)  create the Hanning or Hamming window
    window = 0.5*(1-np.cos(window))

    data_tdms = data_tdms * window                                                        # here the data is multiplied with the window
    # Calculate the FFT for the 25 minutes datasets. Only save the spectrum between 0 and 300 Hz.
        
    freq_res = measure_freq/len(data_tdms)
        
    fft_25min = np.fft.fft(data_tdms)[(freqs_25min>=0)*(freqs_25min<cutoff)]*1.63/(len(data_tdms)*np.sqrt(freq_res)) #factor 1.63 from hanning window correction, y-as

        
    # Frequencies corresponding to the fft data (between 0 and 300 Hz). 
    freqs_25min = freqs_25min[(freqs_25min>=0)*(freqs_25min<cutoff)]
    return freqs_25min, fft_25min



def calcenergy(freqs, fft, mode, df):
    """
    function that calculates the energy of the mode and the background energy next to the mode input
    inputs:
        freqs: 1D array with frequencies of the fft
        fft: 2D array with the Fourier Transform of the data
        mode: Frequency of the mode of interest
        df: Bandwidth of the energy calculation
    ouputs:
        mode_energy: 1D array containing the energy of the mode for every row of freqs
        bg_energy: 1D array containing the energy of the background for every row of freqs 
    """
    pltnrs = np.arange(len(fft.shape[0]))
    
    freqres = freqs[1]-freqs[0]
    
    colors=['blue', 'red', 'green', 'black', 'orange', 'gold']
    
    fig, axs = plt.subplots(nrows=1, sharex=True, figsize=(18, 8))

    start_index = int(round((mode1-df_mode1)/(freqres)))   # starting index around mode1    
    end_index   = int(round((mode1+df_mode1)/(freqres)))   # end index around mode1
    #total_index = end_index-start_index
    
    df_bg=-0.1                     # how far displaced is the place where you determine the background
    start_index_bg = int(round(start_index - (df_bg/freqres)))
    
    for i in plotnrs:  #range(len(tdms_files)):
        if i==0: 
            mode1_energy = np.zeros(len(plotnrs))          # fill an empty array with zeros for the calculated mode energy
            mode1_bg_energy = np.zeros(len(plotnrs))       # idem for the background noise just before the mode freq
            data_nr = np.arange(len(plotnrs))              # this is the horizontal scale for future plots

        axs.plot(freqs, fft[i,:], linewidth=.5)# Hier stond freqs_25min_temp,  label=time_labels[i] + ' ' + date_labels[i])   #plot spectrum
        mode_energy= fft[i,start_index:end_index]                                     # make selection around the mode frequency
        mode_energy_temp = mode_energy
        mode_energy = mode_energy * mode_energy                                             # and square it
        mode1_energy[i]=np.sum(mode_energy)  *(freqres)                                               # sum the datapoints  (the square root of this total is the rms voltage corresponding to the mode motion)
        axs.plot([mode1-df_mode1,mode1+df_mode1], [np.sqrt(mode1_energy[i]/total_index),np.sqrt(mode1_energy[i]/total_index)])   
                                                          # plot a horizontal line at the average of the points in units V/sqrtHz, that is why we divide by total_index, which is the number of points
        background_energy = fft[i,start_index_bg:start_index_bg+total_index]          # make a selection before the mode. Make sure it has the same number of points as the integration around the mode freq
        background_energy = background_energy * background_energy                           # and square it
        mode1_bg_energy[i] = np.sum(background_energy) *(freqres)
        axs.plot([mode1-df_mode1-df_bg,mode1+df_mode1-df_bg], [np.sqrt(mode1_bg_energy[i]/total_index),np.sqrt(mode1_bg_energy[i]/total_index)])   # plot a little bar at th eprooer V/rtHz

    axs.set_ylabel('Amplitude')
    axs.set_yscale('log')
    #axs.set_ylim(0.1, 1e6)
    axs.legend()

    axs.set_xlabel('Frequency (Hz)')
    axs.set_xlim(mode1-10*df_mode1, mode1+10*df_mode1)                                                                 # Set the range you want to display here
    plt.title('Show where the energy is calculated from')

    plt.tight_layout()
    plt.show()
    
    return mode_energy, bg_energy


def readblocks(tdms_block):
    """
    Function to append the data from multiple tdms_files of len(tdms_block) 
    inputs: 
        tdms_block: array of paths to mulitple tdms_files
    outputs:
        data_tdms: time data of multiple files appended to eachother
        props_tdms: the properties of the tdms_files
    """
    props_tdms = []
    
    for i in trange(len(tdms_block), file=sys.stdout):
        
        # Read .tdms files         
        data_tdms_temp, props_tdms_temp = read_tdms(str(tdms_block[i]))
        if i == 0:
                data_tdms = data_tdms_temp
        else:       
            data_tdms = np.append(data_tdms, data_tdms_temp, axis = 1)
        props_tdms.append(props_tdms_temp)
    
    return data_tdms, props_tdms


def calcspectrablocks(tdms_files, size = 20):
    """
    Function that divides the files into blocks of <size> files, calculates the fft and saves a running average of the correlationmatrix
    inputs:
        tdms_files: Str that contains path + filename.type
        size: The size of the blocks in which the files should be read
    outputs:
        freqs_25min: 1D array containing the frequencies of the fft (starts at 
                     0 and ends before 300Hz)
        fft_array: 2D array of size (len(tdms_files)*div, len(freqs_25min))
        fft_array_conj: conjugated fft_array
        correlationmatrix: 3D array of size (16, 16, len(freqs_25min))
    """
    
    #save time it took for this function to run
    start_time = time.time()
    
    for i in range(int(len(tdms_files)/size)):
        print("Iteration ", i+1, " of ", int(len(tdms_files)/size))         
        
        # Make blocks of files
        tdms_block = tdms_files[size*i:size*(i+1)]                            # Verdeel alle files in blokken van vooraf gekozen size
        
        # For every block, read the 16 channels and save their FFT
        data_tdms, props_tdms = readblocks(tdms_block)                        # Lees de files in het blok uit
        
        for k in range(16):
            freqs_25min, fft_25min = takefourier(data_tdms[k,:], props_tdms)  # Neem van alle kanalen de fourier getransformeerde
            if k == 0:
                fft_array_new = fft_25min                                   
            else:
                fft_array_new = np.vstack((fft_array_new, fft_25min))
                
        fft_array_conj_new = np.conj(fft_array_new)                           # Neem de complex geconjugeerde en sla die op in een aparte array
        
        # Fill in correlationmatrix
        correlationmatrix_new = np.zeros((16,16,fft_array_new.shape[1]), dtype = "complex_")
        for n in range(16):
            for m in range(16):
                if n >= m:
                    correlationmatrix_new[n,m,:] = fft_array_new[n,:] * fft_array_conj_new[m,:]
        
        # Save the new Fourier spectra as a running average
        if i == 0:
            fft_array = fft_array_new
            fft_array_conj = fft_array_conj_new
            correlationmatrix = correlationmatrix_new
        else:
            fft_array = (fft_array*i+fft_array_new)/(i+1) 
            fft_array_conj = (fft_array_conj*i+fft_array_conj_new)/(i+1)
            correlationmatrix = (correlationmatrix*i+correlationmatrix_new)/(i+1)
        
            
    print("It took ", convert(time.time()-start_time), " to run this program")
    return freqs_25min, fft_array, fft_array_conj, correlationmatrix


def dftoa(f, df, magnet):
    """
    Function to calculate the amplitude of the z mode as a function of f and df
    inputs:
        f: frequency of the mode
        df: frequency deviation due to large amplitudes
        magnet: the class of the magnet the amplitude should be calculated for (Huron, Superior)
    outputs:
        Amplitude of the mode
    """
    return magnet.z0*np.sqrt((48*df)/(35*f))


def smooth(array, binsize):
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


def propstort(data, props):
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