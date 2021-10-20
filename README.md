# CommonFunctions

This is the first version of commonfunctions, some functions will be deleted since they are rarely used. I will eventually elaborate more on all of the fuctions.

## Convert
Used to convert seconds to an &lt;hours&gt;:&lt;minutes&gt;:&lt;seconds&gt; string.
  
## readtdms
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
   
## takefourier
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
  
## smooth
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
  
## propstort
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
