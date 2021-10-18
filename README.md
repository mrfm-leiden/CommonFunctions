# CommonFunctions

This is the first version of commonfunctions, some functions will be deleted since they are rarely used. I will eventually elaborate more on all of the fuctions.

## Convert
Used to convert seconds to an &lt;hours&gt;:&lt;minutes&gt;:&lt;seconds&gt; string.
  
## readtdms
Used to read a bunch of tdms files without the possibilty of getting an error in between.

## read_tdms
Function written by Timon to read tdms files.
  
## readcalcfft
Function that uses the read_tdms to read out a file, split the data in &lt;div&gt; pieces and calculate the fft of each piece. Not used a lot.
  
## takefourier
Function to take the fourier transform of data_tdms.
  
## calcenergy
Function that calculates the energy of the mode and the background energy next to the mode input.
  
## readblocks
Function to append the data from multiple tdms_files of len(tdms_block) 
  
## calcspectrablocks
Function that divides the files into blocks of &lt;size&gt; files, calculates the fft and saves a running average of the correlationmatrix.
  
## dftoa
Function to calculate the amplitude of the z mode as a function of f and df.
  
## smooth
Function to smooth the input array by dividing it into bits of size <binsize> and returning and array containing the average of the bits.
  
## propstort
Function to obtain the time array using real time from data and props.
