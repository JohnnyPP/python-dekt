import numpy as np 
import matplotlib.pyplot as plt
import math

filename = '/home/kolan/mycode/python/dektak/data/t10_1_1_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/t10_1_3_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/t10_1_6_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/t10_1_7_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/t10_1_3_parallel.csv'



def FindHeaderLength():
    """
    Finds the positionon the 'Scan Data' and adds additional 2 lines
    to give as a result the lenght of the header in number of lines.
    This is then used in csv function
    """

    lookup = 'Lateral um'
    
    with open(filename) as myFile:
        for FoundPosition, line in enumerate(myFile, 1):
            if lookup in line:
                print 'Scan Data found at line:', FoundPosition
                break
    
    return FoundPosition+4


x=np.loadtxt(filename,dtype=float,delimiter=',',skiprows=FindHeaderLength(),usecols=(0,))
y=np.loadtxt(filename,dtype=float,delimiter=',',skiprows=FindHeaderLength(),usecols=(1,))

coefficients = np.polyfit(x, y, 1)
polynomial = np.poly1d(coefficients)
ys = polynomial(x)
print coefficients
print polynomial

yLevelled=y-ys

plt.figure(1)
plt.plot(x,y)
plt.plot(x,ys)
plt.title('Raw data plot')
plt.xlabel('Lateral [um]')
plt.ylabel('Raw Micrometer [um]')
plt.grid(True)

plt.figure(2)
plt.title('Histogram of y')
n, bins, patches = plt.hist(y, 256, normed=1, facecolor='g', alpha=0.75)
plt.grid(True)

plt.figure(3)
d = np.diff(y)
plt.plot(d)
plt.title('Derivative of y')
plt.xlabel('Point []')
plt.ylabel('Raw Micrometer [um]')
plt.grid(True)

plt.figure(4)
plt.plot(x,yLevelled)
plt.title('Levelled data plot')
plt.xlabel('Lateral [um]')
plt.ylabel('Micrometer [um]')
plt.grid(True)

plt.figure(5)
plt.title('Histogram of yLevelled')
n, bins, patches = plt.hist(yLevelled, 256, normed=1, facecolor='g', alpha=0.75)
plt.grid(True)

dataLenght = len(yLevelled) 
xDiff = np.delete(x,dataLenght-1)   #diff consumes one last element from the array

plt.figure(6)
d = np.diff(y)
plt.plot(xDiff,d)
plt.title('Derivative of y')
plt.xlabel('Lateral [um]')
plt.ylabel('Raw Micrometer [um]')
plt.grid(True)

yLevelledMin = np.min(yLevelled)
yLevelledZeroShift = yLevelled - yLevelledMin

plt.figure(7)
plt.plot(x,yLevelledZeroShift)
plt.title('Levelled and shifted data plot')
plt.xlabel('Lateral [um]')
plt.ylabel('Micrometer [um]')
plt.grid(True)

##FFT###########################################################################

dataLenghtFFT = len(yLevelled)/2        #divide by 2 to satify rfft
                    # scale by the number of points so that
                    # the magnitude does not depend on the length 
                    # of the signal or on its sampling frequency 

calculatedFFT = np.fft.rfft(yLevelled) 
#calculatedFFT = np.fft.rfft(yLevelledZeroShift) 

amplitudeFFT = np.abs(calculatedFFT)    #calculates FFT amplitude from 
                                        #complex calculatedFFT output
phaseFFT = np.angle(calculatedFFT)      #calculates FFT phase from 
                                        #complex calculatedFFT output
phaseDegreesFFT = np.rad2deg(phaseFFT)  #convert to degrees
amplitudeScaledFFT = amplitudeFFT/float(dataLenghtFFT)
                 # scale by the number of points so that
                 # the magnitude does not depend on the length 
                 # of the signal
amplitudeScaledRMSFFT = amplitudeFFT/float(dataLenghtFFT)/math.sqrt(2)


# Scaling to Root mean square amplitude (dataLenghtFFT/sqrt{2}),
#############################################################################
# Plot the results
#############################################################################

xFFT = np.linspace(0,dataLenghtFFT+1,dataLenghtFFT+1)   
                                #the range is two times smaller +1 for RFFT
                                #sinus signal without noise used for fit

plt.figure("FFT amplitude and phase coefficients")
plt.subplot(2,1,1)
plt.vlines(xFFT,0,amplitudeScaledFFT)
plt.title("FFT amplitude coefficients")
plt.xlabel("Harmonics")
plt.ylabel("Amplitude [V]")
plt.xlim(0,dataLenghtFFT/2+1) #adjuts the x axis to maximum of numberOfPoints
plt.grid(True)

plt.subplot(2,1,2)
plt.vlines(xFFT,0,phaseDegreesFFT)
plt.title("FFT phase coefficients")
plt.xlabel("Harmonics")
plt.ylabel("Phase [deg]")
plt.tight_layout()      #removes the overlapping of the labels in subplots
plt.xlim(0,dataLenghtFFT+1)
plt.grid(True)


##############################################################################
##Moving average
##############################################################################
plt.figure('LevelledData with moving average ')
yLevelledMA = np.convolve(yLevelled, np.ones(10)/10)
plt.plot(yLevelled)
plt.hold(True)
plt.plot(yLevelledMA)
plt.title('Filtered levelled data plot')
plt.xlabel('Sample []')
plt.ylabel('Micrometer [um]')
plt.grid(True)

##orizontal line



diffMA = np.convolve(d, np.ones(10)/10)

dataLenghtDiff = len(d)
dataLenghtDiffMA = len(diffMA)

xLine = np.linspace(0,dataLenghtDiffMA,dataLenghtDiffMA)
yLine = np.linspace(0.05,0.05,dataLenghtDiffMA)  

plt.figure('Derivative with moving average')
plt.plot(d)
plt.hold(True)
plt.plot(diffMA)
plt.plot(yLine)
plt.title('Derivative with moving average')
plt.xlabel('Sample []')
plt.ylabel('Micrometer [um]')
plt.grid(True)



print dataLenghtDiff
print dataLenghtDiffMA



#thresholded = np.array(diffMA)
#x = np.where(thresholded == 0.05)[0]
#print x
#plt.figure('Derivative with moving average thresholded')
#plt.plot(thresholded)
#plt.title('Derivative with moving average')
#plt.xlabel('Sample []')
#plt.ylabel('Micrometer [um]')
#plt.grid(True)
#
#itemindex = np.where(diffMA > 0.05 and diffMA < 0.051)

plt.show()