import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as scipy
import sys
import math
from numpy import NaN, Inf, arange, isscalar, asarray, array

filename = '/home/kolan/mycode/python/dektak/data/t10_1_1_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/data/t10_1_3_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/data/t10_1_6_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/data/t10_1_7_normal.csv'    #first peak very good    
#filename = '/home/kolan/mycode/python/dektak/data/t10_1_12_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/data/t10_1_15_normal.csv'   #
#filename = '/home/kolan/mycode/python/dektak/data/t10_1_19_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/data/t10_1_21_normal.csv'   #no top & bottom
#filename = '/home/kolan/mycode/python/dektak/data/t10_1_24_normal.csv'  #no top & bottom
#filename = '/home/kolan/mycode/python/dektak/data/t10_1_3_parallel.csv'  #no top & bottom
#filename = '/home/kolan/mycode/python/dektak/data/t10_1_15_parallel.csv'
#filename = '/home/kolan/mycode/python/dektak/data/t10_1_19_parallel.csv'  #0.035 too low 0.04 ok
#filename = '/home/kolan/mycode/python/dektak/data/t10_1_24_parallel.csv' #first peak very good 
#filename = '/home/kolan/mycode/python/dektak/data/t10_3_1_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/data/t10_3_3_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/data/t10_3_6_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/data/t10_3_7_normal.csv'    #short peak
#filename = '/home/kolan/mycode/python/dektak/data/t10_3_15_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/data/t10_3_19_normal.csv'

def FindHeaderLength():
    """
    Finds the positionon the 'Scan Data' and adds additional 4 lines
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


def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)


def coefA(x0,y0,x1,y1):
    return -(y1-y0)/(x1-x0)


def coefC(x0,y0,x1,y1):
    return (x1*y0-x0*y1)/(x1-x0)


def FindThresholdLine(x,y,threshold):
    thresholdLineArray = np.array([0,1,threshold])
    for i in xrange(0, len(y)):
        if i < (len(y)-1):
            if y[i]<threshold and y[i+1]>threshold: #incerasing
                #print "Increasing line detected"
                x0 = x[i]
                y0 = y[i]
                x1 = x[i+1]
                y1 = y[i+1]
                coefAinc = coefA(x0,y0,x1,y1)
                coefCinc = coefC(x0,y0,x1,y1)
                incPointsCrossingThreshold = np.array([x0,y0,x1,y1])
                lineEquationCooficientsCrossongThr = np.array([coefAinc, 1, coefCinc])
                increasingPoints.append(incPointsCrossingThreshold)
                incLineEquationCoefficients.append(lineEquationCooficientsCrossongThr)
                
            else:
                if y[i]>threshold and y[i+1]<threshold: #decreasing
                    #print "Decreasing line detected"
                    x0 = xDiff[i]
                    y0 = y[i]
                    x1 = xDiff[i+1]
                    y1 = y[i+1]
                    coefAdec = coefA(x0,y0,x1,y1)
                    coefCdec = coefC(x0,y0,x1,y1)
                    decPointsCrossingThreshold = np.array([x0,y0,x1,y1])
                    lineEquationCooficientsCrossongThrDec = np.array([coefAdec, 1, coefCdec])
                    decreasingPoints.append(decPointsCrossingThreshold)
                    decLineEquationCoefficients.append(lineEquationCooficientsCrossongThrDec)
                #else:
                    #print "Neither dereasing nor incereasing line"
                
        else:
            break
    
    A2=thresholdLineArray[0]
    B2=thresholdLineArray[1]
    C2=thresholdLineArray[2]
    
    for i in xrange(0, incLineEquationCoefficients.__len__()):
        A1=incLineEquationCoefficients[i][0]
        B1=incLineEquationCoefficients[i][1]
        C1=incLineEquationCoefficients[i][2]
    
        detW = float(A1*B2-A2*B1)
        detWx = float((C1)*B2-(C2)*B1)
        detWy = float(A1*(C2)-A2*(C1))
        pointX = float(detWx/detW)
        pointY = float(detWy/detW)
        incIntersectionPoints.append(np.array([pointX,pointY]))
        #plt.plot(xDiff[i],incIntersectionPoints[i][1],'go')
        plt.plot(incIntersectionPoints[i][0],incIntersectionPoints[i][1],'go')
        
    for i in xrange(0, decLineEquationCoefficients.__len__()):
        decA1=decLineEquationCoefficients[i][0]
        decB1=decLineEquationCoefficients[i][1]
        decC1=decLineEquationCoefficients[i][2]
         
        decdetW = float(decA1*B2-A2*decB1)
        decdetWx = float((decC1)*B2-(C2)*decB1)
        decdetWy = float(decA1*(C2)-A2*(decC1))
        decpointX = float(decdetWx/decdetW)
        decpointY = float(decdetWy/decdetW)
        decIntersectionPoints.append(np.array([decpointX,decpointY]))
        #plt.plot(xDiff[i],decIntersectionPoints[i][1],'ro')
        plt.plot(decIntersectionPoints[i][0],decIntersectionPoints[i][1],'ro')
    return incIntersectionPoints, decIntersectionPoints


###############################################################################
###############################################################################
##load data from the file

x, y=np.loadtxt(filename,dtype=float,delimiter=',',skiprows=FindHeaderLength(),usecols=(0,1), unpack=True)

##############################################################################
##levelling of the surface tilt

coefficients = np.polyfit(x, y, 1)
polynomial = np.poly1d(coefficients)
ys = polynomial(x)

print 'Fitted line equation f(x) =', polynomial

yLevelled=y-ys          #levelled line scan


plt.figure('Full data after levelling and averaging')
plt.plot(x,yLevelled, 'ro', markersize=2,label='Raw data')
plt.title('Full data after levelling and averaging')
plt.xlabel('Lateral [um]')
plt.ylabel('Raw Micrometer [um]')
plt.grid(True)

###############################################################################
##calculate moving average of the levelled data

movingAverageSize = 177
yLevelMovingAverage=scipy.medfilt(yLevelled,movingAverageSize)

plt.plot(x,yLevelMovingAverage,label='Moving average')
plt.xlabel('Lateral [um]')
plt.ylabel('Raw Micrometer [um]')
plt.legend()
plt.grid(True)

##############################################################################
##FFT amplitude and phase plot of the raw data after moving average

dataLenghtFFT = len(yLevelMovingAverage)/2        #divide by 2 to satify rfft
                    # scale by the number of points so that
                    # the magnitude does not depend on the length 
                    # of the signal or on its sampling frequency 

calculatedFFT = np.fft.rfft(yLevelMovingAverage) 

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


xFFT = np.linspace(0,dataLenghtFFT+1,dataLenghtFFT+1)   
                                #the range is two times smaller +1 for RFFT
                                #sinus signal without noise used for fit

plt.figure("FFT amplitude and phase harmonics")
plt.subplot(2,1,1)
plt.vlines(xFFT,0,amplitudeScaledFFT)
plt.title("FFT amplitude harmonics")
plt.xlabel("Harmonics")
plt.ylabel("Amplitude [V]")
plt.xlim(0,dataLenghtFFT/2+1) #adjuts the x axis to maximum of numberOfPoints
plt.grid(True)

plt.subplot(2,1,2)
plt.vlines(xFFT,0,phaseDegreesFFT)
plt.title("FFT phase harmonics")
plt.xlabel("Harmonics")
plt.ylabel("Phase [deg]")
plt.tight_layout()      #removes the overlapping of the labels in subplots
plt.xlim(0,dataLenghtFFT+1)
plt.grid(True)

averageStructureHeight = amplitudeScaledFFT.max()
maxHarmonic = np.where(amplitudeScaledFFT==amplitudeScaledFFT.max())[0][0]

print 'Average structures height:', averageStructureHeight*2 #averageStructureHeight is amplitude
print 'Number of structures:', maxHarmonic-1

###############################################################################
##calculate first order difference along the averaged levelled data

yDiff = np.diff(yLevelMovingAverage)
dataLength = len(yLevelMovingAverage) 
xDiff = np.delete(x,dataLength-1)   #diff consumes one last element from the array

plt.figure('First order difference along the averaged levelled data')
plt.plot(xDiff,yDiff,'ko', markersize=2, label='Raw data')
#plt.plot(xDiff,yDiff)
plt.title('First order difference along the averaged levelled data')
plt.xlabel('Lateral [um]')
plt.ylabel('Raw Micrometer [um]')
plt.grid(True)


##############################################################################
##FFT filtering of the averaged difference data

FirstHarmonics=1800 #only first 'FirstHarmonics' will be left in the FFT data

calculatedFFTFiltered = np.fft.rfft(yDiff) 
calculatedFFTFiltered[FirstHarmonics:]=0    #any harmonics greater than 'FirstHarmonics' #are set to 0
yCalculatedIFFTFiltered = np.fft.irfft(calculatedFFTFiltered) #caclulate IFFT from the filtered FFT


plt.plot(xDiff,yCalculatedIFFTFiltered, label='FFT filtered data')
plt.title('First order difference along the averaged levelled data')
plt.xlabel('Lateral [um]')
plt.ylabel('Raw Micrometer [um]')
plt.legend()
plt.grid(True)




#plt.figure('FFT filtering')
#plt.plot(xDiff,calculatedIFFTFiltered) 

#y=d
y=yCalculatedIFFTFiltered

#plt.figure('Thresholded')
#plt.plot(y)
#plt.plot(thresholdLine)
#plt.title('Levelled data plot')
#plt.xlabel('Lateral [um]')
#plt.ylabel('Micrometer [um]')
#plt.grid(True)


start = 0
stop = 2500

#start = 2500
#stop = 4500

#start = 4500
#stop = 6500

#start = 6500
#stop = 8500
#
#start = 8500
#stop = 10500



#
##plt.figure('Data after median')
#plt.plot(x, scipy.medfilt(yLevelled,177))
#plt.title('Data after median')
#plt.xlabel('Lateral [um]')
#plt.ylabel('Raw Micrometer [um]')
#plt.grid(True)


###FFT and IFFT of the raw signall
#FirstHarmonicsRaw=500
#
#calculatedFFTRaw = np.fft.rfft(yLevelled) 
#calculatedFFTFilteredRaw = calculatedFFTRaw
#calculatedFFTFilteredRaw[FirstHarmonicsRaw:]=0
#calculatedIFFTFilteredRaw = np.fft.irfft(calculatedFFTFilteredRaw)
#
#plt.figure('Raw Data after FFT filtering')
#plt.plot(calculatedIFFTFilteredRaw)
#plt.title('Raw Data after FFT filtering')
#plt.xlabel('Lateral [um]')
#plt.ylabel('Raw Micrometer [um]')
#plt.grid(True)

##############################################################################
##THRESHOLD
##############################################################################
#threshold=-0.05


plt.figure('rest Data after levelling')





#find the value of the xDiff for index comming from incIntersectionPoints[0]
#and decIntersectionPoints[0]


#thresholdLineArray = np.array([0,1,threshold])

increasingPoints = []
incLineEquationCoefficients = []
incIntersectionPoints = []
decLineEquationCoefficients = []
decreasingPoints = []
decIntersectionPoints = []
top = []
bottom = []


    

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

##############################################################################
##second diff
#diff2x=np.diff(calculatedIFFTFiltered)
##plt.plot(xDiff,d,'ko', markersize=2)
#plt.figure("Second diff")
#plt.plot(diff2x)
#plt.title('Second derivative of y')
#plt.xlabel('Lateral [um]')
#plt.ylabel('Raw Micrometer [um]')
#plt.grid(True)

##############################################################################
##Peak detect
#print "detecting peaks"
#peakind = signal.find_peaks_cwt(calculatedIFFTFiltered, np.arange(100,300))
#peakindxDiff = xDiff[peakind]
#peakindy = calculatedIFFTFiltered[peakind]

#Za kumuny chociaz niczego nie bylo to wszystko bylo a teraz wszystko jest ale nic nie ma




#if __name__=="__main__":
#    from matplotlib.pyplot import plot, scatter, show
#    series = [0,0,0,2,0,0,0,-2,0,0,0,2,0,0,0,-2,0]
#    maxtab, mintab = peakdet(series,.3)
#    plot(series)
#    scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
#    scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
#    show()

##############################################################################
##IFFT plot

maxtab, mintab = peakdet(yCalculatedIFFTFiltered,0.05, xDiff)


signalIFFT = np.column_stack((xDiff,yCalculatedIFFTFiltered))
xIFFT = signalIFFT[:,0]
yIFFT = signalIFFT[:,1]
#plt.plot(xIFFT,yIFFT)

##############################################################################

#plt.plot(signalFFT[:,0],scipy.medfilt(signalFFT[:,1],15))

#thresholdStd = np.linspace(calculatedIFFTFiltered.std(),calculatedIFFTFiltered.std(),xDiff.max())
#plt.plot(thresholdStd)
#plt.plot(2*thresholdStd)
#plt.plot(3*thresholdStd)
#plt.plot(-thresholdStd)
#plt.plot(-2*thresholdStd)
#plt.plot(-3*thresholdStd)
#plt.plot(xDiff[920:1320],calculatedIFFTFiltered[920:1320])

#diff3x=np.diff(calculatedIFFTFiltered[920:1320])
##plt.figure(1)
#plt.plot(calculatedIFFTFiltered[920:1320])
##plt.figure(2)
#xDiffAbs = np.absolute(diff3x)
#plt.plot(30*xDiffAbs)
#plt.grid(True)

##############################################################################
##find threshold line

#threshold = 0.035
#
#aincPositve, adecPositve = FindThresholdLine(xDiff,y,threshold)
#increasingPoints = []
#incLineEquationCoefficients = []
#incIntersectionPoints = []
#decLineEquationCoefficients = []
#decreasingPoints = []
#decIntersectionPoints = []
#aincNegative, adecNegative = FindThresholdLine(xDiff,y,-threshold)

##############################################################################




#aincPositve, adecPositve = FindThresholdLine(xIFFT[:2500],yIFFT[:2500],0.019)

thresholdStep = 0.001


for threshold in reversed(np.arange(0, 0.15, thresholdStep)): 
    aincPositve, adecPositve = FindThresholdLine(xIFFT[start:stop],yIFFT[start:stop],threshold)
    if aincPositve.__len__() >= 2 or adecPositve.__len__() >= 2:
        increasingPoints = []
        incLineEquationCoefficients = []
        incIntersectionPoints = []
        decLineEquationCoefficients = []
        decreasingPoints = []
        decIntersectionPoints = []
        aincPositveLast, adecPositveLast = FindThresholdLine(xIFFT[start:stop],yIFFT[start:stop],threshold+thresholdStep)
        break
    increasingPoints = []
    incLineEquationCoefficients = []
    incIntersectionPoints = []
    decLineEquationCoefficients = []
    decreasingPoints = []
    decIntersectionPoints = []

increasingPoints = []
incLineEquationCoefficients = []
incIntersectionPoints = []
decLineEquationCoefficients = []
decreasingPoints = []
decIntersectionPoints = []

for threshold in reversed(np.arange(0, 0.15, 0.001)): 
    aincNegative, adecNegative = FindThresholdLine(xIFFT[start:stop],yIFFT[start:stop],-threshold)
    if aincNegative.__len__() >= 2 or adecNegative.__len__() >= 2:
        increasingPoints = []
        incLineEquationCoefficients = []
        incIntersectionPoints = []
        decLineEquationCoefficients = []
        decreasingPoints = []
        decIntersectionPoints = []
        aincNegativeLast, adecNegativeLast = FindThresholdLine(xIFFT[start:stop],yIFFT[start:stop],-threshold-thresholdStep)
        break 
    increasingPoints = []
    incLineEquationCoefficients = []
    incIntersectionPoints = []
    decLineEquationCoefficients = []
    decreasingPoints = []
    decIntersectionPoints = []

#for i in xrange(0, adecPositve.__len__()):

abottom = aincNegativeLast[0][0] - aincPositveLast[0][0]
#
#for i in xrange(0, adecNegative.__len__()):
atop = adecNegativeLast[0][0] - adecPositveLast[0][0]

#plt.plot(xIFFT,yIFFT)


plt.plot(xIFFT[start:stop],yIFFT[start:stop])
#plt.plot(xIFFT[3000:6000],yIFFT[3000:6000])
plt.grid(True)

#yPointTop1 = yLevelled[adecPositveLast[0][0]]
#yPointTop2 = yLevelled[adecNegativeLast[0][0]]
#
#yPointBottom1 = yLevelled[aincPositveLast[0][0]]
#yPointBottom2 = yLevelled[aincNegativeLast[0][0]]

iyTop1 = np.where(x>adecPositveLast[0][0])
iyTop2 = np.where(x>adecNegativeLast[0][0])

iyBottom1 = np.where(x>aincPositveLast[0][0])
iyBottom2 = np.where(x>aincNegativeLast[0][0])

xPointTop1 = iyTop1[0][0]
xPointTop2 = iyTop2[0][0]

xPointBottom1 = iyBottom1[0][0]
xPointBottom2 = iyBottom2[0][0]

yPointTop1 = yLevelled[xPointTop1]
yPointTop2 = yLevelled[xPointTop2]
yPointBottom1 = yLevelled[xPointBottom1]
yPointBottom2 = yLevelled[xPointBottom2]

xPointTop1 = adecPositveLast[0][0]
xPointTop2 = adecNegativeLast[0][0]
xPointBottom1 = aincPositveLast[0][0]
xPointBottom2 = aincNegativeLast[0][0]

xLineTop = []
yLineTop = []
xLineBottom = []
yLineBottom = []

xLineTop.append(xPointTop1)
xLineTop.append(xPointTop2)
yLineTop.append(yPointTop1)
yLineTop.append(yPointTop2)

xLineBottom.append(xPointBottom1)
xLineBottom.append(xPointBottom2)
yLineBottom.append(yPointBottom1)
yLineBottom.append(yPointBottom2)

plt.figure('Data after levelling')
plt.plot(xPointTop1,yPointTop1,'ro')
plt.plot(xPointTop2,yPointTop2,'ro')
plt.plot(xLineTop,yLineTop)
plt.plot(x[start:stop],yLevelled[start:stop])
plt.plot(xPointBottom1,yPointBottom1,'ro')
plt.plot(xPointBottom2,yPointBottom2,'ro')
plt.plot(xLineBottom,yLineBottom)
plt.title('Data after levelling')
plt.xlabel('Lateral [um]')
plt.ylabel('Raw Micrometer [um]')
plt.grid(True)




plt.figure('Full xDiff Data after levelling')
plt.plot(xIFFT,yIFFT)
plt.plot(maxtab[:,0],maxtab[:,1],'o')
plt.plot(mintab[:,0],mintab[:,1],'o')
#plt.plot(xIFFT[3000:6000],yIFFT[3000:6000])
plt.grid(True)

plt.show()