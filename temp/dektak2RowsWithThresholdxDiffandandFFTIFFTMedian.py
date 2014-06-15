import numpy as np 
import matplotlib.pyplot as plt
#from scipy.signal import medfilt
import scipy.signal as scipy
#from scipy import *
#import scipy.signal as signal
#import scipy.signal.medfilt as scipymedfilt  
#import math

#filename = '/home/kolan/mycode/python/dektak/t10_1_1_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/t10_1_3_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/t10_1_6_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/t10_1_7_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/t10_1_12_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/t10_1_15_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/t10_1_19_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/t10_1_21_normal.csv'   #no top & bottom
filename = '/home/kolan/mycode/python/dektak/t10_1_24_normal.csv'  #no top & bottom
#filename = '/home/kolan/mycode/python/dektak/t10_1_3_parallel.csv'  #no top & bottom
#filename = '/home/kolan/mycode/python/dektak/t10_1_15_parallel.csv'
#filename = '/home/kolan/mycode/python/dektak/t10_1_19_parallel.csv'  #0.035 too low 0.04 ok
#filename = '/home/kolan/mycode/python/dektak/t10_1_24_parallel.csv'
#filename = '/home/kolan/mycode/python/dektak/t10_3_1_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/t10_3_3_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/t10_3_6_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/t10_3_7_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/t10_3_15_normal.csv'
#filename = '/home/kolan/mycode/python/dektak/t10_3_19_normal.csv'
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

#def medfilt(x, k):
#    """Apply a length-k median filter to a 1D array x.
#    Boundaries are extended by repeating endpoints.
#    """
#    assert k % 2 == 1, "Median filter length must be odd."
#    assert x.ndim == 1, "Input must be one-dimensional."
#    k2 = (k - 1) // 2
#    y = np.zeros ((len (x), k), dtype=x.dtype)
#    y[:,k2] = x
#    for i in range (k2):
#        j = k2 - i
#        y[j:,i] = x[:-j]
#        y[:j,i] = x[0]
#        y[:-j,-(i+1)] = x[j:]
#        y[-j:,-(i+1)] = x[-1]
#    return np.median (y, axis=1)





coefficients = np.polyfit(x, y, 1)
polynomial = np.poly1d(coefficients)
ys = polynomial(x)
print coefficients
print polynomial

yLevelled=y-ys

plt.figure('Data after levelling')
plt.plot(x,yLevelled)
plt.title('Data after levelling')
plt.xlabel('Lateral [um]')
plt.ylabel('Raw Micrometer [um]')
plt.grid(True)

#plt.figure('Data after median')
plt.plot(x, scipy.medfilt(yLevelled,177))
plt.title('Data after median')
plt.xlabel('Lateral [um]')
plt.ylabel('Raw Micrometer [um]')
plt.grid(True)


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



#y=yLevelled
y=scipy.medfilt(yLevelled,177)
#thresholdLine = np.linspace(threshold,threshold,len(y))


dataLength = len(yLevelled) 
xDiff = np.delete(x,dataLength-1)   #diff consumes one last element from the array

plt.figure('Derivative of y')
d = np.diff(y)
plt.plot(xDiff,d,'ko', markersize=2)
plt.title('Derivative of y')
plt.xlabel('Lateral [um]')
plt.ylabel('Raw Micrometer [um]')
plt.grid(True)



##############################################################################
##FFT filtering
FirstHarmonics=1800

calculatedFFT = np.fft.rfft(d) 
calculatedFFTFiltered = calculatedFFT
calculatedFFTFiltered[FirstHarmonics:]=0
calculatedIFFTFiltered = np.fft.irfft(calculatedFFTFiltered)
#plt.figure('FFT filtering')
#plt.plot(xDiff,calculatedIFFTFiltered) 

#y=d
y=calculatedIFFTFiltered

#plt.figure('Thresholded')
#plt.plot(y)
#plt.plot(thresholdLine)
#plt.title('Levelled data plot')
#plt.xlabel('Lateral [um]')
#plt.ylabel('Micrometer [um]')
#plt.grid(True)

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

def coefA(x0,y0,x1,y1):
    return -(y1-y0)/(x1-x0)

def coefC(x0,y0,x1,y1):
    return (x1*y0-x0*y1)/(x1-x0)

def FindThresholdLine(x,y,threshold):
    thresholdLineArray = np.array([0,1,threshold])
    for i in xrange(0, len(y)):
        if i < (len(y)-1):
            if y[i]<threshold and y[i+1]>threshold: #incerasing
                print "Increasing line detected"
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
                    print "Decreasing line detected"
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

#Za kumuny chociaz niczego nie bylo to wszystko bylo a teraz wszystko jest a nic nie ma

import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array

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

#maxtab, mintab = peakdet(calculatedIFFTFiltered,0.05, xDiff)
#plt.plot(maxtab[:,0],maxtab[:,1],'o')
#plt.plot(mintab[:,0],mintab[:,1],'o')
#
signalIFFT = np.column_stack((xDiff,calculatedIFFTFiltered))
xIFFT = signalIFFT[:,0]
yIFFT = signalIFFT[:,1]
plt.plot(xIFFT,yIFFT)

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

threshold = 0.035

aincPositve, adecPositve = FindThresholdLine(xDiff,y,threshold)
increasingPoints = []
incLineEquationCoefficients = []
incIntersectionPoints = []
decLineEquationCoefficients = []
decreasingPoints = []
decIntersectionPoints = []
aincNegative, adecNegative = FindThresholdLine(xDiff,y,-threshold)

##############################################################################

#for i in xrange(0, adecPositve.__len__()):
#    bottom.append(aincNegative[i][0] - aincPositve[i][0])
#
#for i in xrange(0, adecNegative.__len__()):
#    top.append(adecNegative[i][0] - adecPositve[i][0])
#plt.plot(xIFFT,yIFFT)
#FindThresholdLine(xIFFT[:2500],yIFFT[:2500],0.05)
#
#plt.plot(xIFFT[:2500],yIFFT[:2500])
#plt.plot(xIFFT[3000:6000],yIFFT[3000:6000])
plt.show()