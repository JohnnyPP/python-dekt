import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scipy
import sys
import math
from numpy import NaN, Inf, arange, isscalar, asarray, array

def FindHeaderLength(file_path, file_name):
    """
    Finds the position of the 'Scan Data' and adds additional 4 lines
    to give as a result the length of the header in number of lines.
    This is then used in np.loadtxt function
    """

    lookup = 'Lateral um'

    file_name_and_path = file_path + file_name

    with open(file_name_and_path) as myFile:
        for FoundPosition, line in enumerate(myFile, 1):
            if lookup in line:
                print 'Scan Data found at line:', FoundPosition
                break

    return FoundPosition + 4


def peakdet(v, delta, x=None):
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
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)


def coefA(x0, y0, x1, y1):
    """
    Finds the coefficient A of the line going thorugh the two points P0 (x0,y0),
    P1 (x1,y1). Run dektakTest.py for a simple example
    :param x0: 
    :param y0:
    :param x1:
    :param y1:
    :return: coefA = -(y1 - y0) / (x1 - x0)
    
    http://en.wikipedia.org/wiki/Linear_equation
    General (or standard) form

    In the general (or standard[1]) form the linear equation is written as:

    coefAx + coefBy = coefC, \,

    where A and B are not both equal to zero. The equation is usually written 
    so that A >= 0, by convention. The graph of the equation is a straight line,
    and every straight line can be represented by an equation in the above 
    form. If A is nonzero, then the x-intercept, that is, the x-coordinate of 
    the point where the graph crosses the x-axis (where, y is zero), is C/A. 
    If B is nonzero, then the y-intercept, that is the y-coordinate of the 
    point where the graph crosses the y-axis (where x is zero), is C/B, and 
    the slope of the line is -A/B. The general form is sometimes written as:
    ax + by + c = 0

    where a and b are not both equal to zero. The two versions can be 
    converted from one to the other by moving the constant term to the other 
    side of the equal sign.

    """
    return -(y1 - y0) / (x1 - x0)


def coefC(x0, y0, x1, y1):
    """
    Finds the coefficient C of the line going thorugh the two points P0 (x0,y0),
    P1 (x1,y1). Run dektakTest.py for a simple example
    :param x0: 
    :param y0:
    :param x1:
    :param y1:
    :return: coefC = -(x1 * y0 - x0 * y1) / (x1 - x0)
    """
    return (x1 * y0 - x0 * y1) / (x1 - x0)


def FindThresholdLine(x, y, threshold):
    """
    Finds the intersection points for a given x, y data that lie on a threshold
    line. This is a kind of interlopation of the x data as the threshold is
    given for the y axis.
    :param x:
    :param y:
    :param threshold:
    :return: incIntersectionPoints, decIntersectionPoints
    """
    
    increasingPoints = []
    incLineEquationCoefficients = []
    incIntersectionPoints = []
    decLineEquationCoefficients = []
    decreasingPoints = []
    decIntersectionPoints = []

    for i in xrange(0, len(y)):
        if i < (len(y) - 1):
            if y[i] < threshold and y[i + 1] > threshold:  # incerasing
                #print "Increasing line detected"
                x0 = x[i]
                y0 = y[i]
                x1 = x[i + 1]
                y1 = y[i + 1]
                coefAinc = coefA(x0, y0, x1, y1)
                coefCinc = coefC(x0, y0, x1, y1)
                incPointsCrossingThreshold = np.array([x0, y0, x1, y1])
                lineEquationCoefficientsCrossongThr = np.array([coefAinc, 1, coefCinc])
                increasingPoints.append(incPointsCrossingThreshold)
                incLineEquationCoefficients.append(lineEquationCoefficientsCrossongThr)
            else:
                if y[i] > threshold and y[i + 1] < threshold:  # decreasing
                    #print "Decreasing line detected"
                    x0 = x[i]
                    y0 = y[i]
                    x1 = x[i + 1]
                    y1 = y[i + 1]
                    coefAdec = coefA(x0, y0, x1, y1)
                    coefCdec = coefC(x0, y0, x1, y1)
                    decPointsCrossingThreshold = np.array([x0, y0, x1, y1])
                    lineEquationCoefficientsCrossongThrDec = np.array([coefAdec, 1, coefCdec])
                    decreasingPoints.append(decPointsCrossingThreshold)
                    decLineEquationCoefficients.append(lineEquationCoefficientsCrossongThrDec)
                    #else:
                    #print "Neither dereasing nor incereasing line"
                    
        else:
            break

    thresholdLineArray = np.array([0, 1, threshold])    # coefA=0, coefB=1
                                                        # coefC=threshold
    A2 = thresholdLineArray[0]  # threshold line coeficients
    B2 = thresholdLineArray[1]
    C2 = thresholdLineArray[2]

    """
    The intersection points may be calculated when both line equations are
    known (line coefficients). This is done separately for increasing line:
    incA1, incB1, incC1 coeficients 
    and decreasing line coefficients:
    decA1, decB1, decC1 
    Threshold line coefficients:
    A2, B2, C2
    
    The intersection points are calculated by means of determinant and
    linear system of equations:
    http://en.wikipedia.org/wiki/Determinant
    http://en.wikipedia.org/wiki/System_of_linear_equations
    http://www.cliffsnotes.com/math/algebra/algebra-ii/linear-sentences-in-two-
    variables/linear-equations-solutions-using-determinants-with-two-variables
    
    W=|A1B1
       A2B2|=A1*B2-A2*B1
    Wx=|-C1B1
        -C2B2|=(-C1)*B2-(-C2)*B1   #determinant [[5, 0], [6, 1]]
    Wy=|A1-C1
        A2-C2|=A1*(-C2)-A2*(-C1)   #determinant [[4, 5], [1, 6]]

    Ax+By+C=0
    A=(y_1-y_0)/(x_1-x_0)
    B=1 always
    C=(x_1*y_0-x_0*y_1)/(x_1-x_0)
    
    Geometric interpretation

    For a system involving two variables (x and y), each linear equation 
    determines a line on the xy-plane. Because a solution to a linear system 
    must satisfy all of the equations, the solution set is the intersection 
    of these lines, and is hence either a line, a single point, 
    or the empty set. We are satisfied with single point solution.
    """
    
    for i in xrange(0, incLineEquationCoefficients.__len__()):
        incA1 = incLineEquationCoefficients[i][0]
        incB1 = incLineEquationCoefficients[i][1]
        incC1 = incLineEquationCoefficients[i][2]

        detW = float(incA1 * B2 - A2 * incB1)
        detWx = float((incC1) * B2 - (C2) * incB1)
        detWy = float(incA1 * (C2) - A2 * (incC1))
        pointX = float(detWx / detW)
        pointY = float(detWy / detW)
        incIntersectionPoints.append(np.array([pointX, pointY]))
        
    for i in xrange(0, decLineEquationCoefficients.__len__()):
        decA1 = decLineEquationCoefficients[i][0]
        decB1 = decLineEquationCoefficients[i][1]
        decC1 = decLineEquationCoefficients[i][2]

        decdetW = float(decA1 * B2 - A2 * decB1)
        decdetWx = float((decC1) * B2 - (C2) * decB1)
        decdetWy = float(decA1 * (C2) - A2 * (decC1))
        decpointX = float(decdetWx / decdetW)
        decpointY = float(decdetWy / decdetW)
        decIntersectionPoints.append(np.array([decpointX, decpointY]))

    return incIntersectionPoints, decIntersectionPoints


def load_data(file_name, file_path):
    """
    Loads data from the dektak csv file
    :param file_name:
    :param file_path:
    :return: x, y raw data
    """

    #file_path = '/home/kolan/mycode/python/dektak/data/'
    # filename = '/home/kolan/mycode/python/dektak/data/t10_1_3_normal.csv'
    #filename = '/home/kolan/mycode/python/dektak/data/t10_1_6_normal.csv'
    #filename = '/home/kolan/mycode/python/dektak/data/t10_1_7_normal.csv'    #first peak very good   18thPositive peak short
    #filename = '/home/kolan/mycode/python/dektak/data/t10_1_12_normal.csv' #abottom IndexError: list index out of range
    #filename = '/home/kolan/mycode/python/dektak/data/t10_1_15_normal.csv'  #abottom IndexError: list index out of range
    #filename = '/home/kolan/mycode/python/dektak/data/t10_1_19_normal.csv'
    #filename = '/home/kolan/mycode/python/dektak/data/t10_1_21_normal.csv'   #no top & bottom
    #filename = '/home/kolan/mycode/python/dektak/data/t10_1_24_normal.csv'  #no top & bottom
    #filename = '/home/kolan/mycode/python/dektak/data/t10_1_3_parallel.csv'  #no top & bottom
    #filename = '/home/kolan/mycode/python/dektak/data/t10_1_15_parallel.csv'  #abottom IndexError: list index out of range
    #filename = '/home/kolan/mycode/python/dektak/data/t10_1_19_parallel.csv'  #0.035 too low 0.04 ok BADabottom
    #filename = '/home/kolan/mycode/python/dektak/data/t10_1_24_parallel.csv' #first peak very good
    #filename = '/home/kolan/mycode/python/dektak/data/t10_3_1_normal.csv'
    #filename = '/home/kolan/mycode/python/dektak/data/t10_3_3_normal.csv'
    #filename = '/home/kolan/mycode/python/dektak/data/t10_3_6_normal.csv'
    #filename = '/home/kolan/mycode/python/dektak/data/t10_3_7_normal.csv'    #short peak
    #filename = '/home/kolan/mycode/python/dektak/data/t10_3_15_normal.csv'
    #filename = '/home/kolan/mycode/python/dektak/data/t10_3_19_normal.csv'

    file_name_and_path = file_path + file_name

    x, y = np.loadtxt(file_name_and_path, dtype=float, delimiter=',', skiprows=FindHeaderLength(file_path, file_name),
                      usecols=(0, 1), unpack=True)
    return x, y


def surface_tilt_levelling(x, y):
    """
    Performs polynomial fit to the tilted profile and returns the difference 
    between raw data and the fitted curve what levels (straightens) the profile
    :param x: raw data
    :param y: raw data
    :return: yLevelled levelled profile
    """

    coefficients = np.polyfit(x, y, 2)
    polynomial = np.poly1d(coefficients)
    yPoly = polynomial(x)

    print 'Fitted line equation f(x) =', polynomial

    yLevelled = y - yPoly  # levelled line scan

    plt.figure('Full raw data')
    plt.plot(x, y, label='Full raw data')
    plt.plot(x, yPoly, label='Polynomial')
    plt.title('Full raw data')
    plt.xlabel('Lateral [um]')
    plt.ylabel('Raw Micrometer [um]')
    plt.legend()
    plt.grid(True)
    #plt.show()


    plt.figure('Full data after levelling and averaging')
    plt.plot(x, yLevelled, 'ro', markersize=2, label='Raw data')
    plt.title('Full data after levelling and averaging')
    plt.xlabel('Lateral [um]')
    plt.ylabel('Raw Micrometer [um]')
    plt.grid(True)

    return yLevelled


def moving_average(x_data, data_to_average, average_size):
    """
    Calculates moving average of the levelled data
    :param data_to_average:
    :param average_size:
    :return: yLevelMovingAverage averaged levelled data
    """

    yLevelMovingAverage = scipy.medfilt(data_to_average, average_size)
    plt.plot(x_data, yLevelMovingAverage, label='Moving average')
    plt.xlabel('Lateral [um]')
    plt.ylabel('Raw Micrometer [um]')
    plt.legend()
    plt.grid(True)
    return yLevelMovingAverage


def amplitude_and_phase_FFT(data_to_analyze):
    """
    Calculates FFT of the data_to_analyze np.array. Plots the amplitude vs. harmonics and phase vs. harmonics.
    :param data_to_analyze: input np.array with the data
    :return: averageStructuresHeight, maxHarmonic
    """

    ##############################################################################
    ##FFT amplitude and phase plot of the raw data after moving average

    dataLenghtFFT = len(data_to_analyze) / 2  # divide by 2 to satisfy rfft
    # scale by the number of points so that
    # the magnitude does not depend on the length
    # of the signal or on its sampling frequency

    calculatedFFT = np.fft.rfft(data_to_analyze)
    amplitudeFFT = np.abs(calculatedFFT)  # calculates FFT amplitude from
    # complex calculatedFFT output
    phaseFFT = np.angle(calculatedFFT)  # calculates FFT phase from
    # complex calculatedFFT output
    phaseDegreesFFT = np.rad2deg(phaseFFT)  # convert to degrees
    amplitudeScaledFFT = amplitudeFFT / float(dataLenghtFFT)
    # scale by the number of points so that
    # the magnitude does not depend on the length
    # of the signal
    amplitudeScaledRMSFFT = amplitudeFFT / float(dataLenghtFFT) / math.sqrt(2)
    # Scaling to Root mean square amplitude (dataLengthFFT/sqrt{2}),


    xFFT = np.linspace(0, dataLenghtFFT + 1, dataLenghtFFT + 1)
    #the range is two times smaller +1 for RFFT
    #sinus signal without noise used for fit

    plt.figure("FFT amplitude and phase harmonics")
    plt.subplot(2, 1, 1)
    plt.vlines(xFFT, 0, amplitudeScaledFFT)
    plt.title("FFT amplitude harmonics")
    plt.xlabel("Harmonics")
    plt.ylabel("Amplitude [V]")
    plt.xlim(0, dataLenghtFFT / 2 + 1)  # adjusts the x axis to maximum of numberOfPoints
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.vlines(xFFT, 0, phaseDegreesFFT)
    plt.title("FFT phase harmonics")
    plt.xlabel("Harmonics")
    plt.ylabel("Phase [deg]")
    plt.tight_layout()  #removes the overlapping of the labels in subplots
    plt.xlim(0, dataLenghtFFT + 1)
    plt.grid(True)

    averageStructuresHeight = amplitudeScaledFFT.max()
    maxHarmonic = np.where(amplitudeScaledFFT == amplitudeScaledFFT.max())[0][0] - 1

    print 'Average structures height calculated using FFT:', averageStructuresHeight * 2, 'um'  #averageStructureHeight is amplitude
    print 'Number of structures calculated using FFT:', maxHarmonic

    return averageStructuresHeight, maxHarmonic


def first_order_difference(x_data, data_to_analyze):
    """
    Calculates the difference between subsequent data points in the data_to_analyze np.array
    :param data_to_analyze: input np.array with the data
    :return: xDiff, yDiff arrays with calculated data
    """

    yDiff = np.diff(data_to_analyze)
    dataLength = len(data_to_analyze)
    xDiff = np.delete(x_data, dataLength - 1)  # diff consumes one last element from the array
    plt.figure('First order difference along the averaged levelled data')
    plt.plot(xDiff, yDiff, 'ko', markersize=2, label='Raw data')
    plt.title('First order difference along the averaged levelled data')
    plt.xlabel('Lateral [um]')
    plt.ylabel('Raw Micrometer [um]')
    plt.grid(True)

    return xDiff, yDiff


def FFT_filtering(x_data, y_data_to_filter, first_harmonics_to_keep):
    """
    Filters the data by means of FFT
    :param x_data: used only for plot
    :param y_data_to_filter: data to be filtered
    :param first_harmonics_to_keep: the number of harmonics that are not set to 0 counting from the 0th harmonics
    :return:
    """

    calculatedFFTFiltered = np.fft.rfft(y_data_to_filter)
    calculatedFFTFiltered[first_harmonics_to_keep:] = 0  # any harmonics greater than 'FirstHarmonics' #are set to 0
    yCalculatedIFFTFiltered = np.fft.irfft(calculatedFFTFiltered)  # calculate IFFT from the filtered FFT

    plt.plot(x_data, yCalculatedIFFTFiltered, label='FFT filtered data')
    plt.title('First order difference along the averaged levelled data')
    plt.xlabel('Lateral [um]')
    plt.ylabel('Raw Micrometer [um]')
    plt.legend()
    plt.grid(True)

    return yCalculatedIFFTFiltered


def find_minima_and_maxima(xdata, ydata, peak_threshold, max_harmonic):
    """
    Function finds the maxima and minima in FFT filtered data
    :param xdata:
    :param ydata:
    :param threshold: # reliable results between 0.05 and 0.09
    :return: maxtab, mintab
    """

    maxtab, mintab = peakdet(ydata, peak_threshold, xdata)

    plt.plot(maxtab[:, 0], maxtab[:, 1], 'o')
    plt.plot(mintab[:, 0], mintab[:, 1], 'o')

    peakDetectMaxima = len(maxtab)
    peakDetectMinima = len(mintab)

    print 'Number of found maxima in first order difference data', peakDetectMaxima
    print 'Number of found minima in first order difference data', peakDetectMinima

    if peakDetectMaxima != peakDetectMinima:
        print 'Number of minima and maxima is not equal. Try to adjust peak_threshold parameter'
    if peakDetectMaxima != max_harmonic and peakDetectMinima != max_harmonic:
        print 'Number of structures found by FFT not equals the number of minima \
                and maxima found by peakdetect(). Try to adjust peakThreshold parameter'

    maxtabDiff = np.diff(maxtab, axis=0)[:, 0]  #uses only 1st column
    mintabDiff = np.diff(mintab, axis=0)[:, 0]  #uses only 1st column

    print 'Mean distance between structures from maxima', maxtabDiff.mean()
    print 'Mean distance between structures from minima', mintabDiff.mean()

    return maxtab, mintab


def scanning_threshold_positive(x_data, y_data, threshold_step, threshold_length, maxtab, sliceNumber):
    """
    Function scans the positive peak with the given threshold step
    :param x_data:
    :param y_data:
    :param threshold_step:
    :return:
    """

    for threshold in reversed(np.arange(0, 0.15, threshold_step)):
        aincPositve, adecPositve = FindThresholdLine(x_data, y_data, threshold)
        if aincPositve.__len__() >= 2 or adecPositve.__len__() >= 2:
            if maxtab[sliceNumber][1] - threshold < threshold_length:
                print 'Entered threshold length case positive peaks'
                print maxtab[sliceNumber][1] - threshold
                print sliceNumber
            else:
                aincPositveLast, adecPositveLast = FindThresholdLine(x_data, y_data,
                                                                     threshold + threshold_step)
                break
    return aincPositveLast, adecPositveLast


def scanning_threshold_negative(x_data, y_data, threshold_step, threshold_length, mintab, sliceNumber):
    """
    Function scans the negative peak with the given threshold step
    :param x_data:
    :param y_data:
    :param threshold_step:
    :return:
    """

    for threshold in reversed(np.arange(0, 0.15, threshold_step)):
        aincNegative, adecNegative = FindThresholdLine(x_data, y_data, -threshold)
        if aincNegative.__len__() >= 2 or adecNegative.__len__() >= 2:
            if mintab[sliceNumber][1] + threshold > -threshold_length:
                print 'Entered threshold length case negative peaks'
                print mintab[sliceNumber][1] + threshold
                print sliceNumber
                print -threshold
            else:
                aincNegativeLast, adecNegativeLast = FindThresholdLine(x_data, y_data,
                                                                       -threshold - threshold_step)
                break
    return aincNegativeLast, adecNegativeLast


def translate_points(sliceNumber, x_data, start, stop, adecPositveLast, adecNegativeLast, aincPositveLast,
                     aincNegativeLast, yLevelled, axs):
    """
Translate the points found in Diff data to the xy data for plotting the top and bottom lines
:param sliceNumber:
:param x_data:
:param start:
:param stop:
:param adecPositveLast:
:param adecNegativeLast:
:param aincPositveLast:
:param aincNegativeLast:
:return:
"""

    xShiftedToZero = x_data[start:stop] - x_data[start:stop][0]

    iyTop1 = np.where(xShiftedToZero > adecPositveLast[0][0])
    iyTop2 = np.where(xShiftedToZero > adecNegativeLast[0][0])

    iyBottom1 = np.where(xShiftedToZero > aincPositveLast[0][0])
    iyBottom2 = np.where(xShiftedToZero > aincNegativeLast[0][0])

    xPointTop1 = iyTop1[0][0]
    xPointTop2 = iyTop2[0][0]

    xPointBottom1 = iyBottom1[0][0]
    xPointBottom2 = iyBottom2[0][0]

    yPointTop1 = yLevelled[start:stop][xPointTop1]
    yPointTop2 = yLevelled[start:stop][xPointTop2]
    yPointBottom1 = yLevelled[start:stop][xPointBottom1]
    yPointBottom2 = yLevelled[start:stop][xPointBottom2]

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

    axs[sliceNumber].plot(xPointTop1, yPointTop1, 'bo')
    axs[sliceNumber].plot(xPointTop2, yPointTop2, 'bo')
    axs[sliceNumber].plot(xLineTop, yLineTop)
    xShiftedToZero = x_data[start:stop] - x_data[start:stop][0]
    axs[sliceNumber].plot(xShiftedToZero, yLevelled[start:stop])
    axs[sliceNumber].plot(xPointBottom1, yPointBottom1, 'ro')
    axs[sliceNumber].plot(xPointBottom2, yPointBottom2, 'ro')
    axs[sliceNumber].plot(xLineBottom, yLineBottom)
    axs[sliceNumber].grid(True)
    axs[sliceNumber].set_title(str(sliceNumber))

    return None

def calculate_start_slice(x_axis, start_slice): 
    # calculates the position of the start slice in um.This is used to
    # translate the position of the points from slices to the whole plot
    start_slice_list.append(x_axis[start_slice])
    return start_slice_list
    

#######################################################################################################################
#######################################################################################################################
# parameters to set

file_name = 't10_1_1_normal.csv'
file_path = '/home/kolan/mycode/python/dektak/data/'
moving_average_size = 277
first_harmonics_to_keep = 1500
find_peak_threshold = 0.065            # reliable results between 0.05 and 0.09

#
x, y = load_data(file_name, file_path)

yLevelled = surface_tilt_levelling(x, y)

yLevelMovingAverage = moving_average(x, yLevelled, moving_average_size)

averageStructuresHeight, maxHarmonic = amplitude_and_phase_FFT(yLevelMovingAverage)

xDiff, yDiff = first_order_difference(x, yLevelMovingAverage)

yCalculatedIFFTFiltered = FFT_filtering(xDiff, yDiff, first_harmonics_to_keep)

maxtab, mintab = find_minima_and_maxima(xDiff, yCalculatedIFFTFiltered, find_peak_threshold, maxHarmonic)

##############################################################################
## Slicing

increaseSliceLength = 200           # this is in index
thresholdLengthStep = 0.002
thresholdStep = 0.001


widthTop = []
widthBottom = []
stdWidthTop = []
stdWidthBottom = []
spaceTop = []
spaceBottom = []
start_slice_list = []
adecNegativeLastList = []
aincNegativeLastList = []
adecPositveLastList = []
aincPositveLastList = []
yAdecNegativeLastList = []
yAincNegativeLastList = []
yAdecPositveLastList = []
yAincPositveLastList = []
xlineListTop = []
xlineListBottom = []
ylineListTop = []
ylineListBottom = []

fig, axs = plt.subplots(5, 4, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.5, wspace=.2)
axs = axs.ravel()
plt.suptitle('Sliced structures: x Lateral [um], y Raw Micrometer [um]')

signalIFFT = np.column_stack((xDiff, yCalculatedIFFTFiltered))
xIFFT = signalIFFT[:, 0]    #keeps the xDiff, yCalculatedIFFTFiltered results in one np.array
yIFFT = signalIFFT[:, 1]


for thresholdLength in (np.arange(0.0, 0.04, thresholdLengthStep)):
    for sliceNumber in range(maxHarmonic):

       indexOfMaxOccurrence = np.where(x > maxtab[sliceNumber][0])
       indexOfMinOccurrence = np.where(x > mintab[sliceNumber][0])

       start = indexOfMaxOccurrence[0][0] - increaseSliceLength
       stop = indexOfMinOccurrence[0][0] + increaseSliceLength
       xShiftedToZero = xIFFT[start:stop] - xIFFT[start:stop][0]

       aincPositveLast, adecPositveLast = scanning_threshold_positive(xShiftedToZero, yIFFT[start:stop], thresholdStep,
                                                                           thresholdLength, maxtab, sliceNumber)
       aincNegativeLast, adecNegativeLast = scanning_threshold_negative(xShiftedToZero, yIFFT[start:stop],
                                                                             thresholdStep, thresholdLength, mintab,
                                                                             sliceNumber)

       abottom = aincNegativeLast[0][0] - aincPositveLast[0][0]
       atop = adecNegativeLast[0][0] - adecPositveLast[0][0]

       widthBottom.append(abottom)
       widthTop.append(atop)

       translate_points(sliceNumber, x, start, stop, adecPositveLast, adecNegativeLast, aincPositveLast,
                             aincNegativeLast, yLevelled, axs)
                        
        
    npWidthTop = np.array(widthTop)
    npWidthBottom = np.array(widthBottom)

    print 'Mean top width: ', np.mean(npWidthTop), '+/-', np.std(npWidthTop), 'um'
    print 'Mean bottom width: ', np.mean(npWidthBottom), '+/-', np.std(npWidthBottom), 'um'

    stdWidthTop.append(np.std(npWidthTop))
    stdWidthBottom.append(np.std(npWidthBottom))

    if np.std(npWidthTop) > 4 or np.std(npWidthBottom) > 4:
        break

print stdWidthTop
print stdWidthBottom


npStdWidthTop = np.array(stdWidthTop)
npStdWidthBottom = np.array(stdWidthBottom)

plt.figure('Std+std')
plt.plot(npStdWidthTop + npStdWidthBottom)

sumWidth = npStdWidthTop + npStdWidthBottom
minThresholdLength = np.where(sumWidth == sumWidth.min())[0][0] * thresholdLengthStep

print 'Min threshold length', minThresholdLength

widthTop = []
widthBottom = []
stdWidthTop = []
stdWidthBottom = []

fig, axs = plt.subplots(5, 4, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.5, wspace=.2)
axs = axs.ravel()
plt.suptitle('Sliced structures last: x Lateral [um], y Raw Micrometer [um]')


##############################################################################
# The last run with the minThresholdLength found by the scanning above

for sliceNumber in range(maxHarmonic):

    indexOfMaxOccurrence = np.where(x > maxtab[sliceNumber][0])
    indexOfMinOccurrence = np.where(x > mintab[sliceNumber][0])

    start = indexOfMaxOccurrence[0][0] - increaseSliceLength
    stop = indexOfMinOccurrence[0][0] + increaseSliceLength
    
    start_slice_list = calculate_start_slice(x, start)
    
    thresholdStep = 0.001
    signalIFFT = np.column_stack((xDiff, yCalculatedIFFTFiltered))
    xIFFT = signalIFFT[:, 0]
    yIFFT = signalIFFT[:, 1]

    xShiftedToZero = xIFFT[start:stop] - xIFFT[start:stop][0]

    aincPositveLast, adecPositveLast = scanning_threshold_positive(xShiftedToZero, yIFFT[start:stop], thresholdStep,
                                                                           minThresholdLength, maxtab, sliceNumber)
    aincNegativeLast, adecNegativeLast = scanning_threshold_negative(xShiftedToZero, yIFFT[start:stop],
                                                                             thresholdStep, minThresholdLength, mintab,
                                                                             sliceNumber)

    abottom = aincNegativeLast[0][0] - aincPositveLast[0][0]
    atop = adecNegativeLast[0][0] - adecPositveLast[0][0]

        #need to save the current and next peak
        #this is done in the list
        #after each odd sliceNumber the list need to be reset

    adecNegativeLastList.append(adecNegativeLast[0][0] + start_slice_list[sliceNumber])
    aincNegativeLastList.append(aincNegativeLast[0][0] + start_slice_list[sliceNumber])
    adecPositveLastList.append(adecPositveLast[0][0] + start_slice_list[sliceNumber])
    aincPositveLastList.append(aincPositveLast[0][0] + start_slice_list[sliceNumber])
    
    yAdecNegativeLastList.append(yLevelled[np.where(x > adecNegativeLastList[sliceNumber])[0][0]])
    yAincNegativeLastList.append(yLevelled[np.where(x > aincNegativeLastList[sliceNumber])[0][0]])
    yAdecPositveLastList.append(yLevelled[np.where(x > adecPositveLastList[sliceNumber])[0][0]])
    yAincPositveLastList.append(yLevelled[np.where(x > aincPositveLastList[sliceNumber])[0][0]])

    widthBottom.append(abottom)
    widthTop.append(atop)

    translate_points(sliceNumber, x, start, stop, adecPositveLast, adecNegativeLast, aincPositveLast,
                             aincNegativeLast, yLevelled, axs)
                             
for i in range(maxHarmonic-1):
    # appends the spaceTop and spaceBottom
    spaceTop.append(aincPositveLastList[i+1]-aincNegativeLastList[i])
    spaceBottom.append(adecPositveLastList[i+1]-adecNegativeLastList[i])
    
for i in range(maxHarmonic):
    # appends the lines for the Full raw data with points plot
    xlineListTop.append(adecPositveLastList[i])
    xlineListTop.append(adecNegativeLastList[i])
    xlineListBottom.append(aincPositveLastList[i])
    xlineListBottom.append(aincNegativeLastList[i])
    
    ylineListTop.append(yAdecPositveLastList[i])
    ylineListTop.append(yAdecNegativeLastList[i])
    ylineListBottom.append(yAincPositveLastList[i])
    ylineListBottom.append(yAincNegativeLastList[i])
    

npWidthTop = np.array(widthTop)
npWidthBottom = np.array(widthBottom)
npSpaceTop = np.array(spaceTop)
npSpaceBottom = np.array(spaceBottom)

print 'Mean top width with threshold length: ', np.mean(npWidthTop), '+/-', np.std(npWidthTop), 'um'
print 'Mean bottom width with threshold length: ', np.mean(npWidthBottom), '+/-', np.std(npWidthBottom), 'um'

print 'Mean space top width with threshold length: ', np.mean(npSpaceTop), '+/-', np.std(npSpaceTop), 'um'
print 'Mean space bottom width with threshold length: ', np.mean(npSpaceBottom), '+/-', np.std(npSpaceBottom), 'um'

stdWidthTop.append(np.std(npWidthTop))
stdWidthBottom.append(np.std(npWidthBottom))

plt.figure('Full raw data with points')
plt.plot(x, yLevelled, 'g')
plt.plot(adecNegativeLastList, yAdecNegativeLastList, 'bo')
plt.plot(aincNegativeLastList, yAincNegativeLastList, 'ro')
plt.plot(adecPositveLastList, yAdecPositveLastList, 'bo')
plt.plot(aincPositveLastList, yAincPositveLastList, 'ro')

plt.plot(xlineListTop, ylineListTop)
plt.plot(xlineListBottom, ylineListBottom)
plt.grid(True)

plt.show()
