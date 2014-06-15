import numpy as np 
import matplotlib.pyplot as plt
import math

filename = '/home/kolan/mycode/python/dektak/t10_1_1_normal.csv'
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


##############################################################################
##THRESHOLD
##############################################################################
threshold=0.05



y=yLevelled
thresholdLine = np.linspace(threshold,threshold,len(y))


dataLength = len(yLevelled) 
xDiff = np.delete(x,dataLength-1)   #diff consumes one last element from the array

plt.figure(6)
d = np.diff(y)
plt.plot(xDiff,d)
#plt.plot(d)
plt.title('Derivative of y')
plt.xlabel('Lateral [um]')
plt.ylabel('Raw Micrometer [um]')
plt.grid(True)

y=d

#plt.figure('Thresholded')
#plt.plot(y)
#plt.plot(thresholdLine)
#plt.title('Levelled data plot')
#plt.xlabel('Lateral [um]')
#plt.ylabel('Micrometer [um]')
#plt.grid(True)

#find the value of the xDiff for index comming from incIntersectionPoints[0]
#and decIntersectionPoints[0]
#scalingCoefficient = y.__len__()/xDiff.max()

scalingCoefficient = 9.99


thresholdLineArray = np.array([0,1,threshold])

increasingPoints = []
incLineEquationCoefficients = []
incIntersectionPoints = []
decLineEquationCoefficients = []
decreasingPoints = []
decIntersectionPoints = []

def coefA(x0,y0,x1,y1):
    return -(y1-y0)/(x1-x0)

def coefC(x0,y0,x1,y1):
    return (x1*y0-x0*y1)/(x1-x0)
    
for i in xrange(0, len(y)):
    if i < (len(y)-1):
        if y[i]<threshold and y[i+1]>threshold: #incerasing
            print "Increasing line detected"
            x0 = i
            y0 = y[i]
            x1 = i+1
            y1 = y[i+1]
            #print str(x0) + ',' + str(y0)
            #print str(x1) + ',' + str(y1)
            #print (y1-y0)/(x1-x0)
            #print (x1*y0-x0*y1)/(x1-x0)
            coefAinc = coefA(x0,y0,x1,y1)
            coefCinc = coefC(x0,y0,x1,y1)
            #print coefAprint
            #print coefCprint
            incPointsCrossingThreshold = np.array([x0,y0,x1,y1])
            lineEquationCooficientsCrossongThr = np.array([coefAinc, 1, coefCinc])
            increasingPoints.append(incPointsCrossingThreshold)
            incLineEquationCoefficients.append(lineEquationCooficientsCrossongThr)

        else:
            if y[i]>threshold and y[i+1]<threshold: #decreasing
                print "Decreasing line detected"
                x0 = i
                y0 = y[i]
                x1 = i+1
                y1 = y[i+1]
                coefAdec = coefA(x0,y0,x1,y1)
                coefCdec = coefC(x0,y0,x1,y1)
                decPointsCrossingThreshold = np.array([x0,y0,x1,y1])
                lineEquationCooficientsCrossongThrDec = np.array([coefAdec, 1, coefCdec])
                decreasingPoints.append(decPointsCrossingThreshold)
                decLineEquationCoefficients.append(lineEquationCooficientsCrossongThrDec)
            else:
                print "Neither dereasing nor incereasing line"
            
    else:
        break

A2=thresholdLineArray[0]
B2=thresholdLineArray[1]
C2=thresholdLineArray[2]

#def determitantW(A0,B0,A1,B1):
#def determitantW(lineEquationCoefficients,thresholdLineArray):
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