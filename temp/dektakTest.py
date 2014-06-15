import numpy as np
import matplotlib.pyplot as plt
 
threshold = 6

x = np.linspace(0,10,10)
y = np.array([5, 2, 3, 7, 9, 1, 5, 8, 7, 4])
y2 = np.linspace(threshold,threshold,len(x))

plt.plot(y, '.-r')
plt.plot(y2)

#functions that calculate the line equation that goes thorugh 2 points
#Ax+By+C=0
#threshold line equation
#0x+thresholdB+C=0
#y-6=0
#C=6
#0x+1y-threshold=0
#threshold line array
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
    
for i in xrange(0, len(x)):
    if i < (len(x)-1):
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
    plt.plot(decIntersectionPoints[i][0],decIntersectionPoints[i][1],'ro')


plt.show()

#W=|A1B1
#   A2B2|=A1*B2-A2*B1
#Wx=|-C1B1
#    -C2B2|=(-C1)*B2-(-C2)*B1   #determinant [[5, 0], [6, 1]]
#Wy=|A1-C1
#    A2-C2|=A1*(-C2)-A2*(-C1)   #determinant [[4, 5], [1, 6]]
#threshold line

#print A(0,threshold,1,threshold)
#print C(0,threshold,1,threshold)


#Ax+By+C=0
#A=(y_1-y_0)/(x_1-x_0)
#B=1 always
#C=(x_1*y_0-x_0*y_1)/(x_1-x_0)