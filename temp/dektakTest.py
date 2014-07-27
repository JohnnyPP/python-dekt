import numpy as np
import matplotlib.pyplot as plt

def coefA(x0,y0,x1,y1):
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
    return -(y1-y0)/(x1-x0)

def coefC(x0,y0,x1,y1):
    """
    Finds the coefficient C of the line going thorugh the two points P0 (x0,y0),
    P1 (x1,y1). Run dektakTest.py for a simple example
    :param x0: 
    :param y0:
    :param x1:
    :param y1:
    :return: coefC = -(x1 * y0 - x0 * y1) / (x1 - x0)
    """
    return (x1*y0-x0*y1)/(x1-x0)

threshold = 6  # this may be changed

x = np.linspace(0,10,10)                        # x test data
y = np.array([5, 2, 3, 7, 9, 1, 5, 8, 7, 4])    # y test data
y2 = np.linspace(threshold,threshold,len(x))    # threshold line

plt.plot(y, '.-r')
plt.plot(y2)

thresholdLineArray = np.array([0,1,threshold])

increasingPoints = []
incLineEquationCoefficients = []
incIntersectionPoints = []
decLineEquationCoefficients = []
decreasingPoints = []
decIntersectionPoints = []


"""
Finds the intersection points for a given x, y data that lie on a threshold
line. This is a kind of interlopation of the x data as the threshold is
given for the y axis.
"""


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