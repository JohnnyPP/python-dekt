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
#y = np.array([5, 2, 3, 7, 9, 1, 5, 8, 7, 4])    # y test data
#y = np.array([7,8,7,8,9, 1, 5, 8, 7, 4])
y = np.array([1,5, 8, 7, 4])

y2 = np.linspace(threshold,threshold,len(x))    # threshold line

plt.plot(y, '.-r')
plt.plot(y2)

thresholdLineArray = np.array([0,1,threshold])

#increasing
p = next(ii for ii,v in enumerate(y) if (v>=threshold))
frac=p-1.0+(float(threshold) - y[p-1])/(y[p]-y[p-1])
#print p
print frac

y=y[p:]

plt.figure(2)
plt.plot(y, '.-r')
plt.plot(y2)

p = next(ii for ii,v in enumerate(y) if (v>=threshold))
frac=p-1.0+(float(threshold) - y[p-1])/(y[p]-y[p-1])
#print p
print frac


#decreasing
#p = next(ii for ii,v in enumerate(y) 
#if (v<=threshold))
#frac=p-1.0+(float(threshold) - y[p-1])/(y[p]-y[p-1])
#print p



#print frac


plt.show()