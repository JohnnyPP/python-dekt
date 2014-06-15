import matplotlib.pyplot as plt
import numpy as np 
from scipy import stats
from scipy.optimize import curve_fit

#filename = '/home/kolan/mycode/python/dektak/t10_1_1_normal_short.csv'

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
threshold=0.06



yshort=yLevelled
x = x[500:2700]
thresholdLine = np.linspace(threshold,threshold,len(y))


dataLength = len(yLevelled) 
#xDiff = np.delete(x,dataLength-1)   #diff consumes one last element from the array
#
#plt.figure(6)
#d = np.diff(y)
#plt.plot(xDiff,d,'ko', markersize=2)
##plt.plot(d)


#plt.plot(x[500:2700],y[500:2700])
y=yshort[500:2700]

minumum = y.min()
y = y - minumum
#plt.plot(y)
#plt.title('Derivative of y')
#plt.xlabel('Lateral [um]')
#plt.ylabel('Raw Micrometer [um]')
#plt.grid(True)

#plt.show()


#def trapmf(x,a,b,c,d,h):
#    if x<=a:
#        return 0.0
#    if x>=a and x<=b:
#        return float((x-a)/(b-a))
#    if x>=b and x<=c:
#        return h
#    if x>=c and x<=d:
#        return float((d-x)/(d-c))
#    if d<=x:
#        return 0.0

def trapmf(x,a,b,a1,b1,c1,d1,c,d,h):
    if x<=a:
        return 0.0
    if x>=a and x<=b:
        return float((x-a)/(b-a))
    if x>=a1 and x<=b1:
        return float((x-a1)/(b-a1))
    if x>=b and x<=c:
        return h
    if x>=c1 and x<=d1:
        return float((d1-x)/(d1-c1))
    if x>=c and x<=d:
        return float((d-x)/(d-c))
    if d<=x:
        return 0.0

######################################
##Fitting
######################################

trapmf = np.vectorize(trapmf) 

# Define model functions to be used to fit to the data above:
#def gauss(x, A, mu, sigma):
#    return A*np.exp(-(x-mu)**2/(2.*sigma**2))
#
#def doubleGauss(x, A, A2, mu, mu2, sigma, sigma2):
#    return A*np.exp(-(x-mu)**2/(2.*sigma**2))+A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))

#p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
#p0 = [0.01, -0.006,  0.4]
#p0 is the initial guess for the fitting coefficients (A, A2, mu, mu2, sigma, sigma2 above)
#p0 = [560, 771,  1670, 1824, 6]

p0 = [51, 104, 108, 128, 215, 232, 233, 250, 6]

popt2, pcov2 = curve_fit(trapmf, x, y, p0=p0)
print popt2
print pcov2

# Get the fitted curve
#yFit = gauss(x, *popt)
yFit = trapmf(x, *popt2)


def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value**2

#def trapmf1(x):
#    if x<=popt2[0]:
#        return 0.0
#    if x>=popt2[0] and x<=popt2[1]:
#        return float((x-popt2[0])/(popt2[1]-popt2[0]))
#    if x>=popt2[1] and x<=popt2[2]:
#        return popt2[4]
#    if x>=popt2[2] and x<=popt2[3]:
#        return float((popt2[3]-x)/(popt2[3]-popt2[2]))
#    if popt2[3]<=x:
#        return 0.0

#p0 = [p0, p1, p2, p3, p4, p5, p6, p7, p8]
#p0 = ,a,  a1, b,  b1, c1, c,  d1, d,  h):

def trapmf1(x):
    if x<=popt2[0]:
        return 0.0
    if x>=popt2[0] and x<=popt2[1]:
        return float((x-popt2[0])/(popt2[1]-popt2[0]))
    if x>=popt2[2] and x<=popt2[3]:
        return float((x-popt2[2])/(popt2[1]-popt2[2]))
    if x>=popt2[1] and x<=popt2[6]:
        return popt2[8]
    if x>=popt2[4] and x<=popt2[5]:
        return float((popt2[5]-x)/(popt2[5]-popt2[4]))
    if x>=popt2[6] and x<=popt2[7]:
        return float((popt2[7]-x)/(popt2[7]-popt2[6]))
    if popt2[7]<=x:
        return 0.0
        
#def trapmf1(x):
#    if x<=popt2[0]:
#        return 0.0
#    if x>=popt2[0] and x<=popt2[1]:
#        return float((x-popt2[0])/(popt2[1]-popt2[0]))
#    if x>=popt2[0] and x<=popt2[1]:
#        return float((x-popt2[0])/(popt2[1]-popt2[0]))
#    if x>=popt2[1] and x<=popt2[2]:
#        return popt2[4]
#    if x>=popt2[2] and x<=popt2[3]:
#        return float((popt2[3]-x)/(popt2[3]-popt2[2]))
#    if x>=popt2[2] and x<=popt2[3]:
#        return float((popt2[3]-x)/(popt2[3]-popt2[2]))
#    if popt2[3]<=x:
#        return 0.0

trapmf1 = np.vectorize(trapmf1) 

#    if x<=a0:
#        return 0.0
#    if x>=a0 and x<=b1:
#        return float((x-a0)/(b1-a0))
#    if x>=b1 and x<=c2:
#        return h4
#    if x>=c2 and x<=d3:
#        return float((d3-x)/(d3-c2))
#    if d3<=x:
#        return 0.0

#def gauss2(x):
#    return popt2[1]*np.exp(-(x-popt2[3])**2/(2.*popt2[5]**2))


plt.figure('1st ALD cycle')
plt.plot(x, y, 'bo', label='Data')
plt.plot(x, yFit, linewidth=2, label='Fit', color='red')
plt.plot(x, trapmf1(x), label='SiO$2$', color='green')
#plt.plot(x, gauss2(x), label='HfO$2$', color='blue')
#plt.fill_between(x, gauss2(x), color='blue', alpha=0.3)
#plt.fill_between(x, gauss1(x), color='green', alpha=0.3)
plt.title('Raw data plot')
plt.xlabel('Height [nm]')
plt.ylabel('Frequency [counts]')
plt.grid(True)
plt.legend()
print rsquared(y, yFit)
plt.show()