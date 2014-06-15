import numpy as np 
import matplotlib.pyplot as plt

filename = '/home/kolan/mycode/python/dektak/t1_1_12_normal.csv'



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
    
    return FoundPosition


x=np.loadtxt(filename,dtype=float,delimiter=',',skiprows=FindHeaderLength(),usecols=(0,))
y=np.loadtxt(filename,dtype=float,delimiter=',',skiprows=FindHeaderLength(),usecols=(1,))
t=np.loadtxt(filename,dtype=float,delimiter=',',skiprows=FindHeaderLength(),usecols=(2,))

plt.figure(1)
plt.plot(x,y)
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
plt.plot(x,t)
plt.title('Trace Filtered Data Micrometer plot')
plt.xlabel('Lateral [um]')
plt.ylabel('Trace Filtered Data [um]')
plt.grid(True)

plt.show()