import numpy as np
import matplotlib.pyplot as plt
import timeit


k=1.38e-16 # erg/K
aB = 2e-13 # cm^3 / s

#a

# here no need for nH nor ne as they cancel out
def equilibrium1(T,Z=0.015,Tc=10**4,psi=0.929):
	"""Returns Gamma_pe - Lambda_rr, divided by nH, ne, alphaB because those appear in both expressions"""
	return psi*Tc*k - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T*k



def NR(func, deriv, c, max_iter, accur):
	"""Uses the Newton-Rhapson method to find the root of func, using the initial guess c and the derivative deriv. It iterates either max_iter times or until f(c) < accur."""
	for i in range(0, max_iter):       
        
        	#estimate the x position of the root
		c = c - func(c)/deriv(c)
        
		print("Step ", i+1, "f(c) =", func(c))
        	#return if accuracy reached
		if np.abs(func(c)) < accur:
			return c
        
	return c



def equilder1(T,Z=0.015):
	"""Returns the derivative of Gamma_pe - Lambda_rr, divided by nH, ne, alphaB because those appear in both expressions"""
	return - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)) - 0.0416)*k


#We want an accuracy of at least 0.1K, but the function is around zero for a long time so we want a high accuracy
#Secant diverges, so I use Newton-Rhapson with 10**3.5 as initial guess

starttime = timeit.default_timer()
T_equil = NR(equilibrium1, equilder1, 10**3.5, 30, 10**(-15))
timetaken = timeit.default_timer() - starttime
print("Time taken with NR", timetaken, " s")
print("T_equil = ", T_equil)
np.savetxt("NUR2Q2TandttNR.txt", [T_equil, timetaken])

#plotting to show that the T is found within 0.1K
T_arr = np.linspace(1,10**7,10000000)
zeroes = np.zeros(10000000)
plt.plot(T_arr,equilibrium1(T_arr), label='equilibrium function')
plt.plot(T_arr, zeroes, label='zero')
plt.scatter(T_equil, equilibrium1(T_equil), label='root by NR')
plt.xscale('log')
plt.xlabel("Temperature (K)")
plt.ylabel("Equilibrium function")
#set xlimits 0.1K around the found equilibrium to see how close we are
plt.xlim(T_equil-1e-1, T_equil+1e-1)
plt.ylim(1e-18, -1e-18)
plt.legend()
plt.savefig("NUR2Q2plot1.pdf")
plt.close()



#nH = ne so we can divide out one
def equilibrium2(T, Z=0.015,Tc=10**4,psi=0.929,  A=5*10**(-10), xi=10**(-15)):
	"""Returns Gamma_pe + Gamma_CR + Gamma_MHD - Lambda_rr - Lambda_FF, divided by ne, because that appear in all expressions"""
	return (psi*Tc - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T - .54 * ( T/1e4 )**.37 * T)*k*nH*aB + A*xi + 8.9e-26 * (T/1e4)

#derivative
def equilder2(T, Z=0.015,Tc=10**4,psi=0.929,  A=5*10**(-10), xi=10**(-15)):
	"""Returns the derivative of Gamma_pe + Gamma_CR + Gamma_MHD - Lambda_rr - Lambda_FF, divided by ne, because that appear in all expressions"""
	return ( - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)) - 0.0416) - .54 * 1.37 * ( T/1e4 )**.37)*k*nH*aB + 8.9e-26 * (1/1e4)


def Bisection(func, a, b, max_iter, accur):
    """Finds the root of func with bisection between a and b, doing either max_iter steps or until f(c) < accur"""
    if func(a) * func(b) > 0:
	#we cannot find a root with bisection this way
        print("invalid!")
        return 0
    #take c in between a and b
    c = (a+b)*0.5 
    
    for i in range(0, max_iter):
        print("step", i, "accur =", func(c))
        #check if accuracy is reached
        if np.abs(func(c)) < accur:
            return c

    	#see whether b and c, or c and a properly bracket the root
        if func(c) * func(b) < 0:
            a = c
        elif func(c) * func(a) < 0:
            b = c
        else:
            print("can't find root")
            return 0 
        #take a new estimate for c with the new bracket
        c = (a+b)*0.5 
        
    return c


def Secant(func, a, b, max_iter, accur):
    """Finds the root of func with secant between a and b, doing either max_iter steps or until f(c) < accur"""
    if func(a) * func(b) > 0:
	#we cannot find a root this way
        print("invalid!")
        return 0
    

    for i in range(0, max_iter):
        #estimate root position
        c = b + func(b)*(b-a)/(func(a) - func(b))
        a,b = b,c
        
        print("Step", i+1, "accur =", func(c))
        #check if accuracy is reached
        if np.abs(func(c)) < accur:
            return c
        
    return c


T_arr2 = np.linspace(1,10**15, 10**7)

nH = 1e-4
#for nH 1e-4 NR diverges and its really steep near the zero, so go use bisection
starttime2_1 = timeit.default_timer()
T_bisequil2_1 = Bisection(equilibrium2, 1, 10**15, 30, 10**(-17))
timetaken2_1 = timeit.default_timer() - starttime2_1
print("Time taken with Bisection", timetaken2_1, " s")
print("T_bisequil = ", T_bisequil2_1)
np.savetxt("NUR2Q2TandttBis.txt", [T_bisequil2_1, timetaken2_1])

plt.plot(T_arr2,equilibrium2(T_arr2), label='equilibrium function')
plt.plot(T_arr2, zeroes, label='zero')
plt.scatter(T_bisequil2_1, equilibrium2(T_bisequil2_1), label='root by bis')
plt.xlabel("Temperature (K)")
plt.ylabel("Equilibrium function")
plt.xscale('log')
#plt.xlim(T_bisequil2_1-1e3, T_bisequil2_1+1e3)
#plt.ylim(1e-17, -1e-17)
plt.legend()
plt.savefig("NUR2Q2plot2.pdf")
plt.close()

#with nH >=1 it is almost always > 0 except at T=1, so use Secant
nH = 1
starttime2_2 = timeit.default_timer()
T_bisequil2_2 = Secant(equilibrium2, 1, 10**15, 30, 10**(-17))
timetaken2_2 = timeit.default_timer() - starttime2_2
print("Time taken with Secant", timetaken2_2, " s")
print("T_equil = ", T_bisequil2_2)
np.savetxt("NUR2Q2TandttSec.txt", [T_bisequil2_2, timetaken2_2])

plt.plot(T_arr2,equilibrium2(T_arr2), label='equilibrium function')
plt.plot(T_arr2, zeroes, label='zero')
plt.scatter(T_bisequil2_2, equilibrium2(T_bisequil2_2), label='root by secant')
plt.xscale('log')
plt.xlabel("Temperature (K)")
plt.ylabel("Equilibrium function")
plt.xscale('log')
plt.xlim(1, T_bisequil2_2+1e3)
#plt.ylim(1e-17, -1e-17)
plt.legend()
plt.savefig("NUR2Q2plot3.pdf")
plt.close()



nH = 1e4
starttime2_3 = timeit.default_timer()
T_bisequil2_3 = Secant(equilibrium2, 1, 10**15, 30, 10**(-17))
timetaken2_3 =  timeit.default_timer() - starttime2_3
print("Time taken with Secant", timetaken2_3, " s")

print("T_equil = ", T_bisequil2_3)
np.savetxt("NUR2Q2TandttSec2.txt", [T_bisequil2_3, timetaken2_3])

plt.plot(T_arr2,equilibrium2(T_arr2), label='equilibrium function')
plt.plot(T_arr2, zeroes, label='zero')
plt.scatter(T_bisequil2_3, equilibrium2(T_bisequil2_3), label='root by secant')
plt.xscale('log')
plt.xlabel("Temperature (K)")
plt.ylabel("Equilibrium function")
plt.xscale('log')
plt.xlim(1, T_bisequil2_3+1e3)
plt.ylim(1e-17, -1e-17)
plt.legend()
plt.savefig("NUR2Q2plot4.pdf")
plt.close()


