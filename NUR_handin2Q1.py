import numpy as np
import matplotlib.pyplot as plt

#simple trapezoid integration
def trapezoid(N, x0, xmax, func):
	"""Trapezoid integration with N steps, integrating func from x0 to xmax"""
	#step size is range divided by number of steps
	h = (xmax-x0)/N
	xes = np.linspace(x0,xmax,N)
	#trapezoid integration formula
	integr = h*(func(xes[0])*0.5 + np.sum(func(xes[1:N-1])) + func(xes[N-1])*0.5)

	return integr

#Romberg integration
def Romberg(N, m, x0, xmax, func):
	"""Romberg integration with N steps, an order of m, integrating func from x0 to xmax"""
	#step size is range divided by number of steps
	h = (xmax-x0)/N
	#r has the size of the order
	r = np.zeros(m)
	#first estimate is simply the trapezoid integration
	r[0] = trapezoid(N, x0, xmax, func)
	
	Np = N
	for i in range(1,m):
		#iteratively improve the integration estimate
		r[i] = 0
		diff = h
		h *= 0.5
		x = x0+h

		for k in range(Np):
			r[i] += func(x)
			x += diff

		
		r_i = r[i].copy()

		r[i] = 0.5*(r[i-1]+diff*r_i)


		Np *= 2



	Np = 1
	for i in range(1,m):
		#combine all estimates into one
		Np *= 4

		for j in range(0,m-i):
			r[j] = (Np*r[j+1] - r[j])/(Np-1)
	#return final Romberg integration value	    
	return r[0]



A=1
Nsat=100
a,b,c = 2.4,0.25,1.6



def n2(x,a=2.4,b=0.25,c=1.6):
	"""Returns n(x)*x^2 /(A*Nsat)"""
	return  4*np.pi * (x**(a-1)/(b**(a-3)))*np.exp(-(x/b)**c)

#we have that x = r/r_vir, so integrating over spherical coordinates gives dx = dr/r_vir, and n(x) doesnt depend on theta or phi so the 3D integral is simply 4pi times the integral over r^2*dr (or r_vir^3*x^2*dx)
#To solve for A we can rewrite the integral to get (4pi/(A*N_sat))*int n(x)*x^2 *dx = 1/(A*r_vir^3)
#a < 3 so we're dividing by zero if x=0, so put the x**2 into it

A_inv = Romberg(50,6,0,5,n2)
A_intgr = 1/A_inv

print("The result from integration with 50 steps and order 6:", A_inv)
print("And A*r_vir^3", A_intgr)

np.savetxt("NUR2Q1Asol.txt", [A_intgr])



#b


#N(x)dx is 4*pi*n(x)*x^2 dx, because integrating that divided by Nsat gives 1, so integrating p(x)dx also gives 1, which is what we want

def N(x,A=A_intgr,a=2.4,b=0.25,c=1.6):
	"""Returns N(x)dx/Nsat"""
	return 4*np.pi* A*(x**(a-1)/(b**(a-3)))*np.exp(-(x/b)**c)

def LCG64(I0,size, a=1664525, m=2**32, c=1013904223, a1=21, a2=25, a3=4):
    """Random number generator using 64-bit XOR and LCG, gives back an array of random numbers between 0 and 1 of size size with I0 as seed."""
    #do 64-bit XOR first with the seed I0
    rndnrs = np.zeros(size,dtype=int)
    xors = np.zeros(size+1,dtype=int)
    xors[0] = int(I0)
    
    for i in range(1,size+1):
	#generate all 64-bit XOR numbers
        x0 = xors[i-1]
        x1 = x0 ^ (x0 >> a1)
        x2 = x1 ^ (x1 << a2)
        x3 = x2 ^ (x2 >> a3)
        
        xors[i] = x3
        
    for k in range(0,size):
    	#the first xors number is still the seed, so do RCG with xors[1:] so we dont use the seed anymore
        rndnrs[k] = (a*xors[k+1] + c)%m
        
        #return between 0 and 1
    return rndnrs/np.amax(rndnrs)



def RejectSamp(prob, a, b, N):
	"""Rejection sampling N points with a probability prob from x = [a,b]"""
	sample = np.zeros(N)
	i=0
	j=0 
	x_range = np.linspace(a,b,N)
	probs = prob(x_range)
	maxprob = np.amax(probs)
	while i < N:
		#make sure that the seed changes every time so we get a different number
		rndnr = LCG64(I0=(j+12), size=50)
		x = a + (b-a)*rndnr[0] #U(a,b)
		y = maxprob*rndnr[1] #y goes from zero to max(p(x))
		
		if y <= prob(x):
			sample[i] = x
			i += 1
			j += 1
		else:
			#reject but still take another seed
			j += 1

	return sample

x_sample = RejectSamp(N, a=0, b=5, N=10000)

xes = np.linspace(0,5,1000)

#setting density=True gives a probability density distribution by dividing the bins by their width and the total number of counts

#the bins below 5*10**(-3) are empty because there the probability reaches ~10^(-4) so with 10000 points you expect at most 1 x there
plt.hist(x_sample, bins=np.logspace(np.log10(10**(-4)),np.log10(5), 20), density=True, rwidth=0.9, label='histogram of sampled x')
plt.xlabel('x')

plt.loglog(xes,N(xes), label='p(x)')
plt.legend()
plt.savefig("NUR2Q1plot1.pdf")
plt.close()



#c


def rowswapvec(M, i, j):
	"""swap indices i and j of a vector M"""
	B = np.copy(M).astype("float64")

	row = B[i].copy()
	B[i] = B[j]
	B[j] = row
	return B

def Quicksort_part(arr, ind_first, ind_last, indsave, indices):
	"""Sorts array arr from ind_first to ind_last by taking a pivot and sorting it with respect to the pivot"""
	#arr is the full array, a is the part of the array we're sorting
	a = arr[ind_first:ind_last+1].copy()
	Ntot = len(arr)
	N = len(a)
	middle = int(N*0.5)
    	#if N < 3, then we dont have to sort
	if N < 3:
		return arr, indices
    	#pivot in the middle
	x_piv = a[middle]

	#i and j are for arr, i_2 and j_2 are for a
	i = ind_first
	j = ind_last
	i_2 = 0
	j_2 = N-1
	#sort arr (which we return) based on a, and sort a too
	for k in range(0,N):
		if i >= j:
	    		break
		if arr[i] >= x_piv:
			pass
		else:
			i += 1
			i_2 += 1
		if arr[j] <= x_piv:
			pass
		else:
			j -= 1
			j_2 -= 1
 		#swap everything (the indices too if indsave is True
		if arr[i] >= x_piv and arr[j] <= x_piv:
			arr = rowswapvec(arr,i,j)
			a = rowswapvec(a,i_2,j_2)
			if indsave:
				indices = rowswapvec(indices,i,j)
                        
	#sort left and right of the pivot

	first = a[0]
	last = a[N-1]

	indxes = np.arange(0,Ntot,1, dtype=int)
	indxes2 = np.arange(0,N,1, dtype=int)
	#get the indices of the new pivot place in both arr and a
	indx_piv = (indxes[arr == x_piv])[0]
	indx_piv2 = (indxes2[a == x_piv])[0]
	in_first = (indxes[arr == first])[0]
	in_last = (indxes[arr == last])[0]
	
	#if the pivot is at the edge, only sort right or left of it, otherwise, sort both sides
	if indx_piv2 <= 1:
		arr, indices = Quicksort_part(arr, indx_piv+1, in_last, indsave, indices)
	if indx_piv2 >= N-2:
		arr, indices = Quicksort_part(arr, in_first, indx_piv, indsave, indices)
	else:
		arr, indices = Quicksort_part(arr, in_first, indx_piv, indsave, indices)
		arr, indices = Quicksort_part(arr, indx_piv+1, in_last, indsave, indices)
        

	return arr, indices

 
            
def Quicksort(arr, indsave=False):
	"""Sorts array arr using the Quicksort algorithm, if indsave is true, we give back the indices of the sorted array instead of the sorted array itself."""
	a = arr.copy()
	N = len(arr)
	middle = int(N*0.5)
	#make an array of indices to keep track of
	indxes = np.arange(0,N,1, dtype=int)
    	#first sort the first middle and last of the array
	fml = np.array([a[0], a[middle], a[N-1]]).copy()
	fml_ind = np.array([0,middle,N-1])
	a[0], indxes[0] = np.amin(fml), fml_ind[fml == np.amin(fml)][0]
        
	a[N-1], indxes[N-1] = np.amax(fml), fml_ind[fml == np.amax(fml)][0]


	if fml[(fml > a[0]) & (fml < a[N-1])].size == 0: 
        	#this means that there are 2 equal numbers 
		pass
	else:
		a[middle] = fml[(fml > a[0]) & (fml < a[N-1])]
		indxes[middle] = fml_ind[fml == fml[(fml > a[0]) & (fml < a[N-1])]]
    
    

	in_first = 0
	in_last = N-1

	#sort the rest
	a, indxes = Quicksort_part(a, in_first, in_last, indsave, indxes)        


	if indsave:
		return indxes
    
	return a
        
#I will draw 100 random galaxies by shuffling the sample array and taking the first 100 of the shuffled array
#shuffle a random array by sorting it and getting the indices
rand_arr = LCG64(I0=55, size=len(x_sample))
rand_ind = Quicksort(rand_arr, indsave=True)

rand_sample = x_sample[rand_ind.astype(int)][0:100]
#sort the random 100 galaxies
sort_sample = Quicksort(rand_sample)

print("Sorted 100 random galaxies", sort_sample)

#to get the number within a certain radius, we can use the sorted sample
#there are 100 samples so counts go from 0 to 100
counts = np.arange(0,101,1)
counts_fin = np.append(counts,100)


xes_samp = np.insert(sort_sample, 0, 10**(-4))
xes_samp = np.insert(xes_samp, len(xes_samp), 5)

plt.plot(xes_samp, counts_fin)
plt.xscale('log')
plt.xlabel("r/r_vir")
plt.ylabel("number of haloes with r < r/r_vir")
plt.savefig("NUR2Q1plot2.pdf")
plt.close()



#d


def centr_diff(func, x, h):
    """calculates the derivative with the central difference method"""
    return (func(x+h) - func(x-h))/(2*h)

def Ridder(func, x, h, d, accur, analder, m=100):
    """calculates the derivative of func at x with stepsize h, order d, up until accuracy accur. The accuracy is calculated with the analytic derivative."""
    #get the Ridder estimates, the first is simply with central difference method
    r = np.zeros((m, len(x)))
    r[0,:] = centr_diff(func,x,h)
    d_inv = 1/d
    
    for i in range(1,m):
	#get the other first estimates
        h *= d_inv
        r[i,:] = centr_diff(func,x,h)
        
    
    
    Np = 1
    for i in range(1,m):
	#just as with Romberg, now we iterate to improve the solution
        Np *= d**2
        
	#before improving the solution, save r_0 and the accuracy in case the accuracy drops
        accurprev = np.mean(np.abs(r[0,:] - analder(x)))
	r_0prev = r[0,:].copy()
        
        for j in range(0,m-i):
            r[j,:] = (Np*r[j+1,:] - r[j,:])/(Np-1)
        
	    
        accurnow = np.mean(np.abs(r[0,:] - analder(x)))
	#if the error grew, give the previous estimate, if not, give the latest estimate
        if accurnow > accurprev:
            print("error grew", accurprev, accurnow)
            return r_0prev
        if accurnow < accur:
            print("Number of steps taken, error:", i, np.mean(np.abs(r[0,:] - analder(x))))
            return r[0,:]
            
    return r[0,:]

def n(x,A=A_intgr,Nsat=100, a=2.4,b=0.25,c=1.6):
	"""Returns n(x)"""
	return Nsat* A*((x/b)**(a-3))*np.exp(-(x/b)**c)

def dn(x,A=A_intgr,Nsat=100, a=2.4,b=0.25,c=1.6):
	"""Returns dn(x)/dx"""
	return Nsat* A*((x)**(a-4)/(b)**(a-3))*np.exp(-(x/b)**c)*(a-3-c*(x/b)**c)

xje = np.ones(1)
analderiv, Ridderderiv = dn(1), Ridder(n,xje,0.1,2,accur=10**(-10),analder=dn)
print("Analytical derivative at x=1, and via Ridder's method:", analderiv, Ridderderiv)

np.savetxt("NUR2Q1derivs.txt", [analderiv, Ridderderiv])

