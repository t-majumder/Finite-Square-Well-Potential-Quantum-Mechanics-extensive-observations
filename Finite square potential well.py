import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#----------------------------------------------------
  #::: SOLVING USING FINITE DIFFERENCE METHOD :::
#----------------------------------------------------

def pothump(a, V0):

    # Proper choice of constants
    hbar=197     #in eV*nm 197 (this is hbar*c)
    m=0.511*1e6  #in eV -----> This is the rest mass energy of an electron (mc**2)
    N = 1000     #Discrete step size  
    b = 10       #Total length (observation length) (in nm)
    x = np.linspace(-b / 2., b / 2., N)
    h = x[1] - x[0]  # step size        
    g=hbar**2/(2*m*a**2) #in units of g
    
#----------------------------------------------------
    # ::: DEFINING THE POTENTIAL FUNCTION :::
#----------------------------------------------------
    
    V = np.zeros(N)
    for i in range(N):
        if x[i] > -a / 2. and x[i] < a / 2.:
            V[i] = V0
            
#----------------------------------------------------
    # ::: SOLVING THE DIFFERENTIAL EQUATION :::
#----------------------------------------------------
    
    #here dd is second order differentiation
    dd = 1. / (h * h) * (-np.diag(np.ones(N - 1), -1) + 2 * np.diag(np.ones(N), 0) - np.diag(np.ones(N - 1), 1))
    H = dd + np.diag(V)

    E, psiT = np.linalg.eigh(H)  # This computes the eigen values and eigenvectors

    E=E*g # Energy in eV
    
    psi = psiT.T *.2  #  (for good looking graph we scale the wave function)
    
    print('\nGiven Potential: ', np.round(V0*g,8), 'eV')
    print(np.round(E[:6],8))
    
#----------------------------------------------------    
            # ::: P L O T T I N G :::
#----------------------------------------------------
    #plt.clf()   # Use for changing a     
    plt.plot(x, V*g, color="Gray") #  <--- This is the potential well
    for i in range(0, 6):
        if E[i] < 0:
            plt.xlim((-a, a))       # Use for changing V
            #plt.xlim((-b/2, b/2))  # Use for changing a
            plt.axhline(E[i], ls='--', color='r')
            plt.title("Wave function of energy levels")
            plt.xlabel(r"Position (in nm) $\longrightarrow$")
            plt.ylabel(r"Energy (in eV) $\longrightarrow$")
            plt.axhline(E[i], ls='--', color='r', label=f'$E_{i}$={E[i]:.3f} eV')
            # Adjust the sign of the wave function to ensure positive sign
            if np.trapz(psi[i], x) < 0:
                psi[i] = -psi[i]
            plt.plot(x, E[i] + psi[i])
            plt.axhline(E[i], ls='--', )
    plt.legend(loc='upper right',fontsize='small')
    plt.show()
    return E,psiT
#----------------------------------------------------
    #     ::: THE ANALYTICAL SOLUTION :::
#----------------------------------------------------
def analytical(V0):
    hbar = 197  # eV*nm
    m = 0.511 * 1e6  # eV
    # Energy range
    E = np.linspace(-V0, 0, 100000)

    # Define functions
    def f1(E, V0):
        return np.sqrt(E + V0) * np.tan(np.sqrt(E + V0)) - np.sqrt(-E)
    def f2(E, V0, eps=1e-10):
        return np.sqrt(E + V0 + eps) / np.tan(np.sqrt(E + V0 + eps)) + np.sqrt(-E)

    # Find zero crossings
    f1s = f1(E, V0)
    f2s = f2(E, V0)

    zero_crossings_even = np.where(np.diff(np.sign(f1s)) * (np.abs(f1s[:-1]) < 3).astype(float))[0]
    zero_crossings_odd = np.where(np.diff(np.sign(f2s)) * (np.abs(f2s[:-1]) < 3).astype(float))[0]
    zero_crossings = np.sort(np.concatenate([zero_crossings_even, zero_crossings_odd]))

    Es_analytical = (E[zero_crossings] + E[zero_crossings + 1]) / 2

    # Calculate eigenvalues
    g = hbar ** 2 / (2 * m*4)
    E_ana = Es_analytical * g
    print("Eigenvalues (in eV):", E_ana)
    return E_ana
E_ana=analytical(70)

'''
#----------------------------------------------------
#-------------------- CHANGING V0 -------------------
#----------------------------------------------------

#Animation with changing time
def update(frame):
    plt.clf()
    pothump(2, -5 * frame)  #5*frame is the depth of the potentail  

fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=np.arange(1, 70), interval=700, repeat=False)
plt.show()



#----------------------------------------------------
#-------------------- CHANGING a  -------------------
#----------------------------------------------------
# Fixed potential (-36 V) and changing the length of the well (parameter 'a')
a_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.2,3.4,3.6,3.8,4]  
#a_values=a_values[::-1]

fig, ax = plt.subplots()
ani = FuncAnimation(fig, pothump, frames=a_values, fargs=(-100,), interval=300, repeat=False)

plt.show()
'''

#----------------------------------------------------------
# ::: Lets look at one single energy level and observe :::
#----------------------------------------------------------
a = 2 ; N = 1000 ; b = 10 
x = np.linspace(-b / 2., b / 2., N)
h = x[1] - x[0]

E,psi=pothump(a, -100) # Calling the defined function to get the result
"""
for j in range(6):
    #error=("Error in calculation for E",j,":",(E[j]-E_ana[j])*100)
    print("Error in calculation for E",j,":",np.round(abs((E[j]-E_ana[j])*100),4),"%")
psi=psi.T # <------ It's already  normalized
plt.show()
count=0
for i in range(0,6):
    if E[i]<0:
        count=count+1
w=count   # <-------- Its the no of bound energies we are getting

# Define functions
def plot_wave_function(a, i):
    plt.xlim((-a, a))
    plt.axhline(0, ls='--', color='r')
    plt.plot(x, psi[i], label="For energy $E_{}$={:>8.3f}".format(i, E[i]))
    plt.title("Wave function of energy levels")
    plt.xlabel(r"Position $\longrightarrow$")
    plt.ylabel(r"$\psi \longrightarrow$")
    plt.grid()

def plot_prob_density(a, i):
    plt.xlim((-a, a))
    plt.axhline(0, ls='--', color='r')
    plt.plot(x, psi[i]**2, label="For energy $E_{}$={:>8.3f}".format(i, E[i]))
    plt.title("Probability Amplitude")
    plt.xlabel(r"Position $\longrightarrow$")
    plt.ylabel(r"$| \psi |^2 \longrightarrow$")
    plt.grid()

# Define a function to create subplots
def create_subplots(a, i):
    plt.figure(figsize=(12, 6))   
    # Subplot 1: Wave function
    plt.subplot(1, 2, 1)
    plt.axvline(-a/2, ls='--', color='k')
    plt.axvline(a/2, ls='--', color='k')
    plot_wave_function(a, i)
    plt.legend(loc='upper right',fontsize='small')
    # Subplot 2: Probability density
    plt.subplot(1, 2, 2)
    plt.axvline(-a/2, ls='--', color='k')
    plt.axvline(a/2, ls='--', color='k')
    plot_prob_density(a, i)
    plt.legend(loc='upper right',fontsize='small')
    #plt.tight_layout()
    plt.show()

for i in range(w):
    create_subplots(a, i)

#verifying the normalization
for i in range(0,w):
    total_probability=np.sum(psi[i]**2)
print("\nTotal Probability: " , np.round(total_probability,w))

#Tunneling probability:
def p_t(x,b,i):
    # Finding the index corresponding to -b/2 and b/2
    index_left = np.abs(x - (-b/2)).argmin()
    index_right = np.abs(x - (b/2)).argmin()
    # Slicing the psi array to get values from -b/2 to b/2
    psi_range = psi[i, index_left:index_right+1]
    # Calculating the tunneling probability
    tunneling_probability = 1 - np.sum(psi_range**2)
    print("Tunneling Probability for energy level E",i," is:" ,np.round(tunneling_probability*100 ,4), "%")
    tunneling_probability=0

for i in range(0,w):
    p_t(x,a,i)
"""
