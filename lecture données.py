from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def lecturedonnees(file, longueur):
    ordonnées = np.loadtxt(file, skiprows=1, usecols=1)
    for i in range(len(ordonnées)//2):
        if (abs(ordonnées[i]-ordonnées[i-1]))/ordonnées[i] > 0.1:
            ordonnées[:i] =0
    ordonnéesbis= np.zeros(int(6.3*len(ordonnées)//longueur+1))
    écart = len(ordonnéesbis)-len(ordonnées)
    ordonnéesbis[écart//2:écart//2+len(ordonnées)] = ordonnées
    conversion =6.3/ len(ordonnéesbis)
    abscisses = np.arange(len(ordonnéesbis)) * conversion
    normalisation = np.loadtxt("rayon 4 1.53.csv", skiprows=1, usecols=1)
    normalisation = normalisation*np.max(ordonnéesbis)/np.max(normalisation)
    for i in range(1, len(ordonnéesbis)//4):
        if ordonnéesbis[i] == 0:
            ordonnéesbis[i] = normalisation[i]-0.1
            if ordonnéesbis[i] <= 0.1:
                ordonnéesbis[i]+= 0.1
        if ordonnéesbis[-i] == 0:
            ordonnéesbis[-i] = normalisation[-i]
    for i in range(len(ordonnéesbis)):
        if ordonnéesbis[i] == 0:
            ordonnéesbis[i] = ordonnéesbis[i-1]
    plt.plot(abscisses, ordonnéesbis)
    plt.show()
#lecturedonnees("h1 5.45.csv",5.45)



def lecturedonneesR(file, longueur):
    ordonnées = np.loadtxt(file, skiprows=1, usecols=1)
    conversion =longueur/ len(ordonnées)  
    absisses = np.arange(len(ordonnées)) * conversion
    plt.plot(absisses, ordonnées)
    plt.show()
#lecturedonneesR("rayon 4 1.53.csv", 1.53)



def doublesigm(x,a,b,c,d):
    return c*(erf((x-a)*d)-erf((x-b)*d))

def fitrayon(file, longueur):
    ordonnées = np.loadtxt(file, skiprows=1, usecols=1)
    conversion =longueur/ len(ordonnées)  
    abscisses = np.arange(len(ordonnées)) * conversion
    popt, pcov = curve_fit(doublesigm, abscisses, ordonnées)
    ordonnéesfit = doublesigm(abscisses, *popt)
    return abscisses, ordonnées, ordonnéesfit, *popt, *pcov
#fitrayon("rayon3 1.31.csv", 1.31)

def fithauteur(file, longueur):
    ordonnées = np.loadtxt(file, skiprows=1, usecols=1)
    for i in range(len(ordonnées)//2):
        if (abs(ordonnées[i]-ordonnées[i-1]))/ordonnées[i] > 0.1:
            ordonnées[:i] =0
    ordonnéesbis= np.zeros( int(6.3*len(ordonnées)//longueur+1))
    écart = len(ordonnéesbis)-len(ordonnées)
    ordonnéesbis[écart//2:écart//2+len(ordonnées)] = ordonnées
    conversion =6.3/ len(ordonnéesbis)
    abscisses = np.arange(len(ordonnéesbis)) * conversion
    normalisation = np.loadtxt("rayon 4 1.53.csv", skiprows=1, usecols=1)
    normalisation = normalisation*np.max(ordonnéesbis)/np.max(normalisation)
    for i in range(1, len(ordonnéesbis)//4):
        if ordonnéesbis[i] == 0:
            ordonnéesbis[i] = normalisation[i]-0.1
            if ordonnéesbis[i] <= 0.1:
                ordonnéesbis[i]+= 0.1
        if ordonnéesbis[-i] == 0:
            ordonnéesbis[-i] = normalisation[-i]
    for i in range(len(ordonnéesbis)):
        if ordonnéesbis[i] == 0:
            ordonnéesbis[i] = ordonnéesbis[i-1]
    popt, pcov = curve_fit(doublesigm, abscisses, ordonnéesbis)
    ordonnéesfit = doublesigm(abscisses, *popt)
    return abscisses, ordonnéesbis, ordonnéesfit, *popt, *pcov
 
 
hauteur = fithauteur("h2 5.3.csv", 5.3)
rayon = fitrayon("rayon2 1.03.csv", 1.03)
incerth= np.sqrt(np.diag(hauteur[7:]))
print('h=',abs(hauteur[3]-hauteur[4]),'incertitude=',np.round(incerth[1]+incerth[2], decimals = 3))
incertr= np.sqrt(np.diag(rayon[7:]))
print('h=',abs(rayon[3]-rayon[4]),'incertitude=',np.round(incertr[1]+incertr[2], decimals = 3))
ax = plt.subplot(211)
ax.plot(hauteur[0], hauteur[1], label="mesures")
ax.plot(hauteur[0], hauteur[2], label="fit")
ax.legend()
ax.set_title("hauteur")
ax = plt.subplot(212)
ax.plot(rayon[0], rayon[1], label="mesures")
ax.plot(rayon[0], rayon[2], label="fit")
ax.legend()
ax.set_title("rayon")
plt.xlabel("position[cm]")
plt.ylabel("Niveau de gris[-]")
plt.show()  
    


