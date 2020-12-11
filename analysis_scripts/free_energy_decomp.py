

#Script to decompose free energies of solvation into different components
#This should mainly be used for obtaining the decoupled to WCA free energy
#Additionally, this allows for reasonably low-error computation of solute-system
#potential energies for different levels of solute-system interactions, which in
#turn allows for computation of the relative entropy associated with each process.

import pickle
import numpy as np



mbarFile = 'mbar_object.pkl'
alchFile = 'alchemical_U_noXYres.txt'
uDecompFile = 'pot_energy_decomp.txt'

#Load in the mbar object
with open(mbarFile, 'r') as infile:
  mbarObj = pickle.load(infile)

#Read in the alchemical information file
alchDat = np.loadtxt(alchFile)

#Read in the potential energy decomposition data
decompDat = np.loadtxt(uDecompFile)

#The decomposition file should be ordered as fully coupled, no electrostatics, WCA, then decoupled
#Get free energies between these states (less accurate/higher error than from original MBAR)
dGdecomp, dGdecompErr = mbarObj.computePerturbedFreeEnergies(decompDat.T)
print('Raw free energies (and uncertainties) between states computed in energy decomp file:')
print(dGdecomp[0])
print(dGdecompErr[0])

dGwca = dGdecomp[0][2] - dGdecomp[0][-1]
dGwcaErr = np.sqrt(dGdecompErr[0][2]**2 + dGdecompErr[0][-1]**2)
dGljattr = dGdecomp[0][1] - dGdecomp[0][2]
dGljattrErr = np.sqrt(dGdecompErr[0][1]**2 + dGdecompErr[0][2]**2)
dGlj = dGdecomp[0][1] - dGdecomp[0][-1]
dGljErr = np.sqrt(dGdecompErr[0][1]**2 + dGdecompErr[0][-1]**2)
dGq = dGdecomp[0][0] - dGdecomp[0][1]
dGqErr = np.sqrt(dGdecompErr[0][0]**2 + dGdecompErr[0][1]**2)
dGtot = dGdecomp[0][0] - dGdecomp[0][-1]
dGtotErr = np.sqrt(dGdecompErr[0][0]**2 + dGdecompErr[0][-1]**2)
print('Free energy decompositions (less accurate than from check_dGsolv.py):')
print('\tdG_WCA = %f +/- %f (dG to turn on repulsive part of LJ)'%(dGwca, dGwcaErr))
print('\tdG_LJ_attr = %f +/- %f (dG to turn on attractive part of LJ with replusive on)'%(dGljattr, dGljattrErr))
print('\tdG_LJ = %f +/- %f (dG to fully turn on LJ)'%(dGlj, dGljErr))
print('\tdG_Q = %f +/- %f (dG to turn on electrostatics with LJ on)'%(dGq, dGqErr))
print('\tdG_tot = %f +/- %f (dG to turn on fully coupled from decoupled)'%(dGtot, dGtotErr))

#Next want to compute average potential energy differences en route to relative entropies
#In what's below, it's probably more accurate to use the potential energies from alchDat wherever possible
#(i.e. everywhere except for WCA and LJ attractive contributions)
#But for that to work nicely, need to make bulk file the same as for surface file
#Or set a flag
avgUwca, avgUwcaErr = mbarObj.computeExpectations(decompDat[:,2] - decompDat[:,-1], decompDat[:,2])
avgUljattr, avgUljattrErr = mbarObj.computeExpectations(decompDat[:,1] - decompDat[:,2], decompDat[:,1])
avgUlj, avgUljErr = mbarObj.computeExpectations(decompDat[:,1] - decompDat[:,-1], decompDat[:,1])
avgUq, avgUqErr = mbarObj.computeExpectations(decompDat[:,0] - decompDat[:,1], decompDat[:,0])
avgUtot, avgUtotErr = mbarObj.computeExpectations(decompDat[:,0] - decompDat[:,-1], decompDat[:,0])
#avgUlj, avgUljErr = mbarObj.computeExpectations(alchDat[:,1] - alchDat[:,-1], alchDat[:,1])
#avgUq, avgUqErr = mbarObj.computeExpectations(alchDat[:,0] - alchDat[:,1], alchDat[:,0])
#avgUtot, avgUtotErr = mbarObj.computeExpectations(alchDat[:,0] - alchDat[:,-1], alchDat[:,0])
print('Average potential energies in appropriate ensembles:')
print('\t<U_WCA> = %f +/- %f (dU for repulsive part of LJ)'%(avgUwca, avgUwcaErr))
print('\t<U_LJ_attr> = %f +/- %f (dU for repulsive to attractive LJ)'%(avgUljattr, avgUljattrErr))
print('\t<U_LJ> = %f +/- %f (dU for full LJ compared to decoupled)'%(avgUlj, avgUljErr))
print('\t<U_Q> = %f +/- %f (dU for fully coupled compared to no electrostatics)'%(avgUq, avgUqErr))
print('\t<U_tot> = %f +/- %f (dU from fully decoupled to coupled)'%(avgUtot, avgUtotErr))

#Finally, based on what we've just calculated, back out relative entropies for each step
sRwca = dGwca - avgUwca
sRwcaErr = np.sqrt(dGwcaErr**2 + avgUwcaErr**2)
sRljattr = dGljattr - avgUljattr
sRljattrErr = np.sqrt(dGljattrErr**2 + avgUljattrErr**2)
sRlj = dGlj - avgUlj
sRljErr = np.sqrt(dGljErr**2 + avgUljErr**2)
sRq = dGq - avgUq
sRqErr = np.sqrt(dGqErr**2 + avgUqErr**2)
sRtot = dGtot - avgUtot
sRtotErr = np.sqrt(dGtotErr**2 + avgUtotErr**2)
print('Relative entropy decompositions:')
print('\tSrel_WCA = %f +/- %f (relative entropy for decoupled to WCA)'%(sRwca, sRwcaErr))
print('\tSrel_LJ_attr = %f +/- %f (relative entropy for WCA to full LJ)'%(sRljattr, sRljattrErr))
print('\tSrel_LJ = %f +/- %f (relative entropy for decoupled to full LJ)'%(sRlj, sRljErr))
print('\tSrel_Q = %f +/- %f (relative entropy for LJ only to fully coupled)'%(sRq, sRqErr))
print('\tSrel_tot = %f +/- %f (relative entropy for decoupled to coupled)'%(sRtot, sRtotErr))


