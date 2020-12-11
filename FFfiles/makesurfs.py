import numpy as np
import parmed as pmd
from genetic_lib import *
CH3chain = pmd.load_file('/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM.top', xyz='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM_tilted.gro')
OHchain = pmd.load_file('/home/jmonroe/Surface_Affinities_Project/FFfiles/otmSAM.top', xyz='/home/jmonroe/Surface_Affinities_Project/FFfiles/otmSAM_tilted.gro')
chainStructs = [CH3chain, OHchain]
ch3list = np.zeros(48, dtype=bool)
ohlist = np.ones(48, dtype=bool)
halfpatch = np.zeros(48, dtype=bool)
patchinds = np.arange(0, 48, 8)
for ind in patchinds:
  halfpatch[ind:ind+4] = True

halfspread = np.zeros(48, dtype=bool)
spreadinds = np.arange(0, 48, 16)
for ind in spreadinds:
  halfspread[ind:ind+8] = True

#ch3surf = createSAMSurf(chainStructs, ch3list, 'gen0_fullCH3_6-8', chainsX=6, chainsY=8, latticea=4.97, doMinRelax=False)
#ohsurf = createSAMSurf(chainStructs, ohlist, 'gen0_fullOH_6-8', chainsX=6, chainsY=8, latticea=4.97, doMinRelax=False)
#patchsurf = createSAMSurf(chainStructs, halfpatch, 'gen0_half_patch_6-8', chainsX=6, chainsY=8, latticea=4.97, doMinRelax=False)
#spreadsurf = createSAMSurf(chainStructs, halfspread, 'gen0_half_spread_6-8', chainsX=6, chainsY=8, latticea=4.97, doMinRelax=False)

SO3chain = pmd.load_file('/home/jmonroe/Surface_Affinities_Project/FFfiles/stmSAM.top', xyz='/home/jmonroe/Surface_Affinities_Project/FFfiles/stmSAM_tilted.gro')
NC3chain = pmd.load_file('/home/jmonroe/Surface_Affinities_Project/FFfiles/ntmSAM.top', xyz='/home/jmonroe/Surface_Affinities_Project/FFfiles/ntmSAM_tilted.gro')
chargeStructs = [CH3chain, SO3chain, NC3chain]
chargepatchpatch = np.zeros(48, dtype=int)
chargepatchspread = np.zeros(48, dtype=int)
chargespreadpatch = np.zeros(48, dtype=int)
chargespreadspread = np.zeros(48, dtype=int)
chargepatchpatch[[0, 1, 8, 9, 10, 16, 17, 18]] = 1
chargepatchpatch[[24, 25, 26, 32, 33, 34, 40, 41]] = 2
chargepatchspread[[0, 1, 16, 17, 18, 32, 33, 34]] = 1
chargepatchspread[[8, 9, 10, 24, 25, 26, 40, 41]] = 2
chargespreadpatch[[0, 2, 9, 11, 24, 26, 33, 35]] = 1
chargespreadpatch[[4, 6, 13, 15, 28, 30, 37, 39]] = 2
chargespreadspread[[0, 4, 9, 13, 24, 28, 33, 37]] = 1
chargespreadspread[[2, 6, 11, 15, 26, 30, 35, 39]] = 2

chargepatchpacthsurf = createSAMSurf(chargeStructs, chargepatchpatch, 'gen0_charge_patch_patch_6-8', chainsX=6, chainsY=8, latticea=4.97, doMinRelax=False)
chargepatchspreadsurf = createSAMSurf(chargeStructs, chargepatchspread, 'gen0_charge_patch_spread_6-8', chainsX=6, chainsY=8, latticea=4.97, doMinRelax=False)
chargespreadpatchsurf = createSAMSurf(chargeStructs, chargespreadpatch, 'gen0_charge_spread_patch_6-8', chainsX=6, chainsY=8, latticea=4.97, doMinRelax=False)
chargespreadspreadsurf = createSAMSurf(chargeStructs, chargespreadspread, 'gen0_charge_spread_spread_6-8', chainsX=6, chainsY=8, latticea=4.97, doMinRelax=False)


