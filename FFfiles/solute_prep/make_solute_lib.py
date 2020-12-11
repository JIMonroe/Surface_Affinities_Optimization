#Script to loop over all provided directories and load solutes, then save single .top and .itp file with parmed
#After run this, should manually edit molecule names!

import sys
import glob
import parmed as pmd

alldirs = sys.argv[1:]

allsols = pmd.structure.Structure()

for adir in alldirs:
  
  parmlist = glob.glob("%s/*.parm7"%adir)
  rstlist = glob.glob("%s/*.rst7"%adir)

  if len(parmlist) != 1 or len(rstlist) != 1:
    print("Either found too many or not enough parm7 or rst7 files in %s"%adir)
    sys.exit(2)

  parmfile = parmlist[0]
  rstfile = rstlist[0]

  thisstruc = pmd.load_file(parmfile, xyz=rstfile)

  allsols = allsols + thisstruc

print(allsols)

allsols.save("solute_lib.top", parameters="solute_lib.itp", combine=None)


