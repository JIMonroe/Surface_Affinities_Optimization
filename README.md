# Surface_Affinities_Optimization
Genetic algorithm optimization of surface affinities of small, neutral solutes through chemical group re-patterning

This repository contains code associated with the publication:

Monroe, J. I.; Jiao, S.; Davis, J.; Robinson Brown, D.; Katz, L.; Shell, M. S. Affinity of small-molecule solutes to hydrophobic, hydrophilic, and chemically patterned interfaces in aqueous solution. _PNAS_. January 5, 2021 118 (1) e2020205118; https://doi.org/10.1073/pnas.2020205118

If you use the code in this repository, culminating in a publication, please cite the above paper.

Note that the Jupyter Notebook for generating figures is only for documentation -- it will not run without data generated in the proper directory structure and with the proper names. While the code provided is capable of achieving this, data is also available upon request. It is not directly provided here as it requires more storage capacity than is appropriate for this repository alone.

Additionally, much of the analysis code makes use of a fortran library called waterlib.f90.  For everythin got run, this will need to be compiled using an appropriate fortran compiler in combination with f2py to create a module that can be loaded.  Additionally, the `library` directory may need to be added to your python path.

Questions or other communication may be directed to shell@engineering.ucsb.edu or jacob.ian.monroe@gmail.com.
