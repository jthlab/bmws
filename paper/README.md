This repository contains code to run the model and perform the data
analysis described in "Direct detection of natural selection in Bronze
Age Britain". All python code requires version >= 3.9.1 to run, apart
from the plotting scripts which were written in R version 4.0.2, but may
work on earlier versions. 

### Model code

### Simulations

Simulations reported in the paper can be found in [simulations.ipynb](
simulations.ipynb), which will generate Figure 3.

### Data analysis

- Genotype data and intermediate files can be downloaded from [this
  link](https://upenn.box.com/s/qtf6s2504ib5yjm20w740xbp32cd5jck). Untar the directory and move its contents to the empty data/
  directory in this repository. The README in the directory describes
  the files and their sources. Please read that file before using
  these data.

- The genome-wide analysis can be run using the script
  [run_estimation.py](scripts/run_estimation.py) (run with -h option to see arguments). You could run the entire
  analysis described in the paper using the command `python run_estimation.py -v
  data/allbrit.vcf.gz -m data/allbrit.meta -l 4.5 -o out`. The
  analysis can easily be parallelized by splitting the vcf file into
  chunks. On our servers the analysis took about 10,000 CPU-hours to
  run, so we don't reccomend running in one go. For convenience, we
  also provide the results of this analysis in the file
  data/s_scan_all_brit.txt.gz, and the randomized results in the file
  data/s_scan_all_brit_random.txt.gz.

- You can generate Figures 2, 4, 5 and 6 from the manuscript by running the following
scripts:
  - [plot_brit.R](plot_brit.R)
  - [plot_scan.R](plot_scan.R)
  - [LCT_data_analyis.ipynb](LCT_data_analyis.ipynb)
  - [test_poly.R](test_poly.R)
