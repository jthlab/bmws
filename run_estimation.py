#Run analysis scripted
import estimate, sim
import argparse, gzip
import matplotlib.pyplot as plt
import numpy as np
from betamix import sample_paths, BetaMixture
from scipy.stats import beta
import pdb

################################################################################

def parse_options():
    """
    argparse
    """
    parser=argparse.ArgumentParser()

    parser.add_argument("-e", dest='example', action='store_true', help="run a simulated example to test installation" )
    parser.add_argument('-m', '--meta', type=str, default="", help=
                        "meta-information file")
    parser.add_argument('-v', '--vcf', type=str, default="", help=
                        "vcf input files")
    parser.add_argument('-o', '--out', type=str, default="", help=
                        "output file")
    parser.add_argument('-l', '--lam', type=float, default=5, help=
                        "log10(lambda) to use; default 5")
    parser.add_argument('-e', '--em', type=int, default=3, help=
                        "Number of EM iterations; default 3")
    parser.add_argument('-g', '--gen', type=float, default=29, help=
                        "generation time in years")
    parser.add_argument('-n', '--Ne', type=float, default=10000, help=
                        "Effective population size")

    return parser.parse_args()

################################################################################

def example(options):
    """
    Run simulated example to test installation
    """
    s_mdl = {"s": [0.01] * 100, "h": [0.5] * 100, "f0": 0.1}
    this_res = sim.sim_and_fit(s_mdl,seed=12345, lam=1e5, Ne=1e4, em_iterations=3)
    print(this_res["s_hat"])
    if abs(np.mean(this_res["s_hat"])-0.01)<0.005:
        print("Looks ok!")
    else:
        print("Might not be ok")
    return()

################################################################################

def read_meta_information(options):
    """
    Read in the date information
    First column, sample ID
    Second column, date in years_BP
    """
    meta={}
    maxdate=0
    
    with open(options.meta) as metafile:
        for line in metafile:
            if line[0]=="#":
                continue
            else:
                bits=line[:-1].split()
                gen=int(float(bits[1])/options.gen)
                meta[bits[0]]=gen
                maxdate=max(gen, maxdate)

    meta={k:maxdate-v for k,v in meta.items()}

    return(meta)

################################################################################

class vcf():
    """
    read vcf file
    """
    def __init__(self,  file):
        """
        Open the vcf file, find the header lines, and load the sample ids
        """
        if file[-3:]==".gz":
            self.file=gzip.open(file, "rt")
        else:
            self.file=open(file, "r")

        in_header=True
        while in_header:
            line=self.file.readline()
            if line[:2]=="##":
                continue
            elif line[0]=="#":
                self.ids=line[:-1].split()[9:]
                in_header=False

    def __iter__(self):
        return(self)
                
    def __next__(self):
        """
        Iterate over remaining lines of file. 
        """
        map={"1/1":1, "0/0":0, "./.":None}
        line=self.file.readline()

        if not line:
            self.file.close()
            raise StopIteration
        
        bits=line[:-1].split()
        snpinfo=bits[:5]
        gt=[map[x] for x in bits[9:]]
        return(snpinfo,gt)

################################################################################

def gt_to_obs(ids, gt, meta):
    """
    convert gt to an observation list of (Obs, Alt) counts
    """
    
    maxgen=max(meta.values())
    obs=[[0,0] for x in range(maxgen+1)]
    for i,g in enumerate(gt):
        if g!=None:
            gen=meta[ids[i]]
            obs[gen][0]+=1
            obs[gen][1]+=g

    obs=np.array(obs)
    return(obs)

################################################################################

def main(options):
    if options.example:
        print("Running example; ignoring all input parameters")
        example(options)
        return()

    meta=read_meta_information(options)
    data=vcf(options.vcf)
    ids=data.ids

    lam=10 ** options.lam
    
    for snpinfo,gt in data:
        obs=gt_to_obs(ids, gt, meta)
        Ne = np.full(len(obs) - 1, options.Ne)
        prior=makeprior(obs)
        res, prior = estimate.estimate_em(obs, Ne, lam=lam, em_iterations=options.em_iterations)

        sl1=np.sqrt(np.mean(res*res))
        sl2=np.sqrt(np.mean((res-np.mean(res))**2))
        freq=np.sum(obs[:,1])/np.sum(obs[:,0])
        
        print("\t".join(snpinfo+[str(round(freq,3)), str(round(sl1,6)), str(round(sl2,6))]))
                
################################################################################

if __name__=="__main__":
    options=parse_options()
    main(options)
