#Run analysis scripted
import estimate, sim
import argparse
import matplotlib.pyplot as plt
import numpy as np
from betamix import sample_paths, BetaMixture

################################################################################

def parse_options():
    """
    argparse
    """
    parser=argparse.ArgumentParser()

    parser.add_argument("-e", dest='example', action='store_true', help="run a simulated example to test installation" )
    parser.add_argument('-m', '--meta', type=str, default="", help=
                        "meta-information file")
    parser.add_argument('-g', '--geno', type=str, default="", help=
                        "root for eigenstrat format input files")
    parser.add_argument('-s', '--snps', type=str, default="", help=
                        "list of subset of snps to run on")
    parser.add_argument('-o', '--out', type=str, default="", help=
                        "root for output files")
    parser.add_argument('-l', '--lam', type=int, default=5, help=
                        "log10(lambda) to use; default 5")

    return parser.parse_args()

################################################################################

def example(options):
    """
    Run simulated example to test installation
    """
    s_mdl = {"s": [0.01] * 100, "h": [0.5] * 100, "f0": 0.1}
    this_res = sim.sim_and_fit(s_mdl, seed=12345, lam=1e5, Ne=1e4)
    print(this_res["s_hat"])
    if abs(np.mean(this_res["s_hat"])-0.01)<0.005:
        print("Looks ok!")
    else:
        print("Might not be ok")
    return()

################################################################################

def main(options):
    if options.example:
        print("Running example; ignoring all input parameters")
        example(options)
        return()

################################################################################

if __name__=="__main__":
    options=parse_options()
    main(options)
