# Run analysis scripted
import argparse
import gzip
import sys

import numpy as np

import bmws.sim
from bmws.estimate import estimate_em
from bmws.betamix import BetaMixture, sample_paths

def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="subcommands")
    subparsers.required = True
    test = subparsers.add_parser(
        "test",
        help="run a simulated example to test installation",
    )
    test.set_defaults(func=example)

    analyze = subparsers.add_parser(
        "analyze",
        help="analyze data",
    )
    analyze.set_defaults(func=analyze_data)

    analyze.add_argument("vcf", type=str, default="", help="vcf input files")
    analyze.add_argument("meta", type=str, help="meta-information file")
    analyze.add_argument(
        "-o",
        "--out",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="output file",
    )
    analyze.add_argument('-d', '--data', type=str, required=True,
                             choices=["pseudohaploid", "diploid"],
                             help="are your data diploid or pseudohaploid?")
    analyze.add_argument(
        "-l", "--lam", type=float, default=5, help="log10(lambda) to use; default 5"
    )
    analyze.add_argument(
        "-e", "--em", type=int, default=3, help="Number of EM iterations; default 3"
    )
    analyze.add_argument(
        "-g", "--gen", type=float, default=29, help="generation time in years"
    )
    analyze.add_argument(
        "-n", "--Ne", type=float, default=10000, help="Effective population size"
    )
    analyze.add_argument(
        "-t", "--traj", action='store_true', help="output selection trajectory for each SNP; selection coefficeint for the alt allele in generation i is in column 10+i"
    )

    return parser


def example(args):
    """
    Run simulated example to test installation
    """
    s_mdl = {"s": [0.01] * 100, "h": [0.5] * 100, "f0": 0.1}
    this_res = bmws.sim.sim_and_fit(s_mdl, seed=12345, lam=1e5, Ne=1e4, em_iterations=3)
    print(this_res["s_hat"])
    if abs(np.mean(this_res["s_hat"]) - 0.01) < 0.005:
        print("Looks ok!")
    else:
        print("Might not be ok")


def read_meta_information(args):
    """
    Read in the date information
    First column, sample ID
    Second column, date in years_BP
    """
    meta = {}
    maxdate = 0

    with open(args.meta) as metafile:
        for line in metafile:
            if line[0] == "#":
                continue
            else:
                bits = line[:-1].split()
                gen = int(float(bits[1]) / args.gen)
                meta[bits[0]] = gen
                maxdate = max(gen, maxdate)

    meta = {k: maxdate - v for k, v in meta.items()}

    return meta


class vcf:
    """
    read vcf file
    """

    def __init__(self, file):
        """
        Open the vcf file, find the header lines, and load the sample ids
        """
        if file[-3:] == ".gz":
            self.file = gzip.open(file, "rt")
        else:
            self.file = open(file)

        in_header = True
        while in_header:
            line = self.file.readline()
            if line[:2] == "##":
                continue
            elif line[0] == "#":
                self.ids = line[:-1].split()[9:]
                in_header = False

    def __iter__(self):
        return self

    def __next__(self):
        """
        Iterate over remaining lines of file.
        """
        map={"1/1":2, "1/0":1, "0/1":1, "0/0":0, "./.":None}
        line = self.file.readline()

        if not line:
            self.file.close()
            raise StopIteration

        bits = line[:-1].split()
        snpinfo = bits[:5]
        gt = [map[x] for x in bits[9:]]
        return (snpinfo, gt)


def gt_to_obs(ids, gt, meta, data):
    """
    convert gt to an observation list of (Obs, Alt) counts
    data should be either "pseudohaploid" or "diploid"
    """
    maxgen = max(meta.values())
    obs = [[0, 0] for x in range(maxgen + 1)]
    for i, g in enumerate(gt):
        if g is not None:
            gen = meta[ids[i]]
            if data=="diploid" and g in [0,1,2]:
                obs[gen][0]+=2
                obs[gen][1]+=g
            elif data=="pseudohaploid" and g in [0,2]:
                obs[gen][0]+=1
                obs[gen][1]+=g//2
            else:
                raise Exception("Unknown genotype for data\n"+str(gt))

    obs = np.array(obs)
    return obs


def bmws_main(arg_list=None):
    parser = get_parser()
    args = parser.parse_args(arg_list)
    args.func(args)


def analyze_data(args):
    meta = read_meta_information(args)
    data = vcf(args.vcf)
    ids = data.ids

    lam = 10**args.lam
    for snpinfo, gt in data:
        obs = gt_to_obs(ids, gt, meta, args.data)
        Ne = np.full(len(obs) - 1, args.Ne)
        res, prior = estimate_em(obs, Ne, lam=lam, em_iterations=args.em)

        smn = np.mean(res)
        sl1 = np.sqrt(np.mean(res * res))
        sl2 = np.sqrt(np.mean((res - np.mean(res)) ** 2))
        freq = np.sum(obs[:, 1]) / np.sum(obs[:, 0])

        info = [
                    str(round(freq, 3)),
                    str(round(-smn, 6)),
                    str(round(sl1, 6)),
                    str(round(sl2, 6)),
                ]

        if args.traj:
            info = info + [str(round(-f, 6)) for f in res]
            
        print(
            "\t".join(
                snpinfo
                + info
            ),
            file=args.out,
        )
