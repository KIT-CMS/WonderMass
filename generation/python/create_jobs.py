#!/usr/bin/env python

# example:
# python generation/python/create_jobs.py -j test_job -n 10 -m 91 -p _test_wm -s -r 1 --submit

import os
import sys


jdl = """\
universe = vanilla
executable = ./node.sh
output = out/$(ProcId).$(ClusterID).out
error = err/$(ProcId).$(ClusterID).err
log = log/$(ProcId).$(ClusterID).log
requirements = (OpSysAndVer =?= "SLCern6")
max_retries = 3
RequestCpus = 1
+MaxRuntime = 28800
notification  = Complete
notify_user   = olena.hlushchenko@desy.de
queue arguments from arguments.txt\
"""
# 1h 3600
# 8h 28800

def mkdir(path):
    print path
    if not os.path.exists(path):
        os.mkdir(path)


def parse_arguments():
    # if not len(sys.argv) >= 2:
    #     raise Exception("./create_job.py PATH_TO_JOBDIR")
    import argparse
    defaultArguments = {}
    parser = argparse.ArgumentParser(description='create_job.py parser')

    parser.add_argument('-j', "--jobdir", type=str, required=True, help="Directory for jobs.")
    parser.add_argument('-f', "--file", type=str, default=None, help="Script to run on the node")
    parser.add_argument('-r', "--repeat", type=int, default=10, help="")
    parser.add_argument('-n', "--num-events", type=str, default="1000", help="")
    parser.add_argument('-t', "--type", type=str, nargs='+', default=["DYtest"], help="")
    parser.add_argument('-m', "--mass", type=str, nargs='+', default=["125"], help="")
    parser.add_argument('-p', "--prefix", type=str, default=None, help="")
    parser.add_argument('-s', "--save-minbias", default=False, action='store_true', help="")
    parser.add_argument("--submit", default=False, action='store_true', help="")

    args = parser.parse_args()
    from six import string_types
    if isinstance(args.type, string_types):
        args.type = [args.type]
    if isinstance(args.mass, string_types):
        args.mass = [args.mass]
    return args


def main(args):
    # Build argument list
    num_events = args.num_events
    print("Number of events per mass point: %s" % (num_events))
    arguments = []
    id_counter = 0
    mass_points = args.mass * args.repeat  # range(50, 200) * 1
    proc = args.type  # ["GGH", "QQH"]

    print("Mass points:", mass_points)
    for mass in mass_points:
        for boson_type in proc:
            l = [str(id_counter), boson_type, mass, num_events]
            if args.prefix is not None:
                l.append(args.prefix)
            if args.save_minbias:
                l.append("1")
            l.append("\n")
            print l
            arguments.append(" ".join(l))
            id_counter += 1
    print("Number of jobs: %u" % len(arguments))

    # Create jobdir and subdirectories
    jobdir = os.path.join(args.jobdir)
    print("Jobdir: %s" % jobdir)
    mkdir(jobdir)
    mkdir(os.path.join(jobdir, "out"))
    mkdir(os.path.join(jobdir, "log"))
    mkdir(os.path.join(jobdir, "err"))

    # Write jdl file
    out = open(os.path.join(jobdir, "job.jdl"), "w")
    out.write(jdl)
    out.close()

    # Write argument list
    arglist = open(os.path.join(jobdir, "arguments.txt"), "w")
    for a in arguments:
        arglist.write(a)
    arglist.close()

    # Write job file
    if args.file is None:
        args.file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../scripts/node.sh")
    jobfile = open(args.file, "r").read()
    job = open(os.path.join(jobdir, "node.sh"), "w")
    job.write(jobfile)
    job.close()

    if args.submit:
        os.chdir(jobdir)
        import subprocess
        subprocess.call('condor_submit job.jdl', shell=True)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
