#!/usr/bin/env python

# example:
'''
python ntupleBuilder/python/create_jobs_ntuplizer.py \
    -j test_SUSY \
    --nicknames SUSYGluGluToHToTauTauM800_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 \
    -o /eos/user/o/ohlushch/Nostradamas/mass_regression/Ofiicial \
    --output_tmp /afs/cern.ch/work/o/ohlushch/temp_Ofiicial -f \
    --submit

    -n 10 \

python Kappa/Skimming/scripts/DatasetManager.py -i datasets.json   --nicks "SUSYGluGluToHToTauTauM800_RunIISummer16MiniAODv3_PUMoriond17_13TeV_MINIAOD_pythia8_v2" --print
python Kappa/Skimming/scripts/DatasetManager.py -i datasets.json   --nicks "SUSYGluGluToHToTauTauM800_*2017*" --print
python Kappa/Skimming/scripts/DatasetManager.py -i datasets.json --dbsregex "/SUSYGluGluToHToTauTau.*/RunIIFall17.*/.*"  --print

python Kappa/Skimming/scripts/DatasetManager.py -i datasets.json --dbsregex "/SUSYGluGlu.*/RunIIFall17.*/.*"  --print
python Kappa/Skimming/scripts/DatasetManager.py -i datasets.json --dbsregex "/SUSYGluGluToBBHToTauTauM.*/RunIIFall17.*/.*"  --print

python ntupleBuilder/python/create_jobs_ntuplizer.py \
    -j SUSY_17_ggH_qqH \
    --nicknames SUSYGluGluToHToTauTauM110_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM300_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM100_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM400_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM250_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM450_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM1500_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM2300_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM2600_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM2000_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM200_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM1800_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM1200_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM1600_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM140_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM1400_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM130_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM80_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM700_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM350_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM120_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM800_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM3200_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM90_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM180_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM900_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM600_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToHToTauTauM2900_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM1000_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM250_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM130_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM1400_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM1800_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM1400_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM100_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM90_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM600_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM2300_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM200_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM160_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM180_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM300_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM1800_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM350_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM80_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM2600_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM140_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM3200_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM800_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM700_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM1200_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM800_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM160_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM400_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM1500_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM120_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM900_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM700_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM200_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM1200_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM250_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM120_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM90_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM2000_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM300_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM900_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM3200_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM2300_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM600_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM130_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM2900_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM125_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM400_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM350_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM1600_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM450_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM2000_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM2600_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM110_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM500_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM140_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM180_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 SUSYGluGluToBBHToTauTauM110_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_pythia8_v1 SUSYGluGluToBBHToTauTauM450_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_amcatnlo-pythia8_v1 \
     -o /eos/user/o/ohlushch/Nostradamas/mass_regression/Ofiicial \
    --output_tmp /afs/cern.ch/work/o/ohlushch/temp_Ofiicial -f \
    --submit

'''
import os
import sys


jdl_local = """\
universe = vanilla
executable = ./node.sh
output = out/$(ProcId).$(ClusterID).out
error = err/$(ProcId).$(ClusterID).err
log = log/$(ProcId).$(ClusterID).log
requirements = (OpSysAndVer =?= "SLCern6")
getenv = true
max_retries = 3
RequestCpus = 1
+MaxRuntime = 28800
notification  = Complete
notify_user   = olena.hlushchenko@desy.de
transfer_output_files = ""
queue arguments from arguments.txt\
"""
# 1h 3600
# 8h 28800


def mkdir(path):
    print path
    if not os.path.exists(path):
        os.makedirs(path) #thanks to the exist_ok flag this will not even complain if the directory exists


def replaceInFile(oldfilename, newfilename, find_replace):
    # create a dict of find keys and replace values

    with open(oldfilename) as data:
        with open(newfilename, 'w') as new_data:
            for line in data:
                for key in find_replace:
                    if key in line:
                        line = line.replace(key, find_replace[key])
                new_data.write(line)
    # import fileinput
    # with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
    #     for line in file:
    #         print(line.replace(origin, replacement))


def parseFilelist(filelist1):
    replacement = 'files = [\n'
    for i in filelist1:
        replacement += "'" + i + "',\n"
    replacement += ']\n'

    return replacement



def parse_arguments():
    # if not len(sys.argv) >= 2:
    #     raise Exception("./create_job.py PATH_TO_JOBDIR")
    import argparse
    defaultArguments = {}
    parser = argparse.ArgumentParser(description='create_job.py parser')

    parser.add_argument('-j', "--jobdir", type=str, required=True, help="Directory for jobs.")
    parser.add_argument('-f', "--force", default=False, action='store_true', help="")
    # parser.add_argument('-r', "--repeat", type=int, default=10, help="")
    parser.add_argument('-n', "--num-events", type=str, default="1000", help="")
    # parser.add_argument('-t', "--type", type=str, nargs='+', default=["DYtest"], help="")
    # parser.add_argument('-m', "--mass", type=str, nargs='+', default=["125"], help="")
    parser.add_argument("--nicknames", type=str, nargs='*', default=[], help="")
    parser.add_argument('-o', "--output", type=str, required=True, help="Location of output ntuple")
    parser.add_argument("--output_tmp", type=str, default='/afs/cern.ch/work/o/ohlushch/temp_Ofiicial', help="Location of output ntuple")

    parser.add_argument('-p', "--prefix", type=str, default='""', help="")
    # parser.add_argument('-s', "--save-minbias", default=False, action='store_true', help="")
    # parser.add_argument('--local', default=False, action='store_true', help="")
    parser.add_argument("--submit", default=False, action='store_true', help="")

    args = parser.parse_args()
    from six import string_types
    # if isinstance(args.type, string_types):
    #     args.type = [args.type]
    # if isinstance(args.mass, string_types):
    #     args.mass = [args.mass]
    return args


def main(args):
    # Build argument list
    num_events = args.num_events
    print("Number of events per mass point: %s" % (num_events))
    arguments = []
    id_counter = 0
    # mass_points = args.mass * args.repeat  # range(50, 200) * 1
    # proc = args.type  # ["GGH", "QQH"]

    # access datasets.json
    import json
    with open('datasets/datasets.json') as datasets_json:
        datasets = json.load(datasets_json)
    # output_ntuple = os.path.join(args.output, str(datasets[args.nickname]['year']), args.nickname + '.root')
    # tmp_output_ntuple = os.path.join(args.output_tmp, str(datasets[args.nickname]['year']), args.nickname + '.root')

    print("Nicknames:", args.nicknames)
    for nickname in args.nicknames:
            print nickname
            print datasets[nickname]
            output_ntuple = os.path.join(args.output, str(datasets[nickname]['year']), nickname + '.root')
            tmp_output_ntuple = os.path.join(args.output_tmp, str(datasets[nickname]['year']), nickname + '.root')
            for obj in [output_ntuple, tmp_output_ntuple]:
                if os.path.exists(obj) and not args.force:
                    print(obj + ' exist and overriten is not set')
                    exit(1)

            mkdir(os.path.join(args.output_tmp, str(datasets[nickname]['year'])))
            mkdir(os.path.join(args.output, str(datasets[nickname]['year'])))
            # get the list of input files for the dataset
            from dbs.apis.dbsClient import DbsApi
            url = "https://cmsweb.cern.ch/dbs/prod/global/DBSReader"
            api = DbsApi(url=url)
            das_dataset = datasets[nickname]['dbs']   # full path to a non-wildcarded dataset
            filelist_str = ','.join([i['logical_file_name'] for i in api.listFiles(dataset=das_dataset)])

            l = [str(id_counter), nickname, filelist_str, tmp_output_ntuple, output_ntuple]
            l.append("\n")
            print l
            arguments.append(" ".join(l))
            id_counter += 1
    print("Number of jobs: %u" % len(arguments))

    # Create jobdir and subdirectories
    # if args.jobdir is None:
    #     args.jobdir = args.nickname
    if args.jobdir[0] != '/' and args.jobdir[0] != '.' and not args.jobdir.startswith('jobsdirs/ntuplizer'):
        jobdir = os.path.join('jobs_dirs/ntuplizer', args.jobdir)
    else:
        jobdir = os.path.join(args.jobdir)
    print("Jobdir: %s" % jobdir)  # jobsdirs/ntuplizer/<name>
    # /afs/cern.ch/work/o/ohlushch/temp_Ofiicial
    # for obj in [output_ntuple, tmp_output_ntuple]:
    #     if os.path.exists(obj) and not args.force:
    #         print(obj + ' exist and overriten is not set')
    #         exit(1)
    mkdir(jobdir)
    mkdir(os.path.join(jobdir, "out"))
    mkdir(os.path.join(jobdir, "log"))
    mkdir(os.path.join(jobdir, "err"))

    # get the list of input files for the dataset
    # from dbs.apis.dbsClient import DbsApi
    # url = "https://cmsweb.cern.ch/dbs/prod/global/DBSReader"
    # api = DbsApi(url=url)
    # das_dataset = datasets[args.nickname]['dbs']   # full path to a non-wildcarded dataset
    # filelist1 = [i['logical_file_name'] for i in api.listFiles(dataset=das_dataset)]

    # copy the run file and modify it
    # from shutil import copyfile
    # copyfile('ntupleBuilder/python/run_test_cfi.py', jobdir)
    runfile = os.path.join(jobdir, 'run_test_cfi.py')
    # replacement = parseFilelist(filelist1)
    repl_dict = {
        # '#__filelist': replacement,
        # '#__settype': 'settype = ' + '_'.join([datasets[args.nickname]['process'], datasets[args.nickname]['extension'], datasets[args.nickname]['version']])
        # 'tmpOutputFile = "ntuple.root"': 'tmpOutputFile = "%s"' % (tmp_output_ntuple),
    }
    replaceInFile(oldfilename='ntupleBuilder/python/run_test_cfi.py', newfilename=runfile, find_replace=repl_dict)
    # print 'settype = ' + '_'.join([datasets[args.nickname]['process'], datasets[args.nickname]['extension'], datasets[args.nickname]['version']])


    # Write jdl file
    out = open(os.path.join(jobdir, "job.jdl"), "w")
    out.write(jdl_local)
    out.close()

    # Write argument list
    arglist = open(os.path.join(jobdir, "arguments.txt"), "w")
    for a in arguments:
        arglist.write(a)
    arglist.close()

    # Write job file

    node_file_tmpl = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../scripts/node_local.sh")
    job = os.path.join(jobdir, "node.sh")
    replaceInFile(oldfilename=node_file_tmpl, newfilename=job,
        find_replace={
            'jobs_dirs/ntuplizer/test_SUSY/run_test_cfi.py': runfile,
            'mv tmp_output_ntuple eosspace': 'mv %s %s' % (tmp_output_ntuple, output_ntuple),
        }
    )
    # jobfile = open(node_file_tmpl, "r").read()
    # job = open(os.path.join(jobdir, "node.sh"), "w")
    # job.write(jobfile)
    # job.close()

    if args.submit:
        os.chdir(jobdir)
        import subprocess
        subprocess.call('pwd', shell=True)
        print 'ls'
        subprocess.call('ls', shell=True)
        print 'cat'
        subprocess.call('cat job.jdl', shell=True)
        print 'run'
        subprocess.call('condor_submit job.jdl', shell=True)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
