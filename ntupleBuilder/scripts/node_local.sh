#!/bin/bash

sen() {
    if [[ $# -eq 0 ]] ; then
        echo  "pass" | mail -s "test wm" "greyxray@gmail.com"
    else
        echo $l | mail -s "test wm" "greyxray@gmail.com"
    fi
}

if [ "$USER" == "glusheno" ] || [ "$USER" == "ohlushch" ] ; then
    if ! type send | grep -q 'function' ; then source /afs/cern.ch/user/o/ohlushch/.ssh/app-env ; fi
else
    alias send=sen
fi

set -x

if [ -z "$HOME" ] ; then
    send "exit - HOME not set up while expected"
    # exit
fi

if [ -z "$CMSSW_BASE" ] ; then
    send "exit - CMSSW_BASE not set up while expected"
    # exit
fi
pwd
tree
echo $CMSSW_BASE

cd $CMSSW_BASE/src/WonderMass
# echo "$@"
cmsRun jobs_dirs/ntuplizer/test_SUSY/run_test_cfi.py inputFiles=$3 outputFile=$4

mv $4 $5

send "jobs_dirs/ntuplizer/test_SUSY/run_test_cfi.py is done for $5"

set +x
