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
source $CMSSW_BASE/src/WonderMass/generation/scripts/./run_local.sh "$@"

set +x
