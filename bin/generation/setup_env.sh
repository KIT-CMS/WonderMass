#!/bin/bash
set -x


if [ "$USER" == "glusheno" ] || [ "$USER" == "ohlushch" ] ; then
    if ! type send | grep -q 'function' ; then source /afs/cern.ch/user/o/ohlushch/.ssh/app-env ; fi
else
    alias send="tput bel"
fi


# Set up CMSSW
SCRAM_ARCH=slc6_amd64_gcc630
export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
set +x
source $VO_CMS_SW_DIR/cmsset_default.sh
source /cvmfs/cms.cern.ch/cmsset_default.shlt
set -x
scram project CMSSW CMSSW_9_4_13_UL1
cd CMSSW_9_4_13_UL1/src
if ! eval `scramv1 runtime -sh` ; then send "exit for "${OUTPUTDIR}; exit ; fi

# needed for batch
git config --global user.name 'Foo'
git config --global user.email 'foo@bar.ch'
git config --global user.github 'foo'

# Generator
if ! git cms-addpkg Configuration/Generator ; then send "exit for "${OUTPUTDIR}; exit ; fi

# ntupliser and scripts
git clone git@github.com:KIT-CMS/WonderMass.git
# ntupliser
# if ! curl -O https://transfer.sh/qn9Jr/workspace.tar.gz ; then send "exit for "${OUTPUTDIR}; exit ; fi
# if ! tar -zxvf workspace.tar.gz ; then send "exit for "${OUTPUTDIR}; exit ; fi
# rm workspace.tar.gz
export CORES=`grep -c ^processor /proc/cpuinfo`

if [ -z "$CORES" ]
then
      export CORES=`grep -c ^processor /proc/cpuinfo`
fi

if ! scram b -j $CORES ; then send "exit (compilation error) for "${OUTPUTDIR}; exit ; fi

echo "Environment successfully set"
