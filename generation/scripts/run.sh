#!/bin/bash
set -x

# to run: ./run.sh

#####################################
# Configs to touch
ID=$1
TYPE=$2
MASS=$3
NUM_EVENTS=$4
PREFIX="_condor_wm"
#
# this has gen, sim, and reco levels
# https://github.com/cms-sw/cmssw/blob/master/Configuration/PyReleaseValidation/python/relval_steps.py
# https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/test/runtests.sh
ERA="Run2_2017_FastSim" #  try : ERA="Run2_2017"  # try Run2_2017_FastSim, Run2_2017, see: https://github.com/cms-sw/cmssw/blob/master/Configuration/StandardSequences/python/Eras.py
CONDITIONS=auto:phase1_2017_realistic
#
# https://cms-pdmv.cern.ch/mcm/campaigns?prepid=RunIIFall17MiniAOD&page=0&shown=524287
# 94X_mc2017_realistic_v5 , see twiki: https://twiki.cern.ch/twiki/bin/view/CMS/PdmVMCcampaignRunIIFall17MiniAOD
# 94X_mc2017_realistic_v11, see pdmv: https://cms-pdmv.cern.ch/mcm/campaigns?prepid=RunIIFall17MiniAOD&page=0&shown=524287
#! 94X_mc2017_realistic_v14 , see das: https://cmsweb.cern.ch/das/request?view=list&limit=50&instance=prod%2Fglobal&input=%2FGluGluHToTauTau*powheg*pythia8%2FRunIIFall17MiniAODv2*%2FMINIAODSIM
# 94X_mc2017_realistic_v17, see PPD: https://docs.google.com/presentation/d/1YTANRT_ZeL5VubnFq7lNGHKsiD7D3sDiOPNgXUYVI0I/edit#slide=id.g54ddca43ca_6_0
BEAMSPOT="Realistic25ns13TeVEarly2017Collision"  # seems to be used by everyone else: https://www.sprace.org.br/twiki/bin/view/Main/EduardoCmsMonteCarlo
# ? which tune to use?
# ? test --step GEN,SIM,RECOBEFMIX,DIGI,L1,DIGI2RAW,L1Reco,RECO \
if [  "$CMSSW_VERSION" == "CMSSW_9_4_7" ] ; then
    ERA="Run2_25ns"
    CONDITIONS=auto:run2_mc
fi
#####################################

#####################################
# Setup EOS space as output directory
if [ "$USER" == "glusheno" ] || [ "$USER" == "ohlushch" ] ; then
    source /afs/cern.ch/user/o/ohlushch/.ssh/app-env
    EOS_HOME=/eos/user/o/ohlushch
    OUTPUTDIR=/eos/user/o/ohlushch/Nostradamas/mass_regression
elif [ "$(whoami)" == "swunsch" ] || [ "$(whoami)" == "swunsch" ] ; then
    EOS_HOME=/eos/user/s/swunsch
    OUTPUTDIR=${EOS_HOME}/mass_regression
else
    send "exit for "${OUTPUTDIR}; exit
fi
ls -la $EOS_HOME
OUTPUTDIR=$OUTPUTDIR/${CMSSW_VERSION}/${TYPE}_${ERA}_${CONDITIONS}_${BEAMSPOT}${PREFIX}/${MASS}GeV/${ID}
if [ ! -d "$OUTPUTDIR" ]; then
  mkdir -p $OUTPUTDIR
fi


copylogs() {
    cp *.log ${OUTPUTDIR}/
    cp *.py ${OUTPUTDIR}/
    cp *.txt ${OUTPUTDIR}/
    cp Configuration/Generator/python/"${TYPE}"generatorSnipplet_cfi.py  ${OUTPUTDIR}/"${TYPE}"_generatorSnipplet_cfi.py
    cp workspace/ntupleBuilder/src/ntupleBuilder.cc  ${OUTPUTDIR}/ntupleBuilder.cc
}

echo $'\n'"### Create MinBias with pile-up"
if ! cmsDriver.py MinBias_13TeV_pythia8_TuneCUETP8M1_cfi \
    --conditions ${CONDITIONS} \
    --fast \
    -n ${NUM_EVENTS} \
    --era ${ERA} \
    --eventcontent FASTPU \
    -s GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,RECO \
    --datatier GEN-SIM-RECO \
    --beamspot ${BEAMSPOT} \
    --no_exec &> ${ERA}_${CONDITIONS}_${NUM_EVENTS}_minbias_cmsDrive.log
then
    cat ${ERA}_${CONDITIONS}_${NUM_EVENTS}_minbias_cmsDrive.log
    copylogs
    ltr
    send "exit for "${OUTPUTDIR}
    exit
fi
if ! sed -i "s/Services_cff')/Services_cff'); process.RandomNumberGeneratorService.generator.initialSeed = "${ID}"/g" MinBias_13TeV_pythia8_TuneCUETP8M1_cfi_GEN_SIM_RECOBEFMIX_DIGI_RECO.py ; then send "exit for "${OUTPUTDIR}; exit ; fi # Set random seed with job id
if ! cmsRun MinBias_13TeV_pythia8_TuneCUETP8M1_cfi_GEN_SIM_RECOBEFMIX_DIGI_RECO.py &> minbias_cmsRun.log
then
    cat minbias_cmsRun.log
    copylogs
    ltr
    send "exit for "${OUTPUTDIR}
    exit
fi



echo $'\n'"### Create AODSIM with target process and PU mixing"
# echo "#### Copy generator snipplet and set properties"
if ! cp workspace/ntupleBuilder/python/generatorSnipplet_${TYPE}_cfi.py Configuration/Generator/python/"${TYPE}"_generatorSnipplet_cfi.py ; then send "exit for "${OUTPUTDIR}; exit ; fi
if ! sed -i "s,MASS_MIN,"$(expr $MASS - 1)",g" Configuration/Generator/python/"${TYPE}"_generatorSnipplet_cfi.py ; then send "exit for "${OUTPUTDIR}; exit ; fi
if ! sed -i "s,MASS_MAX,"$(expr $MASS + 1)",g" Configuration/Generator/python/"${TYPE}"_generatorSnipplet_cfi.py ; then send "exit for "${OUTPUTDIR}; exit ; fi
if ! sed -i "s,MASS,"$MASS",g" Configuration/Generator/python/"${TYPE}"_generatorSnipplet_cfi.py ; then send "exit for "${OUTPUTDIR}; exit ; fi

if ! cmsDriver.py "${TYPE}"_generatorSnipplet_cfi \
        --conditions ${CONDITIONS} \
        --fast \
        -n ${NUM_EVENTS} \
        --era ${ERA} \
        --eventcontent AODSIM \
        -s GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,RECO \
        --datatier AODSIM \
        --beamspot ${BEAMSPOT} \
        --pileup_input file:MinBias_13TeV_pythia8_TuneCUETP8M1_cfi_GEN_SIM_RECOBEFMIX_DIGI_RECO.root \
        --pileup AVE_35_BX_25ns \
        --no_exec &> "${TYPE}"_reco_cmsDrive.log
then
    cat "${TYPE}"_reco_cmsDrive.log
    copylogs
    ltr
    send "exit for "${OUTPUTDIR}
    exit
fi
if ! sed -i "s/Services_cff')/Services_cff'); process.RandomNumberGeneratorService.generator.initialSeed = "${ID}"/g" "${TYPE}"_generatorSnipplet_cfi_GEN_SIM_RECOBEFMIX_DIGI_RECO_PU.py ; then send "exit for "${OUTPUTDIR}; exit ; fi
if ! cmsRun "${TYPE}"_generatorSnipplet_cfi_GEN_SIM_RECOBEFMIX_DIGI_RECO_PU.py  &> "${TYPE}"_reco_cmsRun.log
then
    cat "${TYPE}"_reco_cmsRun.log
    copylogs
    ltr
    send "exit for "${OUTPUTDIR}
    exit
fi


echo $'\n'"### Create MiniAODSIM"
if ! cmsDriver.py "${TYPE}"_miniAOD-prod \
        -s PAT \
        --eventcontent MINIAODSIM \
        --runUnscheduled \
        --mc \
        --fast \
        -n ${NUM_EVENTS} \
        --filein file://"${TYPE}"_generatorSnipplet_cfi_GEN_SIM_RECOBEFMIX_DIGI_RECO_PU.root \
        --conditions ${CONDITIONS} \
        --era ${ERA} \
        --customise_commands 'del process.patTrigger; del process.selectedPatTrigger' \
        --no_exec  &> "${TYPE}"_miniaod_cmsDrive.log
then
    cat "${TYPE}"_miniaod_cmsDrive.log
    cp "${TYPE}"_miniaod_cmsDrive.log  ${OUTPUTDIR}/"${TYPE}"_miniaod_cmsDrive.log
    ltr
    send "exit for "${OUTPUTDIR}
    exit
fi
if ! cmsRun "${TYPE}"_miniAOD-prod_PAT.py &> "${TYPE}"_miniaod_cmsRun.log
then
    cat "${TYPE}"_miniaod_cmsRun.log
    copylogs
    ltr
    send "exit for "${OUTPUTDIR}
    exit
fi



echo $'\n'"### Run ntupleBuilder analyzer on MiniAOD"
if ! sed -i -e "s,^files =,files = ['file:""${TYPE}""_miniAOD-prod_PAT.root'] #,g" workspace/ntupleBuilder/python/run_cfi.py  ; then send "exit for "${OUTPUTDIR}; exit ; fi
if ! sed -i -e "s,^settype =,settype = \""$TYPE"\" #,g" workspace/ntupleBuilder/python/run_cfi.py ; then send "exit for "${OUTPUTDIR}; exit ; fi

if ! cmsRun workspace/ntupleBuilder/python/run_cfi.py ; then send "exit for "${OUTPUTDIR}; exit ; fi


echo $'\n'"### Copy files to output folder"
# auto mount of EOS
copylogs
xrdcp -f miniAOD-prod_PAT.root root://eosuser.cern.ch/${OUTPUTDIR}/MiniAOD_id${ID}_mass${MASS}_events${NUM_EVENTS}.root
xrdcp -f output.root root://eosuser.cern.ch/${OUTPUTDIR}/ntuple_id${ID}_type${TYPE}_mass${MASS}_events${NUM_EVENTS}.root

ltr ${OUTPUTDIR}
send "done in "${OUTPUTDIR}
