import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *

# DYtest
# qq -> Z -> tautau
# https://github.com/cms-sw/cmssw/blob/master/Configuration/Generator/python/DYToLL_M-50_13TeV_pythia8_cff.py
# http://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
generator = cms.EDFilter("Pythia8GeneratorFilter",
            comEnergy=cms.double(13000.0),
            crossSection=cms.untracked.double(6.44),
            filterEfficiency=cms.untracked.double(1),
            maxEventsToPrint=cms.untracked.int32(1),
            pythiaHepMCVerbosity=cms.untracked.bool(False),
            pythiaPylistVerbosity=cms.untracked.int32(1),
            PythiaParameters=cms.PSet(
                pythia8CommonSettingsBlock,
                pythia8CUEP8M1SettingsBlock,
                processParameters=cms.vstring(
                    # 'WeakZ0:gmZmode=2',
                    # 'WeakSingleBoson:all=off',
                    'WeakSingleBoson:ffbar2gmZ=on',
                    '23:onMode=off',
                    '23:onIfAny=15',  # tau
                    # '23:m0=MASS',
                    '23:mMin=50.',
                    # '23:mMax=MASS_MAX',
                    # 'PhaseSpace:mHatMin=MASS_MIN',
                    # 'PhaseSpace:mHatMax=MASS_MAX',
                    # '15:onMode=off',
                    # '15:onIfAny=111 211 -211', # pi0 pi+
                ),
                parameterSets=cms.vstring(
                    'pythia8CommonSettings',
                    'pythia8CUEP8M1Settings',
                    'processParameters',
                )
            )
)

ProductionFilterSequence = cms.Sequence(generator)
