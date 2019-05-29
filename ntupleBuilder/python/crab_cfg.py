from WMCore.Configuration import Configuration
config = Configuration()

config.section_("General")
config.General.requestName = 'crabdir_DYJetsToLL_M__50_TuneCP5_13TeV__madgraphMLM__pythia8'
config.General.workArea = 'Official17'
config.General.transferLogs = True

config.section_("User")
config.User.voGroup = 'dcms'

config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'workspace/ntupleBuilder/python/run_cfi.py'

config.section_("Data")
config.Data.inputDataset = '/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIFall17MiniAODv2-PU2017RECOSIMstep_12Apr2018_94X_mc2017_realistic_v14_ext1-v1/MINIAODSIM'
config.Data.inputDBS = 'global'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
config.Data.totalUnits = -1
config.Data.outLFNDirBase = '/store/user/ohlushch/MassRegression/Official17/'
config.Data.publication = False
config.Data.allowNonValidInputDataset = True

config.section_("Site")
config.Site.storageSite = 'T2_DE_DESY'
