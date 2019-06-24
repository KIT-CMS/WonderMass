# [WonderMass]

## Runnong ntuplizer

### Official MC

1. To run on a single file locally:

```
    scramb
    cmsRun ntupleBuilder/python/run_test_cfi.py \
        inputFiles=/store/mc/RunIIFall17MiniAODv2/SUSYGluGluToHToTauTau_M-800_TuneCP5_13TeV-pythia8/MINIAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/00000/E2AE94C9-7042-E811-BD27-0CC47A0AD792.root \
        outputFile=out.root && root -l out.root
```

2. To run in batch on multiple files:

Init proxy and do `setcrab3`. Use `ntupleBuilder/python/create_jobs_ntuplizer.py`, example command is in the header. 

3. To run 

![WonderMassPic](https://s3.gifyu.com/images/ezgif-1-c79300248a93.gif)

