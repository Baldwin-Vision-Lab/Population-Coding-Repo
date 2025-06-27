import os
import sys
from numpy import random, log 
from bvl.expfunc import compute_additional_parameters, load_parameters, make_coherence_stimulus, make_gaussian_stimulus
from psychopy import sound, core, gui, monitors, event, visual, data

experimentFileName = os.path.basename(__file__)

experimentName     = os.path.splitext(experimentFileName)[0]

expInfo = {
    'participantID'          : 'CC',
    'stdOriDeg'              : [20],
    'checkSizeWavelets'      : [0, 1, 3, 9],
    'experimenter'           : ['Carson','Alex'],
    'DebugMode'              : False,
    'IsHardwareConnected'    : False,
    'IsFillBlankChecks'      : False,
    'IsSoundMode'            : True,
    'IsCheckPhaseReversed'   : False,
    'IsCheckPhaseInterweaved': False,
    'gridType'               : ['square', 'hexagonal'],
    'condition'              : ['Mean', 'Coherence'],
    'saveType'               : ['json','csv'],
    }

# === GUI === 
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=experimentName)
if dlg.OK == False:
    core.quit() 

stimDict = {
    'checkSizeWavelets'    : int(expInfo['checkSizeWavelets']),
    'isFillBlankChecks'    : expInfo['IsFillBlankChecks'],
    'oriDeg'               : 45,
    'proportionCoherence'  : 0.5,
    'rmsContrast'          : 0.8,
    'rngSeed'              : random.randint(0,2**31-1),
    'gridType'             : 'square',
    'stdOriDeg'            : int(expInfo['stdOriDeg']),
    }
_thisDir = os.path.dirname(os.path.abspath(__file__))
stimSettingsFilePath = os.path.join(_thisDir, 'settings', 'texture_orientation_stim_parameters.csv')

expParameters    = load_parameters(stimSettingsFilePath)
expParameters    = compute_additional_parameters(expParameters, expInfo)
exampleStim = make_coherence_stimulus(
                        expParameters, 
                        random.default_rng(seed=stimDict['rngSeed']),
                        stimDict['checkSizeWavelets'], 
                        expInfo['IsCheckPhaseReversed'],
                        stimDict['isFillBlankChecks'],
                        stimDict['oriDeg'],
                        stimDict['proportionCoherence'],
                        stimDict['rmsContrast'],
                        expInfo['gridType'],
                        isModelMode=True
                        )

print(exampleStim)
