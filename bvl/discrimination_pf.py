from numpy import array, empty, exp, finfo, log
from numpy import nan, nanmean, nanstd, nansum, percentile, sum, unique
from scipy.optimize import minimize
from scipy.stats import norm
from numpy.random import randint
import psignifit
from psignifit import sigmoids

TUKEY_FACTOR = 3
PPF_VALS_FOR_THRESHOLD = (0.25, 0.75)
SIGMOID = 'norm'
PF_TYPE = 'equal asymptote'

def getStandardParameters(res):
    # NOT SURE IF THIS IS CORRECT
    theta = res['parameter_estimate']
    w     = res['configuration']['width_alpha']
    if res['configuration']['sigmoid'] == 'norm':
        c     = norm.ppf(1-w) - norm.ppf(w)
        theta['sigma'] = theta['width']/c
    else:
        raise(NotImplementedError, 'Only supporting cumulative normal pfs for now')
    return theta

def __get_threshold__(res):

    sigmoid = res.configuration.make_sigmoid()

    widthParams = [res.parameter_estimate['width'],
                   res.confidence_intervals['width'][0][0], # lower 95% CI
                   res.confidence_intervals['width'][0][1]] # upper 95% CI
    
    thresholdFromSlope = []

    for w in widthParams:
        x = []
        for p in PPF_VALS_FOR_THRESHOLD:
            x.append(sigmoid.inverse(p, 0, w, 0, 0))
        thresholdFromSlope.append((x[1] - x[0])/2.0)

    return thresholdFromSlope[0], (thresholdFromSlope[1], thresholdFromSlope[2])

def fit_identification_pf(lev, nT, nP, modelX = None, numBootstraps = None):
    dataMatrix = empty((len(lev),3))
    dataMatrix[:,0] = lev
    dataMatrix[:,1] = nP
    dataMatrix[:,2] = nT
    res = psignifit.psignifit(dataMatrix, sigmoid=SIGMOID,stimulus_range=(min(lev),max(lev)),
                              experiment_type=PF_TYPE)
    
    FIT = res.as_dict()
    FIT['thresholdFromSlope'] = __get_threshold__(res)

    if modelX is not None:
        sigmoid = sigmoids.sigmoid_by_name(SIGMOID)
        modelY = sigmoid._value(modelX, res.parameter_estimate['threshold'], res.parameter_estimate['width'])
        FIT['modelX'] = modelX
        FIT['modelY'] = modelY
    else:
        FIT['modelX'] = None
        FIT['modelY'] = None

    if (0 if numBootstraps is None else numBootstraps) > 0:
        bsParamValDict = __bootstrap_pf_non_parametric__(lev, nT, nP, numBootstraps)
        FIT['bsParamValDict'] = bsParamValDict
    else:
        FIT['bsParamValDict'] = None

    return FIT

def __bootstrap_pf_non_parametric__(lev, nT, nP, numBootstraps):

    bsParamValDict = {'bsEta'                : [],
                      'bsLambda'             : [],
                      'bsBias'               : [],
                      'bsThresholdFromSlope' : [],
                      'bsWidth'              : []}
    (flatLev,flatIsP) = __flatten_data__(lev,nT,nP)

    for iBootstrap in range(0, numBootstraps):
        (s_Lev,s_nT,s_nP) = __resample_data__(flatLev,flatIsP)
        FIT = fit_identification_pf(s_Lev, s_nT, s_nP)
        bsParamValDict['bsEta'].append(FIT['parameter_estimate']['eta'])
        bsParamValDict['bsLambda'].append(FIT['parameter_estimate']['lambda'])
        bsParamValDict['bsBias'].append(FIT['parameter_estimate']['threshold'])
        bsParamValDict['bsThresholdFromSlope'].append(FIT['thresholdFromSlope'])
        bsParamValDict['bsWidth'].append(FIT['parameter_estimate']['width'])
    
    return bsParamValDict

def __flatten_data__(lev,nT,nP):
    flatLev = []
    flatIsP = []
    for iLev, thisLevel in enumerate(lev):
        for iTrial in range(0,int(nT[iLev])):
            flatLev.append(thisLevel)
            if iTrial <= nP[iLev]:
                flatIsP.append(1)
            else:
                flatIsP.append(0)
    return (flatLev,flatIsP)

def __remove_outlier__(y):
    q1 = percentile(y,25)
    q3 = percentile(y,75)
    iqr = q3 - q1
    lowerBound = q1 - TUKEY_FACTOR * iqr
    upperBound = q3 + TUKEY_FACTOR * iqr
    cleanY = y.copy()
    cleanY[y < lowerBound] = nan
    cleanY[y > upperBound] = nan
    return cleanY

def __resample_data__(flatLev,flatIsP):
    
    sample_flat_logLev = []
    sample_flat_isP    = []
    
    numTrialsToSample = len(flatLev)
    
    for i in range(0,numTrialsToSample):
        thisSampleIndex = randint(numTrialsToSample)
        sample_flat_logLev.append(flatLev[thisSampleIndex])
        sample_flat_isP.append(flatIsP[thisSampleIndex])

    sample_logLev = unique(sample_flat_logLev)
    sample_logLev.sort()
    
    logLev_array = array(sample_flat_logLev)
    isP_array = array(sample_flat_isP)
    sample_nP = []
    sample_nT = []
    
    for iLev in range(0,len(sample_logLev)):
        thisData = isP_array[logLev_array==sample_logLev[iLev]]
        sample_nT.append(len(thisData))
        sample_nP.append(sum(thisData))
    
    return (sample_logLev,array(sample_nT),array(sample_nP))

