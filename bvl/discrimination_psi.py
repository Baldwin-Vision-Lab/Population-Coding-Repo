from numpy import argmin, array, empty, exp, finfo, inf, isnan, linspace, log, mean
from numpy import nan, ndarray, ones, sum, where, zeros
from scipy.stats import norm
import json
import random
from collections.abc import Iterable

P_SUM_TO_ONE_CHECK_TOLERANCE = 1e-6
TINY = finfo(float).tiny

def __convert_nd_arrays_to_lists_recursively__(d):
    if isinstance(d, ndarray):
        return d.tolist()
    else:
        if isinstance(d, list):
            d2 = []
            for i,x in enumerate(d):
                d2.append(__convert_nd_arrays_to_lists_recursively__(x))
            return d2
        elif isinstance(d, dict):
            d2 = d.copy()
            for k in d.keys():
                d2[k] = __convert_nd_arrays_to_lists_recursively__(d[k])
            return d2
        else:
            return d

def __does_sum_to_one__(p):
    return abs(sum(array(p).flat) - 1.0)<P_SUM_TO_ONE_CHECK_TOLERANCE

def __get_entropy__(p):
    assert __does_sum_to_one__(p), 'Cannot calculate entropy of matrix that does not sum to 1'
    return -sum((p*log(p)).flat)

def __get_expected_entropy__(H_given_pos_to_x, H_given_neg_to_x, p_pos_given_x, p_neg_given_x):
    return H_given_pos_to_x * p_pos_given_x + H_given_neg_to_x * p_neg_given_x

def __get_param_samples__(minVal, maxVal, stepVal):
    if maxVal == minVal: # max == min, so only 1 value
        return (1, array([maxVal]))
    else:
        numSamples = int((maxVal - minVal) // stepVal) + 1
        sampleArray = linspace(minVal, maxVal, numSamples)
        return (numSamples, sampleArray)

class DiscriminationPsi:

    def __generate_pf_lookup_table__(self):
        pfLUT = empty((self.nStimLevels, self.nAlphaSamples, self.nSigmaSamples, self.nLambdaSamples))
        x     = empty((self.nStimLevels, self.nAlphaSamples, self.nSigmaSamples))
        for i, thisAlpha in enumerate(self.alphaSamples):
            for j, thisSigma in enumerate(self.sigmaSamples):
                x[:,i,j] = (self.stimLevels - thisAlpha) / thisSigma
                
        sigmoidY = norm.cdf(x=x,loc=0,scale=1)
        
        for k, thisLambda in enumerate(self.lambdaSamples):
            sigmoidRangeY = 1 - thisLambda
            y = thisLambda/2.0 + sigmoidY * sigmoidRangeY
            pfLUT[:,:,:,k] = y

        self.pfLUT = pfLUT
        return True
    
    def __get_prior__(self):
        prior = ones((self.nAlphaSamples,self.nSigmaSamples,self.nLambdaSamples))
        if any(isnan(self.priorStDevs)):
            if self.isVerbose:
                print('PRIOR: Using flat prior.')
            self.prior = prior / sum(prior.flat)
            return None
        else:
            raise('Non-flat prior not yet implemented!')

    def __init__(self, paramDict):
        # LOAD MANDATORY PARAMETERS FROM DICTIONARY
        # define minimum and maximum stimulus intensity and step size
        self.minStimLevel  = paramDict['minStimLevel']
        self.maxStimLevel  = paramDict['maxStimLevel']
        self.stimLevelStep = paramDict['stimLevelStep']
        # define minimum and maximum alpha (bias) and step size
        self.minAlpha      = paramDict['minAlpha']
        self.maxAlpha      = paramDict['maxAlpha']
        self.alphaStep     = paramDict['alphaStep']
        # define minimum and maximum log sigma (slope) and step size
        self.minLogSigma   = paramDict['minLogSigma']
        self.maxLogSigma   = paramDict['maxLogSigma']
        self.logSigmaStep  = paramDict['logSigmaStep']
        # define minimum and maximum lambda (lapse rate) and step size
        self.minLambda     = paramDict['minLambda']
        self.maxLambda     = paramDict['maxLambda']
        self.lambdaStep    = paramDict['lambdaStep']
        # define number of trials before terminating psi procedure
        self.numTrials     = paramDict['numTrials']

        # LOAD OPTIONAL PARAMETERS FROM DICTIONARY
        self.isVerbose  = paramDict['isVerbose']  if 'isVerbose'  in paramDict else False
        self.doReflectX = paramDict['doReflectX'] if 'doReflectX' in paramDict else False
        # define mean and st dev for Gaussian prior (or zero st dev for flat)
        self.priorAlphaMean     = paramDict['priorAlphaMean']     if 'priorAlphaMean'     in paramDict else nan
        self.priorAlphaStDev    = paramDict['priorAlphaStDev']    if 'priorAlphaStDev'    in paramDict else nan
        self.priorLogSigmaMean  = paramDict['priorLogSigmaMean']  if 'priorLogSigmaMean'  in paramDict else nan
        self.priorLogSigmaStDev = paramDict['priorLogSigmaStDev'] if 'priorLogSigmaStDev' in paramDict else nan
        self.priorLambdaMean    = paramDict['priorLambdaMean']    if 'priorLambdaMean'    in paramDict else nan
        self.priorLambdaStDev   = paramDict['priorLambdaStDev']   if 'priorLambdaStDev'   in paramDict else nan
        # allow parameters to become marginalised after a number of trials
        self.nTrialsBeforeMarginaliseAlpha  = paramDict['nTrialsBeforeMarginaliseAlpha']  if 'nTrialsBeforeMarginaliseAlpha'  in paramDict else nan
        self.nTrialsBeforeMarginaliseSigma  = paramDict['nTrialsBeforeMarginaliseSigma']  if 'nTrialsBeforeMarginaliseSigma'  in paramDict else nan
        self.nTrialsBeforeMarginaliseLambda = paramDict['nTrialsBeforeMarginaliseLambda'] if 'nTrialsBeforeMarginaliseLambda' in paramDict else nan

        # PARAMETERS COMPUTED FROM DICTIONARY VALUES
        (self.nStimLevels, self.stimLevels)        = __get_param_samples__(self.minStimLevel,
                                                                           self.maxStimLevel,
                                                                           self.stimLevelStep)
        (self.nAlphaSamples, self.alphaSamples)    = __get_param_samples__(self.minAlpha,
                                                                           self.maxAlpha,
                                                                           self.alphaStep)
        (self.nSigmaSamples, self.logSigmaSamples) = __get_param_samples__(self.minLogSigma,
                                                                           self.maxLogSigma,
                                                                           self.logSigmaStep)
        (self.nLambdaSamples, self.lambdaSamples)  = __get_param_samples__(self.minLambda,
                                                                           self.maxLambda,
                                                                           self.lambdaStep)
        self.sigmaSamples = exp(self.logSigmaSamples)

        self.__generate_pf_lookup_table__()

        self.priorMeans  = (self.priorAlphaMean,  self.priorLogSigmaMean,  self.priorLambdaMean)
        self.priorStDevs = (self.priorAlphaStDev, self.priorLogSigmaStDev, self.priorLambdaStDev)

        self.__get_prior__()

        # PREPARE LISTS FOR RECORDING TRIAL-BY-TRIAL DATA
        self.trialStimLevelTested     = []
        self.trialWasResponsePositive = []
        self.posterior_pVL            = []
        self.posteriorAlphaMLE        = []
        self.posteriorAlphaEntropy    = []
        self.posteriorLogSigmaMLE     = []
        self.posteriorLogSigmaEntropy = []
        self.posteriorLambdaMLE       = []
        self.posteriorLambdaEntropy   = []

        # INITIALISE VALUES THAT WILL CHANGE ON EACH TRIAL
        self.paramVectorLikelihood = self.prior

        self.__get_next_stim_level__()

        return None
    
    def __get_expected_entropy__(self,iStimLevel):
        pf_lut_pos_for_x = self.pfLUT[iStimLevel,:,:,:]
        pf_lut_neg_for_x = 1-pf_lut_pos_for_x

        p_pos_to_x = sum((pf_lut_pos_for_x * self.paramVectorLikelihood).flat)
        p_neg_to_x = sum((pf_lut_neg_for_x * self.paramVectorLikelihood).flat)
        assert __does_sum_to_one__((p_pos_to_x,p_neg_to_x)), 'p_pos_resp_to_x + p_neg_resp_to_x neq 1'

        pVL_given_pos_to_x = (pf_lut_pos_for_x * self.paramVectorLikelihood) / p_pos_to_x
        pVL_given_neg_to_x = (pf_lut_neg_for_x * self.paramVectorLikelihood) / p_neg_to_x

        if self.marginalDim == None: # not marginalising over any dimensions
            marginal_pVL_given_pos_to_x = pVL_given_pos_to_x
            marginal_pVL_given_neg_to_x = pVL_given_neg_to_x
        else: # summing over dimensions to be marginalised according to marginalDim
            marginal_pVL_given_pos_to_x = sum(pVL_given_pos_to_x, axis = self.marginalDim)
            marginal_pVL_given_neg_to_x = sum(pVL_given_neg_to_x, axis = self.marginalDim)

        H_given_pos_to_x = __get_entropy__(marginal_pVL_given_pos_to_x)
        H_given_neg_to_x = __get_entropy__(marginal_pVL_given_neg_to_x)

        expH_given_x = __get_expected_entropy__(H_given_pos_to_x, H_given_neg_to_x,
                                                p_pos_to_x, p_neg_to_x)

        return (expH_given_x, pVL_given_pos_to_x, pVL_given_neg_to_x)
    
    def __get_next_stim_level__(self):
        expectedH_given_x = empty(self.nStimLevels)
        paramVectorLikelihood_given_pos_resp_to_x = []
        paramVectorLikelihood_given_neg_resp_to_x = []
        for iStimLevel in range(0,self.nStimLevels):
            (expectedH, pVL_pos, pVL_neg) = self.__get_expected_entropy__(iStimLevel)
            expectedH_given_x[iStimLevel] = expectedH
            paramVectorLikelihood_given_pos_resp_to_x.append(pVL_pos)
            paramVectorLikelihood_given_neg_resp_to_x.append(pVL_neg)

        if self.doReflectX:
            stimLevelsToSearch = abs(self.stimLevels)
        else:
            stimLevelsToSearch = self.stimLevels

        minH = inf
        bestNextStimLevel = 0 # default to 0 in case there is no best stim level
        # enumerating backwards for consistency with argmin which handles
        # multiple equal minimum values by returning the index of the *first*
        for x in set(stimLevelsToSearch):
            H = mean(expectedH_given_x[stimLevelsToSearch == x])
            if H < minH:
                bestNextStimLevel = x
                minH = H

        if self.doReflectX:
            randomSign = random.choice([-1, +1]) # NEED A METHOD FOR RANDOM -1 or +1
            self.nextStimLevel = randomSign * bestNextStimLevel
        else:
            self.nextStimLevel = bestNextStimLevel
            # check that the min entropy from our search loop matches the best pick from taking the min
            assert minH == min(expectedH_given_x), ('min from loop (%0.2f) neq min (%0.2f)') % (minH, min(expectedH_given_x))

        self.pVL_if_pos_response_to_next_trial = paramVectorLikelihood_given_pos_resp_to_x[self.nextStimLevelIndex]
        self.pVL_if_neg_response_to_next_trial = paramVectorLikelihood_given_neg_resp_to_x[self.nextStimLevelIndex]

        return None
    
    def do_response(self, isResponsePositive):

        self.trialStimLevelTested.append(self.nextStimLevel)
        self.trialWasResponsePositive.append(isResponsePositive)

        if self.isVerbose:
            print(('Trial %3.0f, Lev = %5.2f, R = ' % (self.nTrialsCompleted,self.nextStimLevel))+str(isResponsePositive))

        if isResponsePositive:
            self.paramVectorLikelihood = self.pVL_if_pos_response_to_next_trial
        else:
            self.paramVectorLikelihood = self.pVL_if_neg_response_to_next_trial

        self.posterior_pVL.append(self.paramVectorLikelihood)
        (paramMLE, paramEntropy) = self.get_posterior_param_values()

        self.posteriorAlphaMLE.append(paramMLE[0])
        self.posteriorAlphaEntropy.append(paramEntropy[0])
        self.posteriorLogSigmaMLE.append(paramMLE[1])
        self.posteriorLogSigmaEntropy.append(paramEntropy[1])
        self.posteriorLambdaMLE.append(paramMLE[2])
        self.posteriorLambdaEntropy.append(paramEntropy[2])

        self.__get_next_stim_level__()

        return None
    
    def get_posterior_param_values(self):
        paramMLE     = empty(3)
        paramEntropy = empty(3)
        for i in range(0,3):
            x = [self.alphaSamples, self.logSigmaSamples, self.lambdaSamples][i]
            d = [(1,2), (0,2), (0,1)][i]
            p = sum(self.paramVectorLikelihood, axis=d)
            paramMLE[i]     = sum(x*p)
            paramEntropy[i] = __get_entropy__(p)
        return(paramMLE, paramEntropy)
    
    def dict_for_json_export(self):
        d = __convert_nd_arrays_to_lists_recursively__(self.__dict__)
        d['pfLUT'] = None
        return d

    @property
    def nextStimLevelIndex(self):
        assert (self.nextStimLevel in self.stimLevels), 'Next stim level %0.2f not in %s' % (self.nextStimLevel, str(self.stimLevels))
        return where(self.stimLevels==self.nextStimLevel)[0][0]
    
    @property
    def marginalDim(self):
        dims = []
        if not isnan(self.nTrialsBeforeMarginaliseAlpha):
            if self.nTrialsCompleted == self.nTrialsBeforeMarginaliseAlpha:
                if self.isVerbose:
                    print('Exceeded %0.0f trials, marginalising alpha' % self.nTrialsBeforeMarginaliseAlpha)
                dims.append(0)
        if not isnan(self.nTrialsBeforeMarginaliseSigma):
            if self.nTrialsCompleted == self.nTrialsBeforeMarginaliseSigma:
                if self.isVerbose:
                    print('Exceeded %0.0f trials, marginalising sigma' % self.nTrialsBeforeMarginaliseSigma)
                dims.append(1)
        if not isnan(self.nTrialsBeforeMarginaliseLambda):
            if self.nTrialsCompleted == self.nTrialsBeforeMarginaliseLambda:
                if self.isVerbose:
                    print('Exceeded %0.0f trials, marginalising lambda' % self.nTrialsBeforeMarginaliseLambda)
                dims.append(2)
        assert len(dims) < 3, 'Cannot marginalise over ALL THREE parameter dimensions'
        
        if dims == []:
            return None
        else:
            return tuple(dims)
        
    @property
    def nTrialsCompleted(self):
        return len(self.trialStimLevelTested)

    @property
    def mleAlpha(self):
        if self.posteriorAlphaMLE == []:
            return None
        else:
            return self.posteriorAlphaMLE[-1]
    
    @property
    def mleLogSigma(self):
        if self.posteriorLogSigmaMLE == []:
            return None
        else:
            return self.posteriorLogSigmaMLE[-1]
    
    @property
    def mleSigma(self):
        return exp(self.mleLogSigma)
    
    @property
    def mleLambda(self):
        if self.posteriorLambdaMLE == []:
            return None
        else:
            return self.posteriorLambdaMLE[-1]

    @property
    def isExperimentComplete(self):
        return self.nTrialsCompleted >= self.numTrials

    @property
    def pfDataDict(self):
        pfDataDict = {}
        pfDataDict['stimLevel'] = self.stimLevels
        pfDataDict['nTrials']   = zeros(self.nStimLevels)
        pfDataDict['nPositive'] = zeros(self.nStimLevels)
        for iTrial in range(0,self.nTrialsCompleted):
            lev  = self.trialStimLevelTested[iTrial]
            ind  = where(self.stimLevels==lev)[0][0]
            pfDataDict['nTrials'][ind]  = pfDataDict['nTrials'][ind] + 1
            if self.trialWasResponsePositive[iTrial]:
                pfDataDict['nPositive'][ind] = pfDataDict['nPositive'][ind] + 1
        return pfDataDict