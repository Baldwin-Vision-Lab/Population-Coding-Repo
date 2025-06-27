import numpy as np
import math
import csv
import psychopy
#import pypixxlib
from math  import floor, pi, tan
from numpy import add, arctan2, cos, divide, exp, finfo, fft, linspace, log, log2
from numpy import mean, multiply, random, real, rot90, shape, sin, size, stack, zeros
from numpy import zeros, ones, sin, linspace, meshgrid, empty, std
from numpy import abs as numpyabs
from numpy import max as numpymax
from numpy import random, nan, isnan
from scipy import signal
from psychopy import visual, monitors
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
#from pypixxlib.viewpixx import VIEWPixx
#from pypixxlib.shaders import setUpShaderAndWindow


def get_dot_matrix_diamond(imSizePix, stimulusProportionFilled, dotSpacingPix):
    # Make initial square larger by sqrt(2)
    largerSize = int(imSizePix * 2)
    numDotsCanFitWidth = (largerSize*stimulusProportionFilled) // dotSpacingPix + 1
    oddNumDotsCanFitWidth = 2*(numDotsCanFitWidth // 2) - 1
    dotRegionPix = (oddNumDotsCanFitWidth) * dotSpacingPix
    borderGapPix = np.ceil((largerSize-dotRegionPix)/2.0)
    dotRegionPix = largerSize - 2*borderGapPix
    dotPos = np.arange(borderGapPix, borderGapPix+dotRegionPix+dotSpacingPix, dotSpacingPix)
    dotPosX = np.empty([len(dotPos)**2])
    dotPosY = np.empty([len(dotPos)**2])

    ind = 0

    for i, x in enumerate(dotPos):
        for j, y in enumerate(dotPos):
            dotPosX[ind] = x
            dotPosY[ind] = y
            ind = ind + 1
    
    # # Plot 1: Initial dot positions
    # plt.figure(figsize=(15, 5))
    # plt.subplot(131)
    # plt.scatter(dotPosX, dotPosY)
    # plt.title('Initial Dot Positions')
    # plt.axis('equal')
    
    # Rotate dot positions 45 degrees clockwise
    angle = -np.pi/4
    center_x = center_y = largerSize/2
    rotated_x = np.zeros_like(dotPosX)
    rotated_y = np.zeros_like(dotPosY)
    
    for i in range(len(dotPosX)):
        x = dotPosX[i] - center_x
        y = dotPosY[i] - center_y
        rotated_x[i] = x * np.cos(angle) - y * np.sin(angle) + center_x
        rotated_y[i] = x * np.sin(angle) + y * np.cos(angle) + center_y

    # # Plot 2: Rotated dot positions
    # plt.subplot(132)
    # plt.scatter(rotated_x, rotated_y)
    # plt.title('Rotated Dot Positions')
    # plt.axis('equal')

    # Create larger temporary array
    temp_imStim = np.zeros([largerSize, largerSize])
    temp_coordinates = np.zeros([largerSize, largerSize, 2])

    # Fill the larger array
    for i in range(len(rotated_x)):
        x = int(rotated_x[i])
        y = int(rotated_y[i])
        if 0 <= x < largerSize and 0 <= y < largerSize:
            temp_imStim[x, y] = 1
            temp_coordinates[x, y, 0] = x
            temp_coordinates[x, y, 1] = y

    # Cut out the center square
    start = (largerSize - imSizePix) // 2
    end = start + imSizePix
    imStim = temp_imStim[start:end, start:end]
    imDotCoordinatesYX = temp_coordinates[start:end, start:end]

    # # Plot 3: Final cropped pattern
    # plt.subplot(133)
    # plt.imshow(imStim)
    # plt.title('Final Cropped Pattern')
    # plt.axis('equal')
    
    # plt.tight_layout()
    # plt.show()

    return imStim, imDotCoordinatesYX


def load_parameters(settingsFilePath):
    expParameters = {}
    with open(settingsFilePath, mode = 'r') as f:
        experimentSettings = csv.reader(f)
        expParameters = {r[0]:float(r[1]) for r in experimentSettings}
    return expParameters

def raised_cosine_temporal_envelope(plateauWidthFrames, rampWidthFrames):
    
    totalDurationFrames = plateauWidthFrames + 2*rampWidthFrames + 2
    y1 = ones(totalDurationFrames)
    
    rCosX  = linspace(0,pi,rampWidthFrames)
    rampOn = 1.0 - (1.0+cos(rCosX))/2.0
    
    y1[1:(1+rampWidthFrames)] = rampOn
    y1[0] = 0
    
    y2 = y1[::-1]
    y2[1:(1+rampWidthFrames)] = rampOn
    y2[0] = 0
    
    return y2

def compute_additional_parameters(expParameters, expInfo=None):
    # spatial stimulus properties
    pixPerCM = expParameters['screenWidthPix'] / expParameters['screenWidthCM']
    cmPerDeg = tan(pi/180) * expParameters['viewDistanceCM']
    expParameters['pixPerDeg'] = pixPerCM * cmPerDeg
    expParameters['imSizeDeg'] = expParameters['imSizePix'] /expParameters['pixPerDeg'] # image size in degrees
    expParameters['pixPerCy'] = expParameters['pixPerDeg']/expParameters['waveletSpatFreqCyPerDeg']
    expParameters['waveletSizePix'] = expParameters['waveletImSizeCycles'] * expParameters['pixPerCy']
    expParameters['dotSpacingPix'] = expParameters['waveletSpacingCycles'] * expParameters['pixPerCy']
    expParameters['stimSizeDeg'] = (expParameters['dotSpacingPix'] * expParameters['stimSizeWavelets'])/expParameters['pixPerDeg']
    expParameters['stimulusProportionFilled'] = expParameters['stimSizeDeg']/expParameters['imSizeDeg']

    # temporal stimulus properties
    frameRate = expParameters['screenFrameRate']
    frameDur  = 1.0 / round(frameRate)
    expParameters['temporalPlateauWidthFrames']     = seconds_to_frames(expParameters['temporalPlateauWidthSeconds'], frameDur)
    expParameters['temporalRCosWidthFrames']        = seconds_to_frames(expParameters['temporalRCosWidthSeconds'], frameDur)
    rCosEnvelope = raised_cosine_temporal_envelope(expParameters['temporalPlateauWidthFrames'], expParameters['temporalRCosWidthFrames'])
    expParameters['temporalEnvelope'] = rCosEnvelope
    expParameters['temporalPresentationTimeFrames'] = len(rCosEnvelope)

    return expParameters

def check_device_parameters(expParameters, monitor, win):
    # check framerate (allowing some tolerance)
    frameRate = win.getActualFrameRate()
    print('Reported device frame rate = '+frameRate)
    if frameRate == None:
        raise('Could not get framerate')
    else:
        frameRateRatio = frameRate / expParameters['screenFrameRate']
        frameRateErrorPercent = numpyabs(frameRateRatio-1)*100.0
        potentialErrorMsg = "Actual frame rate %0.2f Hz, but settings csv specifies %0.2f Hz" % (frameRate, expParameters['screenFrameRate'])
        assert frameRateErrorPercent < expParameters['screenFrameRateTolerancePercent'], potentialErrorMsg

    # check dimensions
    assert monitor.getWidth()      == expParameters['screenWidthCM'],   "Monitor has width %0.1f cm in PsychoPy setup, but settings csv specifies %0.1f cm"  % (monitor.getWidth(),expParameters['screenWidthCM'])
    assert monitor.getSizePix()[0] == expParameters['screenWidthPix'],  "Monitor has width %0.0f px in PsychoPy setup, but settings csv specifies %0.0f px"  % (monitor.getSizePix()[0],expParameters['screenWidthPix'])
    assert monitor.getSizePix()[1] == expParameters['screenHeightPix'], "Monitor has height %0.0f px in PsychoPy setup, but settings csv specifies %0.0f px" % (monitor.getSizePix()[1],expParameters['screenHeightPix'])
    assert monitor.getDistance()   == expParameters['viewDistanceCM'], "Monitor setting has viewing distance of %0.0f cm in PsychoPy setup, but settings csv specifies %0.0f cm" % (monitor.getDistance(),expParameters['viewDistanceCM'])
        
    return True

def seconds_to_frames(timeInSeconds, frameDur):
    timeInFrames = timeInSeconds//frameDur
    return int(timeInFrames)

def render_from_wavelets(expParameters,xy,theta):
    imSizePix = int(expParameters['imSizePix'])
    imC = zeros((imSizePix,imSizePix))
    WV  = expParameters
    for i in range(0,len(theta)):
        WV['oriDeg'] = theta[i]
        imLG = get_single_log_gabor(WV)
        imLG = imLG / np.std(imLG.flat) # normalise for unit RMS
        (adjCLGShift, xOnInt, yOnInt, xOffInt, yOffInt) = get_position_shift(xy[:,i],WV['waveletSizePix'],0,imLG)
        # Check if indices are within bounds and if shapes match
        if (0 <= xOnInt < imSizePix and 0 <= yOnInt < imSizePix and 
            xOffInt < imSizePix and yOffInt < imSizePix):
            chunk = imC[yOnInt:yOffInt+1,xOnInt:xOffInt+1]
            if chunk.shape == adjCLGShift.shape:
                changedChunk = add(chunk, adjCLGShift)
                imC[yOnInt:yOffInt+1,xOnInt:xOffInt+1] = changedChunk
    return imC

def get_pix_vals(x,y):
    imSize = 40
    mid = int(imSize/2)

    im = zeros((imSize,imSize))
    #python index starts at 1
    xOn = int(round(x*mid))
    yOn = int(round(y*mid))
    xOff = int(xOn + (imSize/2))
    yOff = int(yOn + (imSize/2))
    for i in range(xOn,xOff):
        for j in range(yOn,yOff):
            im[i][j] = 1

    k = zeros((3,3))
    
    k[1][1] = mean(im[0:mid,0:mid])
    k[1][2] = mean(im[0:mid,mid:])
    k[2][1] = mean(im[mid:,0:mid])
    k[2][2] = mean(im[mid:,mid:])
    
    return k

def get_position_shift(xy,gabSize,posSD,adjCLG):
    #print(posSD)
    #print(type(posSD))
    xOn  = xy[0] - gabSize/2 + random.randn()*posSD
    yOn  = xy[1] - gabSize/2 + random.randn()*posSD
    adjCLGShift, xOnInt, yOnInt = subpixel_shift(adjCLG, xOn, yOn)
    xOffInt = xOnInt + adjCLGShift.shape[0] - 1
    yOffInt = yOnInt + adjCLGShift.shape[1] - 1
    return (adjCLGShift, xOnInt, yOnInt, xOffInt, yOffInt)

def raised_cosine(theta, pixModAmp, modPhase):
  
    v = zeros(shape(theta))
    
    if modPhase==0:
        for i in range(0,size(theta)):
            v[i]=(1+cos(theta[i]))/2*pixModAmp
            
    elif modPhase==1:
        for i in range(0,size(theta)):
            v[i]=-(1+cos(theta[i]))/2*pixModAmp
    else: 
        raise Exception("Invalid modPhase value")
    
    return v

def subpixel_shift(im, x, y):
    xInt = floor(x)
    yInt = floor(y)
    xRem = x - xInt
    yRem = y - yInt
    k = get_pix_vals(xRem,yRem)
    imShift = signal.convolve2d(im, k,mode='same')
    return (imShift, xInt, yInt)

def log_gabor(imSize, spatFreqCyclesPerImage, spatFreqBandwidthOctaves,
              orientationDeg, orientationBandwidth, phaseDeg):
    
    tiny      = np.finfo(float).tiny
    theta0    = -(math.radians(orientationDeg)) #idk why this has to be negative, but it does
    phi       = math.radians(phaseDeg)
    thetaBW   = math.radians(orientationBandwidth)
    x         = np.arange(1,imSize+1,1) - (imSize+2)/2.0
    (u,v)     = np.meshgrid(x,x)
    f         = (u**2.0 + v**2.0)**0.5
    theta     = np.arctan2(v,u)
    thetaDiff = theta-theta0
    uft       = f*np.cos(thetaDiff)
    bw        = 0.424*spatFreqBandwidthOctaves
    a         = np.abs(uft) / spatFreqCyclesPerImage
    a[a==0]   = tiny # prevent a div 0 error from log2 containing zeros
    n         = -(np.log2(a)**2.0)
    d         = 2*(bw)**2.0
    logGab1D  = np.exp(n/d)
    
    n         = np.log2(np.abs(np.cos(thetaBW)))
    k         = (n / bw)**2.0
    
    b         = spatFreqCyclesPerImage*np.sin(thetaBW)
    c         = 1/(np.log(4) - k)
    eta       = b*(c+0j)**0.5
    etaSq     = eta**2.0
    
    n         = -(f*np.sin(thetaDiff))**2.0
    d         = 2.0 * etaSq
    orthFunc  = np.exp(n/d)
    
    logGab2D  = logGab1D * orthFunc

    cx1 = np.zeros([imSize,imSize]) + 0j
    cx2 = np.zeros([imSize,imSize]) + 0j
    
    a   = +logGab2D[uft>0]*np.sin(phi)
    b   = -logGab2D[uft>0]*np.cos(phi)
    cx1[uft>0] = a + 1j * b
    a   = +logGab2D[uft<0]*np.sin(phi)
    b   = +logGab2D[uft<0]*np.cos(phi)
    cx2[uft<0] = a + 1j * b
    
    imLGX = cx1 + cx2;
    
    return imLGX

def get_single_log_gabor(WV):
    imLGX = log_gabor(int(WV['waveletSizePix']),
                          WV['waveletImSizeCycles'],
                          WV['waveletSpatFreqBWOct'],
                          WV['oriDeg'],
                          WV['waveletOriBWDeg'],
                          WV['waveletPhaseDeg'])
    imLG_scrambled_X = imLGX
    return np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fftshift(imLG_scrambled_X))))
# TODO: Diagonalize
def get_wavelet_positions(expParameters, checkSizeWavelets, isCheckPhaseReversed,  ):
    imSizePix = int(expParameters['imSizePix'])

    im, imDotCoordinatesYX = get_dot_matrix_diamond(imSizePix, expParameters['stimulusProportionFilled'], expParameters['dotSpacingPix'])
    
    checkSizePix = expParameters['dotSpacingPix'] * checkSizeWavelets 

    # Extract X and Y coordinates from the dot matrix
    imDotCoordinatesX = imDotCoordinatesYX[:, :, 0]
    imDotCoordinatesY = imDotCoordinatesYX[:, :, 1]

    # Apply the mask to the coordinates

    if checkSizePix > 0:
        imMask = create_diagonal_checkered_pattern(imSizePix, checkSizePix, isCheckPhaseReversed)
        imDotCoordinatesX = imDotCoordinatesX * imMask
        imDotCoordinatesY = imDotCoordinatesY * imMask

    # Replace masked coordinates with NaN
    imDotCoordinatesX[imDotCoordinatesX == 0] = nan
    imDotCoordinatesY[imDotCoordinatesY == 0] = nan

    # Remove NaN values from the coordinates
    x = imDotCoordinatesX[~isnan(imDotCoordinatesX)]
    y = imDotCoordinatesY[~isnan(imDotCoordinatesY)]

    numWavelets = len(x) # number of wavelets to be drawn
    
    # Combine Y and X coordinates into a single array
    yx = empty((2, numWavelets))
    yx[0, :] = y
    yx[1, :] = x
    return yx


def make_coherence_stimulus(expParameters, rng, checkSizeWavelets, isCheckPhaseReversed, isFillBlankChecks, oriDeg, proportionCoherence, weberContrast,isDiagnonalCheckeredPattern = False, isSquareGrid=False,isDiamondGrid=False):
    
    yx = get_wavelet_positions(expParameters, checkSizeWavelets, isCheckPhaseReversed)
    numWavelets = yx.shape[1]
    
    # Start of code specific to coherence stimulus
    
    waveletThetaDeg        = rng.uniform(low = 0.0, high = 360.0, size = numWavelets)
    numCoherentWavelets    = round(proportionCoherence * numWavelets)
    coherentWaveletIndices = rng.choice(numWavelets, numCoherentWavelets, replace=False)
        
    print('Total number of wavelets: %0.0f, proportion coherence: %0.2f, number coherent wavelets: %0.0f' % (numWavelets,
                                                                                                            proportionCoherence,
                                                                                                            numCoherentWavelets))
    
    waveletThetaDeg[coherentWaveletIndices] = oriDeg
    
    # end of code specific to coherence stimulus
    
    # Render the wavelets from the defined parameters
    imC = render_from_wavelets(expParameters, yx, waveletThetaDeg)
    
    if isFillBlankChecks and checkSizeWavelets > 0:
        # fill blank checks with uniform random oriented wavelets
        blankSpaceFill = make_coherence_stimulus(expParameters, rng, checkSizeWavelets, not isCheckPhaseReversed, False, 0, 0, None)
        imC = imC + blankSpaceFill
        
    if weberContrast: # Normalize the image
        imC = (imC / numpymax(numpyabs(imC.flat))) * weberContrast

    return imC


def make_gaussian_stimulus(expParameters, rng, checkSizeWavelets, isCheckPhaseReversed, isFillBlankChecks, meanOriDeg, stdOriDeg, weberContrast,isDiamondGrid=False):
    yx = get_wavelet_positions(expParameters, checkSizeWavelets, isCheckPhaseReversed,isDiamondGrid)
    numWavelets = yx.shape[1]
    
    # Start of code specific to coherence stimulus
    
    waveletThetaDeg = rng.normal(loc = meanOriDeg, scale = stdOriDeg, size = numWavelets)

    print('Mean orientation = %0.2f, st dev = %0.2f' % (mean(waveletThetaDeg),std(waveletThetaDeg)))
        
    # end of code specific to coherence stimulus
    
    # Render the wavelets from the defined parameters
    imC = render_from_wavelets(expParameters, yx, waveletThetaDeg)
    
    if isFillBlankChecks:
        # fill blank checks with uniform random oriented wavelets
        blankSpaceFill = make_coherence_stimulus(expParameters, rng, checkSizeWavelets, not isCheckPhaseReversed, False, 0, 0, None, False)
        imC = imC + blankSpaceFill
        
    if weberContrast: # Normalize the image
        imC = (imC / numpymax(numpyabs(imC.flat))) * weberContrast

    return imC


def create_diagonal_checkered_pattern(imSizePix, checkSizePix, checkPhaseReverse):
    # Create a larger square to accommodate rotation
    large_size = int((2**0.5) * imSizePix)
    
    # Create initial checkered pattern in larger square
    x = np.linspace(0, large_size, large_size)
    y = np.linspace(0, large_size, large_size)
    xx, yy = np.meshgrid(x, y)
    
    # Create checkered pattern based on checkSizePix
    imCheck = np.zeros((large_size, large_size))
    check_period = int(checkSizePix)
    imCheck[(xx.astype(int)//check_period + yy.astype(int)//check_period) % 2 == 0] = 1
    
    # Rotate the pattern by 45 degrees
    center = large_size / 2
    angle = -np.pi/4  # negative for clockwise rotation
    
    # Create coordinate arrays for rotation
    xx_centered = xx - center
    yy_centered = yy - center
    
    # Apply rotation transformation
    xx_rot = xx_centered * np.cos(angle) - yy_centered * np.sin(angle) + center
    yy_rot = xx_centered * np.sin(angle) + yy_centered * np.cos(angle) + center
    
    # Use interpolation to get rotated image
    coords = np.array([yy_rot.flatten(), xx_rot.flatten()])
    imCheck_rotated = map_coordinates(imCheck, coords, order=1).reshape(large_size, large_size)
    
    # Cut out center square of size imSizePix x imSizePix
    start_idx = int((large_size - imSizePix) / 2)
    end_idx = start_idx + imSizePix
    final_check = imCheck_rotated[start_idx:end_idx, start_idx:end_idx]
    
    if checkPhaseReverse:
        final_check = 1 - final_check
        
    return final_check


def make_high_bit_depth_vpixx_window(doSetVideoMode=True):
  my_monitor = monitors.Monitor('VIEWPixxNoGamma')

  my_device = VIEWPixx() # Opens and initiates the device
  my_device.setVideoMode('M16'if doSetVideoMode else 'C24')
 # Set the right video mode
  my_device.updateRegisterCache() # Update the device

  win = visual.Window(size=(1920, 1200), useFBO=True, fullscr=True, screen=1, gamma=1.92,
                      monitor=my_monitor, units='pix')

  setUpShaderAndWindow(my_device, win)
  return win,my_monitor

def load_psi_parameters(psiParamFilePath):
    psiParamDict = {}
    with open(psiParamFilePath, mode = 'r') as f:
        experimentSettings = csv.reader(f)
        psiParamDict = {r[0]:float(r[1]) for r in experimentSettings}

    psiParamDict['minLogSigma'] = log(psiParamDict['minSigma'])
    psiParamDict['maxLogSigma'] = log(psiParamDict['maxSigma'])
    psiParamDict['doReflectX']  = True if (psiParamDict['doReflectX']==1.0) else False
    psiParamDict['isVerbose']   = True if (psiParamDict['isVerbose']==1.0)  else False

    return psiParamDict