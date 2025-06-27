import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace, meshgrid, pi, sin
from scipy.ndimage import map_coordinates

imSizePix = 1000
stimulusProportionFilled = 0.8
dotSpacingPix = 10

def get_dot_matrix(imSizePix,stimulusProportionFilled,dotSpacingPix,isSquareGrid=False): #This needs to make a circle of dots, not a square, 
    # square can be cut out later, think that should be a separat function that is then passed to get wavelet positions
    
    numDotsCanFitWidth = (imSizePix * stimulusProportionFilled) // dotSpacingPix + 1
    oddNumDotsCanFitWidth = 2*(numDotsCanFitWidth // 2) - 1
    dotRegionPix = (oddNumDotsCanFitWidth) * dotSpacingPix
    borderGapPix = np.ceil((imSizePix-dotRegionPix)/2.0)
    dotRegionPix = imSizePix - 2*borderGapPix
    dotPos  = np.arange(borderGapPix,borderGapPix+dotRegionPix+dotSpacingPix,dotSpacingPix)
    dotPosX = np.empty([len(dotPos)**2])
    dotPosY = np.empty([len(dotPos)**2])

    ind = 0
    if isSquareGrid:
        for i,x in enumerate(dotPos):
            for j,y in enumerate(dotPos):
                dotPosX[ind] = x
                if (i%2) == 1:
                    dotPosY[ind] = y 
                else:
                    dotPosY[ind] = y 
                ind = ind + 1
    else:
        for i,x in enumerate(dotPos):
            for j,y in enumerate(dotPos):
                dotPosX[ind] = x
                if (i%2) == 1:
                    dotPosY[ind] = y + dotSpacingPix/4
                else:
                    dotPosY[ind] = y - dotSpacingPix/4
                ind = ind + 1

    imStim = np.zeros([imSizePix,imSizePix])
    imDotCoordinatesYX = np.zeros([imSizePix,imSizePix,2])

    for i,x in enumerate(dotPosX):
        y = dotPosY[i]
        imStim[int(x),int(y)] = 1
        imDotCoordinatesYX[int(x),int(y),0] = int(x)
        imDotCoordinatesYX[int(x),int(y),1] = int(y)

    return (imStim,imDotCoordinatesYX)

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
    
    # Plot 1: Initial dot positions
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.scatter(dotPosX, dotPosY)
    plt.title('Initial Dot Positions')
    plt.axis('equal')
    
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

    # Plot 2: Rotated dot positions
    plt.subplot(132)
    plt.scatter(rotated_x, rotated_y)
    plt.title('Rotated Dot Positions')
    plt.axis('equal')

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

    # Plot 3: Final cropped pattern
    plt.subplot(133)
    plt.imshow(imStim)
    plt.title('Final Cropped Pattern')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

    return imStim, imDotCoordinatesYX
def create_checkered_pattern(imSizePix, checkSizePix, checkPhaseReverse):
    relativeCheckSize = checkSizePix / imSizePix
    x = linspace(-pi/relativeCheckSize,pi/relativeCheckSize,imSizePix)
    xx,yy = meshgrid(x,x)
    imCheckY = sin(yy)
    imCheckX = sin(xx)
    imCheck  = imCheckY * imCheckX
    imCheck[imCheck>0] = 1
    imCheck[imCheck<1] = 0
    if checkPhaseReverse:
        imCheck = 1-imCheck
    return imCheck

def create_diagonal_checkered_pattern(imSizePix, checkSizePix, checkPhaseReverse):
    # Create a larger square to accommodate rotation
    large_size = int((2**0.5)*(imSizePix))
    
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
# Test the function and plot results
diamondGrid, coordinates = get_dot_matrix_diamond(imSizePix, stimulusProportionFilled, dotSpacingPix)
plt.imshow(diamondGrid)
plt.title('Diamond Dot Pattern')
plt.show()
# Test both checkered patterns
normal_check = create_checkered_pattern(imSizePix, 2*dotSpacingPix, False)
diagonal_check = create_diagonal_checkered_pattern(imSizePix, 2*dotSpacingPix, False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.imshow(normal_check)
ax1.set_title('Normal Checkered Pattern')

ax2.imshow(diagonal_check)
ax2.set_title('Diagonal Checkered Pattern')

plt.show()


