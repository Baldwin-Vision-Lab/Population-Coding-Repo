import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
n_patches = 16
orientations = np.linspace(-90, 90, 180)
true_orientation = 30  # degrees
noise_sigma = 40
signal_sigma = 10
blank_sigma = 1e6
eta = 0.1

# Stimulus types: 'signal', 'noise', 'blank'
stimulus_types = ['signal']*8 + ['noise']*4 + ['blank']*4
np.random.shuffle(stimulus_types)

# Patch orientations
patch_orientations = []
for s in stimulus_types:
    if s == 'signal':
        patch_orientations.append(true_orientation)
    elif s == 'noise':
        patch_orientations.append(np.random.uniform(-90, 90))
    else:
        patch_orientations.append(None)

# Reliability scores (can be learned)
reliability_scores = {'signal': 1.0, 'noise': 0.5, 'blank': 0.01}

# Likelihoods
posterior = np.ones_like(orientations)
for ori, s_type in zip(patch_orientations, stimulus_types):
    if s_type == 'blank':
        likelihood = np.ones_like(orientations) / len(orientations)
    else:
        sigma = signal_sigma if s_type == 'signal' else noise_sigma
        likelihood = norm.pdf(orientations, loc=ori, scale=sigma)
    # Adaptive weighting
    r = reliability_scores[s_type]
    weighted_likelihood = likelihood ** r
    posterior *= weighted_likelihood

# Normalize and decode
posterior /= np.sum(posterior)
decoded_orientation = orientations[np.argmax(posterior)]

# Visualization
plt.plot(orientations, posterior, label='Posterior')
plt.axvline(true_orientation, color='green', linestyle='--', label='True Orientation')
plt.axvline(decoded_orientation, color='red', linestyle='--', label='Decoded Orientation')
plt.title('Adaptive Bayesian Inference - One Trial')
plt.legend()
plt.xlabel('Orientation (degrees)')
plt.ylabel('Probability')
plt.show()
