import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load Landsat 8 Panchromatic Band Response
l8_pan = pd.read_csv('L8_B8_RSR.csv')
l8_wavelength = l8_pan.iloc[:, 0].values  # Wavelength (nm)
l8_response = l8_pan.iloc[:, 1].values    # Relative Spectral Response

# Load Sentinel-2A RSRs for B1-B5
s2a_rsr = pd.read_csv('SENT2A-RSR.csv')
s2a_wavelength = s2a_rsr['SR_WL'].values
s2a_bands = ['S2A_SR_AV_B1', 'S2A_SR_AV_B2', 'S2A_SR_AV_B3', 'S2A_SR_AV_B4', 'S2A_SR_AV_B5']

# Plot the Panchromatic band response
plt.figure(figsize=(10, 6))
plt.plot(l8_wavelength, l8_response, label='Landsat 8 Panchromatic (B8)', color='black', linewidth=2)

# Plot Sentinel-2A band responses as lines
colors = ['b', 'g', 'r', 'c', 'm']
for i, band in enumerate(s2a_bands):
    plt.plot(s2a_wavelength, s2a_rsr[band].values, label=band.replace('S2A_SR_AV_', 'S2A B'), color=colors[i])

plt.xlabel('Wavelength (nm)')
plt.ylabel('Relative Spectral Response')
plt.title('Landsat 8 Panchromatic Band and Sentinel-2A Bands (B1-B5)')
plt.legend(loc='upper right', fontsize=8)
plt.ylim(0, 1.2)
plt.tight_layout()
plt.show()

# Calculate suggested weightings for each S2A band (B1-B5)
# Weighting = sum of (S2A_RSR * L8_Pan_RSR) over all wavelengths, normalized
weightings = {}
# Interpolate L8 response to S2A wavelengths for fair comparison
l8_interp = np.interp(s2a_wavelength, l8_wavelength, l8_response)
for band in s2a_bands:
    s2a_rsr_band = s2a_rsr[band].values
    overlap = np.sum(s2a_rsr_band * l8_interp)
    weightings[band.replace('S2A_SR_AV_', 'S2A B')] = overlap

# Normalize weightings so they sum to 1
total = sum(weightings.values())
for band in weightings:
    weightings[band] /= total

print('Suggested weightings for simulating a panchromatic band from Sentinel-2A (B1-B5):')
for band, w in weightings.items():
    print(f'{band}: {w:.3f}') 