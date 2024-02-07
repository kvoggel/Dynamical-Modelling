import numpy as np
from sersic2mge import sersic2mge

"""
Example usage of sersic2mge code that uses two separate sersic profiles of the same galaxy (centered on the same position) to generate an output MGE.
}
Returns: A file called mge_output.txt that contains the MGE of each component. 
This file can now be used as input to JAM models.

Written by Karina Voggel, last update 7. Feb 2024

"""
# Solar magnitude and AB magnitude offset                         
scale=0.05        # Scale of each pixel in "/pixel (here we have HST WFC3)
Msun=4.52			#Magnitude of the sun in the given filter. Here we use HST F814. Other filters can be found e.g. here https://iopscience.iop.org/article/10.3847/1538-4365/aabfdf
AB=0.023			# Foreground extinction at the location of the galaxy

# Define the Sersic parameters
sersic_params = {
    're': 8.63 *scale,  # Effective radius multiplied by the scale
    'mag': 16.33,        # Magnitude
    'n': 5.04,           # Sersic index
    'pa': -52.15,        # Position angle
    'q': 0.668           # Axis ratio
}

sersic_params2 = {
    're': 471.62 *scale,  # Effective radius multiplied by the scale
    'mag': 10.228,        # Magnitude
    'n': 2.49,           # Sersic index
    'pa': -10.96,        # Position angle
    'q': 0.668           # Axis ratio
}

# Call the sersic2mge function with the parameters
mge_result1 = sersic2mge(sersic_params, Msun, A_b=AB)
mge_result2 = sersic2mge(sersic_params2, Msun, A_b=AB)

mge_total = vstack(mge_result1, mge_result2)
# Save the result to a text file
output_filename = "mge_output.txt"
np.savetxt(output_filename, mge_total, fmt='%10.5f', header='Luminosity [Lsun/pc^2]  sigma ["]  q  P.A. [degree]', comments='')
print(f"MGE results saved to {output_filename}")

