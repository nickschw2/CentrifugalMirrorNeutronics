### THREADS ###
import psutil
from scipy import constants
# Includes hyperthreads
n_threads = psutil.cpu_count()

### CONVERSIONS ###
TORR2DENSITY = 1.622032e-7 # At 25 C
IN2CM = 2.54

# ALL LENGTH UNITS IN CM

### DETECTOR ###
# LND 2528: https://www.lndinc.com/products/neutron-detectors/2528/
gas_pressure = 3040 # Torr
He3_diameter = 0.96 * IN2CM
He3_length = 8 * IN2CM

detector_diameter = 1 * IN2CM
detector_length = 11.34 * IN2CM
gas_offset = 0.87 * IN2CM

HDPE_height = 13 * IN2CM
HDPE_diameter = 5 * IN2CM

Pb_thickness = 1/16 * IN2CM

enclosure_thickness = 0.07 * IN2CM
enclosure_width = 5.25 * IN2CM
enclosure_length = enclosure_width
enclosure_height = 28 * IN2CM

DET_NUM = 1
duration = 100 # s

### SIMULATION ###
void_radius = 150

root = '.'
sweep_folder = 'param_sweeps'
source_pkl = 'source.pkl'
file_prefix = 'run'