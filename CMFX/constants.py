### THREADS ###
import psutil
from scipy import constants
# Includes hyperthreads
n_threads = psutil.cpu_count()

### CONVERSIONS ###
TORR2DENSITY = 1.622032e-7 # At 25 C
IN2CM = 2.54

### Constants ###
ProtonMass = constants.m_p

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

### CHAMBER ###
centerCondutor_radius = 1.5 * IN2CM
centerCondutor_length = 100
centerCondutor_thickness = 0.065 * IN2CM

chamber_radius = 36.1
chamber_length = centerCondutor_length
chamber_thickness = 0.95

### PLASMA ###
plasma_outerRadius = 25
plasma_innerRadius = 5
plasma_length = 60
Ti_min = 0.2 # keV, # Lower limit found in Table VII of https://iopscience.iop.org/article/10.1088/0029-5515/32/4/I07/pdf
duration = 0.2 # s

### SIMULATION ###
void_radius = 150
void_length = 120
particles = 50000

root = '.'
sweep_folder = 'param_sweeps'
examples_folder = 'examples'
source_pkl = 'source.pkl'
file_prefix = 'run'