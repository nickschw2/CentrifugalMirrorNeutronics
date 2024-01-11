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

### PLASMA ###
plasma_outerRadius = 25
plasma_innerRadius = 5
plasma_length = 60

### SIMULATION ###
void_radius = 100
particles = 50000

root = '.'
sweep_folder = 'param_sweeps'
file_prefix = 'run'