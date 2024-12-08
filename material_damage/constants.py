### THREADS ###
import psutil
# Includes hyperthreads
n_threads = psutil.cpu_count()

### CONVERSIONS ###
TORR2DENSITY = 1.622032e-7 # At 25 C
IN2CM = 2.54

# ALL LENGTH UNITS IN CM
### TARGET ###
target_width = 120
target_height = target_width
target_thickness = 120

### SIMULATION ###
N_mesh_points = 1000

root = '.'
sweep_folder = 'param_sweeps'
examples_folder = 'examples'
figures_folder = 'figures'
source_pkl = 'source.pkl'
file_prefix = 'run'