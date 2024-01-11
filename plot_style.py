import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools

line_styles = ['-','--','-.',':']
style_cycler = itertools.cycle(line_styles)

mpl.rcParams.update({'lines.linewidth': 2})
mpl.rcParams.update({'font.size': 14})
mpl.rcParams.update({'axes.grid': True})
mpl.rcParams.update({'axes.grid.which': 'major'})

# Error bars
capsize = 5

colors = {'qualitative': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999'],
          'sequential': ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']}
default_color_cycle = 'qualitative'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors[default_color_cycle])

def reset_colors(type):
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors[type])

def reset_style_cycler():
    style_cycler = itertools.cycle(line_styles)