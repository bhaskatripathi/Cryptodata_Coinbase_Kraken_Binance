# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 20:00:14 2023

@author: bhask
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# Radar Factory Function
def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    class RadarAxes(PolarAxes):
        name = 'radar'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)
        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
        def draw(self, renderer):
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)
        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self, spine_type='circle', path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
    register_projection(RadarAxes)
    return theta

# Plotting function
def plot_radar_chart(data_values, title, grid_scale):
    fig, ax = plt.subplots(figsize=(6, 6),dpi=300 ,subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=.96, bottom=0.1)  # Adjusting the plot space

    spoke_labels = exchanges
    N = len(spoke_labels)
    theta = radar_factory(N, frame='polygon')
    labels = ['Interval 1', 'Interval 2', 'Interval 3']
    #colors = ['b', 'r', 'g']  # Define the color for each interval

    ax.set_rgrids(grid_scale)
    ax.set_title(title, position=(.1, 1.1), ha='center', weight='bold', size='medium')

    for idx, d in enumerate(data_values):
        ax.plot(theta, d )
        ax.fill(theta, d,  alpha=0.25)

    ax.set_varlabels(spoke_labels)
    legend = ax.legend(labels, bbox_to_anchor=(1.05, 1.0), labelspacing=0.1, fontsize='small')
    plt.show()

# Data
exchanges = ["NASDAQ100", "NYSE Composite", "NIFTY", "BSE SENSEX", "Binance", "Kraken", "Coinbase Pro"]

# The radar_factory function code remains the same as what was provided before. 
theta = radar_factory(len(exchanges), frame='polygon')


mli_values = [
    [1.2874E-04, 5.2687E-04, 2.6878E-02, 6.9096E+02, 6.9097E+02, 8.9826E+02, 1.1056E+03],
    [7.6459E-03, 5.7197E-03, 3.7788E+01, 1.8243E+02, 1.6354E+02, 2.8811E+02, 2.3527E+02],
    [2.6753E+00, 2.0410E-01, 3.0788E+01, 7.6780E+01, 2.9358E+02, 3.9513E+02, 6.7580E+02]
]

amihud_values = [
    [2.8026E-12, 1.4219E-12, 3.4945E-10, 3.6524E-07, 4.0812E-07, 1.0850E-06, 1.4599E-06],
    [2.9147E-12, 1.9111E-12, 1.0906E+05, 2.2661E+07, 2.2246E+07, 6.3029E+07, 5.1906E+07],
    [2.9109E-12, 1.9889E-12, 9.9774E+04, 2.0352E+07, 2.5503E+07, 4.2936E+08, 6.6274E+08]
]

ar_bid_ask_spread_values = [
    [2.8056E+02, 3.7922E+02, 3.8614E+02, 1.6988E+03, 1.7516E+03, 2.1777E+03, 2.6039E+03],
    [7.8898E+02, 6.3122E+02, 9.2599E+02, 2.8133E+03, 2.6659E+03, 4.0615E+03, 3.4374E+03],
    [8.3418E+02, 6.0988E+02, 8.6840E+02, 2.5829E+03, 2.8012E+03, 3.1615E+03, 3.8058E+03]
]

cs_estimator_values = [
    [8.2423E-05, 6.0441E-05, 1.3147E-04, 1.4214E-04, 1.6667E-04, 1.9168E-04, 2.1670E-04],
    [2.4128E-04, 2.4590E-04, 4.1167E-04, 4.2392E-04, 2.7039E-04, 5.2921E-04, 4.7657E-04],
    [2.5928E-04, 2.3166E-04, 2.6588E-04, 3.8223E-04, 3.9934E-04, 4.9100E-04, 4.4517E-04]
]

rolls_estimator_values = [
    [2.9076E-10, 2.4964E-10, 3.4528E-10, 3.6256E-10, 3.7985E-10, 6.9713E-10, 7.1441E-10],
    [1.4470E-11, 1.1000E-11, 2.7840E-10, 5.2380E-10, 3.9010E-10, 1.1701E-09, 8.2743E-10],
    [4.9183E-10, 3.5671E-10, 8.6883E-10, 1.0323E-09, 8.9717E-10, 1.1203E-09, 1.1674E-09]
]


# Plotting the radar chart for MLI Values
#plot_radar_chart(mli_values, 'Martin Liquidity Index', [i for i in range(0, 1300, 100)])
 

# Applying logarithmic transformation to amihud_values
#####################################################################
# Define the bounds and the scaling factor
min_bound = 1.81898940354586E-12
max_bound = 1.073741824E9
scaling_factor = 64

# Take the base-10 logarithm of the amihud values
amihud_values_log = [[np.log10(val + 1e-10) for val in sublist] for sublist in amihud_values]  # 1e-10 to avoid log(0)

# Original min and max values for logarithmic amihud values
original_min_amihud_log = min(map(min, amihud_values_log))
original_max_amihud_log = max(map(max, amihud_values_log))

# Scale transformation function
def scale_value(value, original_min, original_max, target_min, target_max):
    return target_min + ((value - original_min) / (original_max - original_min)) * (target_max - target_min)

# Apply the scaling to the logarithmic amihud_values
amihud_values_scaled = [[scale_value(val, original_min_amihud_log, original_max_amihud_log, min_bound, max_bound) * scaling_factor for val in sublist] for sublist in amihud_values_log]

# Update the max_values dictionary for amihud
#max_values['amihud'] = max(map(max, amihud_values_scaled))

#####################################################################

# Calculating the maximum value for each data set
max_values = {
    "mli": max(map(max, mli_values)),
    "amihud": max(map(max, amihud_values_scaled)),
    "ar_bid_ask_spread": max(map(max, ar_bid_ask_spread_values)),
    "cs_estimator": max(map(max, cs_estimator_values)),
    "rolls_estimator": max(map(max, rolls_estimator_values))
}

# Generating the ranges for each data set based on their maximum values
ranges = {}
for key, value in max_values.items():
    if value < 10:  # If the max value is very small, we adjust the step accordingly
        ranges[key] = [i for i in np.arange(0, value*1.1, value/10)]
    else:
        ranges[key] = [i for i in range(0, int(value * 1.1), int(value / 10))]

print(ranges)

plot_radar_chart(mli_values, 'Martin Liquidity Index', ranges['mli'])
plot_radar_chart(amihud_values_scaled, 'Amihud Illiquidity Measure', ranges['amihud'])
plot_radar_chart(ar_bid_ask_spread_values, 'Average Relative Bid-Ask Spread', ranges['ar_bid_ask_spread'])
plot_radar_chart(cs_estimator_values, 'CS Estimator', ranges['cs_estimator'])
plot_radar_chart(rolls_estimator_values, 'Roll\'s Estimator', ranges['rolls_estimator'])
