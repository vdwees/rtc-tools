import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pylab import get_cmap

forecast_names = ['forecast1', 'forecast2']

# Import Data
def get_results(forecast_name):
    output_dir = '../../../examples/ensemble/output/'
    data_path = output_dir + forecast_name + '/timeseries_export.csv'
    delimiter = ','
    ncols = len(np.genfromtxt(data_path, max_rows=1, delimiter=delimiter))
    datefunc = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return np.genfromtxt(data_path, converters={0: datefunc}, delimiter=delimiter,
                            dtype='object' + ',float' * (ncols - 1), names=True)

# Generate Plot
n_subplots = 2
f, axarr = plt.subplots(n_subplots, sharex=True, figsize=(8, 4 * n_subplots))
axarr[0].set_title('Optimized Water Volume and Discharge')
cmaps = ['Blues', 'Greens']
shades = [0.5, 0.8]
f.autofmt_xdate()

# Upper subplot
for idx, forecast in enumerate(forecast_names):
    results = get_results(forecast)
    axarr[0].set_ylabel('Water Volume in Storage [m3]')
    if idx == 0:
        axarr[0].plot(results['time'], results['V_max'], label='Max',
        linewidth=2, color='r', linestyle='--')
        axarr[0].plot(results['time'], results['V_min'], label='Min',
        linewidth=2, color='g', linestyle='--')
    axarr[0].plot(results['time'], results['V_storage'], label=forecast + ':volume',
                        linewidth=2, color=get_cmap(cmaps[idx])(shades[1]))

    # Lower Subplot
    axarr[1].set_ylabel('Flow Rate [m3/s]')
    axarr[1].plot(results['time'], results['Q_in'], label='{}:Inflow'.format(forecast),
                            linewidth=2, color=get_cmap(cmaps[idx])(shades[0]))
    axarr[1].plot(results['time'], results['Q_release'], label='{}:Release'.format(forecast),
                            linewidth=2, color=get_cmap(cmaps[idx])(shades[1]))

# Shrink each axis by 30% and put a legend to the right of the axis
for i in range(len(axarr)):
    box = axarr[i].get_position()
    axarr[i].set_position([box.x0, box.y0, box.width * 0.7, box.height])
    axarr[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

# Output Plot
plt.show()
