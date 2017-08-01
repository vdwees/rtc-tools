import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


data_path = '../../../examples/lookup_table/output/timeseries_export.csv'
delimiter = ','

# Import Data
ncols = len(np.genfromtxt(data_path, max_rows=1, delimiter=delimiter))
datefunc = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
results = np.genfromtxt(data_path, converters={0: datefunc}, delimiter=delimiter,
                        dtype='object' + ',float' * (ncols - 1), names=True)[1:]

# Generate Plot
n_subplots = 2
f, axarr = plt.subplots(n_subplots, sharex=True, figsize=(8, 3 * n_subplots))
axarr[0].set_title('Water Volume and Discharge')
f.autofmt_xdate()

# Upper subplot
axarr[0].set_ylabel('Water Volume [m³]')
axarr[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
axarr[0].plot(results['time'], results['storage_V'], label='Storage',
              linewidth=2, color='b')
axarr[0].plot(results['time'], results['V_max'], label='Storage Max',
              linewidth=2, color='r', linestyle='--')
axarr[0].plot(results['time'], results['V_min'], label='Storage Min',
              linewidth=2, color='g', linestyle='--')

# Lower Subplot
axarr[1].set_ylabel('Flow Rate [m³/s]')
axarr[1].plot(results['time'], results['Q_in'], label='Inflow',
              linewidth=2, color='g')
axarr[1].plot(results['time'], results['Q_release'], label='Release',
              linewidth=2, color='r')

# Shrink each axis by 20% and put a legend to the right of the axis
for i in range(n_subplots):
    box = axarr[i].get_position()
    axarr[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axarr[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

# Output Plot
plt.show()
