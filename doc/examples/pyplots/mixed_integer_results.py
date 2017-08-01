import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


data_path = '../../../examples/mixed_integer/output/timeseries_export.csv'
delimiter = ','

# Import Data
ncols = len(np.genfromtxt(data_path, max_rows=1, delimiter=delimiter))
datefunc = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
results = np.genfromtxt(data_path, converters={0: datefunc}, delimiter=delimiter,
                        dtype='object' + ',float' * (ncols - 1), names=True)[1:]

# Generate Plot
f, axarr = plt.subplots(2, sharex=True)
axarr[0].set_title('Water Level and Discharge')

# Upper subplot
axarr[0].set_ylabel('Water Level [m]')
axarr[0].plot(results['time'], results['storage_level'], label='Storage',
              linewidth=2, color='b')
axarr[0].plot(results['time'], results['sea_level'], label='Sea',
              linewidth=2, color='m')
axarr[0].plot(results['time'], 0.5 * np.ones_like(results['time']), label='Storage Max',
              linewidth=2, color='r', linestyle='--')

# Lower Subplot
axarr[1].set_ylabel('Flow Rate [m\u00B3/s]')
axarr[1].plot(results['time'], results['Q_orifice'], label='Orifice',
              linewidth=2, color='g')
axarr[1].plot(results['time'], results['Q_pump'], label='Pump',
              linewidth=2, color='r')
axarr[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
f.autofmt_xdate()

# Shrink each axis by 20% and put a legend to the right of the axis
for i in range(len(axarr)):
    box = axarr[i].get_position()
    axarr[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axarr[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

# Output Plot
plt.show()
