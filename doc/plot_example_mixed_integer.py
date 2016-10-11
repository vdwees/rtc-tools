import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Import Data
datefunc = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
results = np.genfromtxt('../examples/mixed_integer/output/timeseries_export.csv',
                        converters={0: datefunc},
                        dtype='object, float, float, float, float, float',
                        delimiter=",",
                        names=True)

# Generate Plot
f, axarr = plt.subplots(2, sharex=True)
axarr[0].set_title('Water Level and Discharge')
axarr[0].plot(results['time'], results['storage_level'], label='Storage', linewidth=2, color='b')
axarr[0].plot(results['time'], results['sea_level'], label='Sea', linewidth=2, color='m')
axarr[0].set_ylabel('Water Level [m]')
axarr[1].plot(results['time'], results['Q_orifice'], label='Orifice', linewidth=2, color='g')
axarr[1].plot(results['time'], results['Q_pump'], label='Pump', linewidth=2, color='r')
axarr[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axarr[1].set_ylabel('Flow Rate [m3/s]')
f.autofmt_xdate()

# Shrink current axis by 20%
box0 = axarr[0].get_position()
box1 = axarr[1].get_position()
axarr[0].set_position([box0.x0, box0.y0, box0.width * 0.8, box0.height])
axarr[1].set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])

# Put a legend to the right of the current axis
axarr[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
axarr[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Output Plot
plt.savefig('./images/mixed_integer_resultplot.png')
