# Get version
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# Print header
print(
"""
*****************************************************************************
  This program contains RTC-Tools {}, 
  a toolbox for control and optimization of water systems.                        
  RTC-Tools is open source software under the GNU General Public License.
  For more information visit https://www.deltares.nl/en/software/rtc-tools/
*****************************************************************************
""".format(__version__))
