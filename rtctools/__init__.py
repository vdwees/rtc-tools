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

# Import dependencies
try:
	import casadi
except ImportError:
	# Don't require casadi for readthedocs.org
	from mock import MagicMock
	import warnings
	import sys

	warnings.warn("CasADi not found.  Please ensure that casadi can be imported before running RTC-Tools.")

	MOCK_MODULES = ['casadi', 'pymola', 'pylab', 'scipy', 'scipy.interpolate']
	sys.modules.update((mod_name, MagicMock()) for mod_name in MOCK_MODULES)
