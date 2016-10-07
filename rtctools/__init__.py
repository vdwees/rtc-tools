__version__ = '2.0.0-beta1'

print \
"""
******************************************************************************
             This program contains RTC-Tools {}, a toolbox for
              control and optimization of environmental systems.                           
    RTC-Tools is open source software under the GNU General Public License.
   For more information visit https://www.deltares.nl/en/software/rtc-tools/
******************************************************************************
""".format(__version__)

try:
	import casadi
except ImportError:
	# Don't require casadi for readthedocs.org
	from mock import MagicMock
	import warnings
	import sys

	warnings.warn("CasADi not found.  Building with mock CasADi instead.  This will result in problems.")

	MOCK_MODULES = ['casadi', 'pyfmi', 'pyfmi.fmi', 'pymodelica', 'pymodelica.compiler', 'pymodelica.compiler_exceptions', 'pyjmi', 'pylab', 'scipy', 'scipy.interpolate', 'matplotlib']
	sys.modules.update((mod_name, MagicMock()) for mod_name in MOCK_MODULES)