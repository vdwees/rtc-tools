__version__ = '2.0.0-beta2'

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

	warnings.warn("CasADi not found.  Please ensure that casadi can be imported before running RTC-Tools.")

	MOCK_MODULES = ['casadi', 'pyfmi', 'pyfmi.fmi', 'pymodelica', 'pymodelica.compiler', 'pymodelica.compiler_exceptions', 'pyjmi', 'pylab', 'scipy', 'scipy.interpolate']
	sys.modules.update((mod_name, MagicMock()) for mod_name in MOCK_MODULES)