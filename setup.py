"""Toolbox for control and optimization of water systems.

RTC-Tools is the Deltares toolbox for control and optimization of water systems.

"""
from setuptools import setup, find_packages
import versioneer
import sys

if sys.version_info[0] == 3 and sys.version_info[1] < 6:
    sys.exit("Sorry, Python 3.6 or newer is required.")

DOCLINES = __doc__.split("\n")

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Information Technology
License :: OSI Approved :: GNU General Public License v3 (GPLv3)
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering :: GIS
Topic :: Scientific/Engineering :: Mathematics
Topic :: Scientific/Engineering :: Physics
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

# Install requirements
with open('requirements.txt', 'r') as req_file:
    install_reqs = req_file.read().split('\n')

setup(
    name = 'rtctools',
    version = versioneer.get_version(),
    maintainer = 'Jorn Baayen',
    author = 'Jorn Baayen',
    description = DOCLINES[0],
    long_description = '\n'.join(DOCLINES[2:]),
    url = 'http://www.deltares.nl/en/software/rtc-tools/',
    download_url='http://gitlab.com/deltares/rtc-tools/',
    license = 'GPL',
    classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms = ['Windows', 'Linux', 'Mac OS-X', 'Unix'],
    packages = find_packages(),
    install_requires = install_reqs,
    tests_require = ['nose'],
    test_suite = 'nose.collector',
    cmdclass = versioneer.get_cmdclass(),
)
