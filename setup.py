"""Toolbox for control and optimization of water systems.

RTC-Tools is the Deltares toolbox for control and optimization of water systems.

"""
from setuptools import setup, find_packages
import versioneer
import sys

if sys.version_info < (3, 5):
    sys.exit("Sorry, Python 3.5 or newer is required.")

DOCLINES = __doc__.split("\n")

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Information Technology
License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)
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

setup(
    name = 'rtc-tools',
    version = versioneer.get_version(),
    maintainer = 'Jorn Baayen',
    author = 'Jorn Baayen',
    description = DOCLINES[0],
    long_description = '\n'.join(DOCLINES[2:]),
    url = 'http://www.deltares.nl/en/software/rtc-tools/',
    download_url='http://gitlab.com/deltares/rtc-tools/',
    classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms = ['Windows', 'Linux', 'Mac OS-X', 'Unix'],
    packages = find_packages("src"),
    package_dir = {"": "src"},
    install_requires = ["casadi >= 3.2.0",
                        "numpy >= 1.14.0",
                        "scipy >= 1.0.0",
                        "pymoca >= 0.2.4"],
    tests_require = ['nose'],
    test_suite = 'nose.collector',
    python_requires='>=3.5',
    cmdclass = versioneer.get_cmdclass(),
)
