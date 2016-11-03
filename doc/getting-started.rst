Getting Started
+++++++++++++++

Installation
============

For most users, especially on Windows, the easiest way to install RTC-Tools
and its dependencies is using the `Windows Installer`_. For users on Linux, it
is necessary to build it `from source`_ using git.

Windows Installer
-----------------

A complete Windows installer is available from the `Deltares Download
Portal`_. This will install a self-containing RTC-Tools 2 configuration
including all prerequisites, and will not conflict with other Python or
JModelica installations on the system.

1. Download the installer from the `Deltares Download Portal`_.

2. Start the installation by opening the downloaded executable.

3. Choose the desired location for RTC-Tools 2.

4. Run the Basic Example via the Start Menu.

If the installation was succesful, you should see that the solver succeeds:

.. image:: images/basic_example_console.png

From Source
-----------

Although not required, it is recommended to build and install RTC-Tools and
JModelica (see `dependencies`_) in a `virtual environment
<https://virtualenv.pypa.io/en/stable/>`_.

Dependencies
~~~~~~~~~~~~

RTC-Tools 2 has the following system dependencies:

* `Python <https://www.python.org>`_ >= 2.7 (*not Python 3*)

* `JModelica <https://svn.jmodelica.org/branches/CasADiUpdate24/>`_ == 1.16
  (CasADi 2.4) with:

  - `CasADi <https://github.com/casadi/casadi/>`_ ~> 2.4.5

  - `BONMIN <http://www.coin-or.org/download/source/Bonmin/>`_ >= 1.8.4
    (required for mixed integer optimization)

  - `IPOPT <http://www.coin-or.org/download/source/Ipopt/>`_ >= 3.12

For most Linux distributions only Python is available from the standard
repositories. JModelica and its dependencies will need to be built and
installed from source. We refer to their respective installation instructions,
and list below the instructions pertaining to RTC-Tools itself.

Acquiring the source
~~~~~~~~~~~~~~~~~~~~

The latest RTC-Tools source can be downloaded using git::

    # get RTC-Tools source
    git clone https://gitlab.com/deltares/rtc-tools.git

    # Get RTC-Tools's Modelica library
    git clone https://gitlab.com/deltares/rtc-tools-channel-flow.git

Ubuntu / Debian
~~~~~~~~~~~~~~~

Building RTC-Tools requires one additional Python package over JMmodelica::

    # Change directory to where RTC-Tools was downloaded
    cd rtc-tools

    # Install additional dependencies of RTC-Tools
    pip install mock

Now all that remains is to actually build and install RTC-Tools::

    python setup.py install

To check whether the installation was succesful, the basic example can be
used. It is importent first to set the correct environment variables for
JModelica and RTC-Tools. Luckily, JModelica comes with a convenient script
which does most of this for you. Only the environment variable pointing to the
Deltares Modelica library remains for the user to set::

    export DELTARES_LIBRARY_PATH=\`readlink -f ../rtc-tools-channel-flow\`

    cd examples/basic/src

    # Set the correct environment variables, and run the example
    /path/to/JModelica/bin/jm_python.sh example.py

Windows
~~~~~~~

Building RTC-Tools on Windows is easiest by using the `JModelica SDK
<http://www.jmodelica.org/sdk>`_. Be sure to:

* Build using JModelica's CasADi 2.4 branch

* Update CasADi to the required version (see `dependencies`_)

A further dependency is on Cython. Instructions for building extensions on
Windows are available in the `Cython docs
<https://github.com/cython/cython/wiki/CythonExtensionsOnWindows>`_.

Using the Visual C++ 2008 32-bit Command Prompt, it is then possible to build
and install RTC-Tools by running::

    python setup.py install

To check whether the installation was succesful, the basic example can be
used. It is importent first to set the correct environment variables for
JModelica and RTC-Tools. Luckily, JModelica comes with a convenient script
which does this for you. Only the environment variable pointing to the
Deltares Modelica library remains for the user to set::

    set DELTARES_LIBRARY_PATH=C:\path\to\rtc-tools-channel-flow

    cd /D C:\path\to\rtc-tools\basic\src

    # Set the correct environment variables, and run the example
    C:\path\to\JModelica\Python.bat example.py

.. _Deltares Download Portal: https://download.deltares.nl/en/download/rtc-tools/


.. _running-rtc-tools:

Running RTC-Tools
=================


RTC-Tools is run from a command line shell. If you installed using the Windows
executable, the RTC-Tools Shell can be started by clicking::

    Start -> Programs -> RTC-Tools -> Shell


Once you have started the shell, navigate to the ``src`` directory of the case
you wish to optimize, e.g.::

    cd \path\to\rtc-tools\examples\basic\src

Then, to run the case with RTC-Tools, run the ``src`` python script, e.g.::

    python example.py

You will see the progress of RTC-Tools in your shell. All your standard shell
commands can be used in the RTC-Tools shell. For example, you can use::

    python example.py > log.txt

to pipe RTC-Tools output to a log file.
