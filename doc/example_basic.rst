Filling a Reservoir
~~~~~~~~~~~~~~~~~~~

.. insert image here? e.g.
   https://commons.wikimedia.org/wiki/File:Badgernet_Clywedog_reservoir_3.JPG

Overview
--------

The purpose of this example is to understand the technical setup of
an RTC-Tools model, how to run the model, and how to interpret the results.

The scenario is the following: A reservoir operator is trying to fill a
reservoir. They are given a six-day forcast of inflows given in 12-hour
increments. The operator wants to save as much of the inflows as possible, but
does not want to end up with too much water at the end of the six days. They
have chosen to use RTC-Tools to calculate how much water to release and when
to release it.

The folder ``<installation directory>\RTCTools2\examples\basic``
contains a complete RTC-Tools optimization problem. An RTC-Tools
directory has the following structure:

* ``input``: This folder contains the model input data. These are several files
  in comma separated value format, ``csv``.
* ``model``: This folder contains the Modelica model. The Modelica model
  contains the physics of the RTC-Tools model.
* ``output``: The folder where the output is saved in the file
  ``timeseries_export.csv``.
* ``src``: This folder contains a Python file. This file contains the
  configuration of the model and is used to run the model .

The Model
---------

The first step is to develop a physical model of the system.
The model can be viewed and editied using the OpenModelica Connection Editor
program. Make sure to load the Deltares libray before loading the example:

#. Load the Deltares library into OpenModelica Conection Editor

   * Using the menu bar: *File -> Open Model/Library File(s)*
   * Select ``<installation directory>\RTCTools2\mo\Deltares\package.mo``

#. Load the example model into OpenModelica Conection Editor

   * Using the menu bar: *File -> Open Model/Library File(s)*
   * Select ``<installation
     directory>\RTCTools2\examples\basic\model\Example.mo``

Once loaded, we have an OpenModelica Connection Editor window that looks like
this:

.. image:: images/simple_storage_openmodelica.png


The model ``Example.mo`` represents a simple system with the following
elements:

* a reservoir, modelled as storage element
  ``Deltares.ChannelFlow.SimpleRouting.Storage.Storage``,
* an inflow boundary condition
  ``Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Inflow``,
* an outfall boundary condition
  ``Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Terminal``,
* connectors (black lines) connecting the elements.

You can use the mouse-over feature help to identify the pre-defined models from
the Deltares library. You can also drag the elements around- the connectors will
move with the elements. Adding new elements is easy- just drag them in from the
Deltares Library on the sidebar. Connecting the elements is just as easy- click
and drag between the ports on the elements.

In text mode, the Modelica model looks as follows (with
annotation statements removed):

.. literalinclude:: _build/mo/basic.mo
  :language: modelica
  :lineno-match:

The three water system elements (storage, inflow, and outfall) appear under
the ``model Example`` statement. The ``equation`` part connects these three
elements with the help of connections. Note that ``storage`` extends the partial
model ``QSISO`` which contains the connectors ``QIn`` and ``QOut``.
With ``QSISO``, ``storage`` can be connected on two sides. The ``storage``
element also has a variable ``Q_release``, which is the descision variable the
operator controls.

OpenModelica Connection Editor will automatically generate the element and
connector entries in the text text file. Defining inputs and outputs requires
editing the text file directly. Relationships between the inputs and outputs and
the library elements must also be defined in the ``equation`` section.

In addition to elements, the input variables ``Q_in`` and ``Q_release`` are also
defined. ``Q_in`` is determined by the forecast and the operator cannot contol
it, so we set ``Q_in(fixed = true)``. The actual values of ``Q_in`` are stored
in ``timeseries_input.csv``. In the ``equation`` section, equations are defined
to relate the inputs to the appropriate water system elements.

Because we want to view the water volume in the storage element in the output
file, we also define an output variable ``V_storage``.

The Optimization Problem
------------------------

The python script is created and edited in a text editor. In general, the python
script consists of the following blocks:

* Import of packages
* Definition of the optimization problem class

  * Constructor
  * Objective function
  * Definition of constraints
  * Any additional configuration

* A run statement

Importing Packages
''''''''''''''''''

Packages are imported using ``from ... import ...`` at the top of the file. In
our script, we import the classes we want the class to inherit, the
package ``run_optimization_problem`` form the ``rtctools.util`` package, and
any extra packages we want to use. For this example, the import block looks
like:

.. literalinclude:: ../examples/basic/src/example.py
  :language: python
  :lines: 1-5
  :lineno-match:

Optimization Problem
''''''''''''''''''''

The next step is to define the optimization problem class. We construct the
class by declaring the class and inheriting the desired parent classes. The
parent classes each perform different tasks realted to importing and exporting
data and solving the optimization problem. Each imported class makes a set of
methods available to the our optimization class.

.. literalinclude:: ../examples/basic/src/example.py
  :language: python
  :pyobject: Example
  :lineno-match:
  :end-before: """

The next, we define an objective function. This is a class method that returns
the value that needs to be minimized.

.. literalinclude:: ../examples/basic/src/example.py
  :language: python
  :pyobject: Example.objective
  :lineno-match:

Constraints can be declared by declairing the ``path_constraints`` method. Path
constraints are constraints that are applied every timestep. To set a constraint
at an individual timestep, we could define it inside the ``constraints`` method.

Other parent classes also declare this method, so we call the ``super()`` method
so that we don't overwrite their behaviour.

.. literalinclude:: ../examples/basic/src/example.py
  :language: python
  :pyobject: Example.path_constraints
  :lineno-match:

Run the Optimization Problem
''''''''''''''''''''''''''''

To make our script run, at the bottom of our file we just have to call
the ``run_optimization_problem()`` method we imported on the optimization
problem class we just created.

.. literalinclude:: ../examples/basic/src/example.py
  :language: python
  :lineno-match:
  :start-after: # Run

The Whole Script
''''''''''''''''

All together, the whole example script is as follows:

.. literalinclude:: ../examples/basic/src/example.py
  :language: python
  :lineno-match:

Running RTC-Tools
---------------------

TODO:

* An explaination of how to run RTC-Tools goes here.
* So does a description of the terminal output

Extracting Results
------------------

The results from the run are found in ``output/timeseries_export.csv``. Any
CSV-reading software can import it, but this is what the results look like when
plotted in Microsoft Excel:

.. image:: images/basic_example_resultplot.png

This plot shows that the operator is able to keep the water level within the
bounds over the entire time horizon and end with a full reservoir.

Feel free to experiment with this example. See what happens if you change the
max of ``Q_release`` (in the modelica file) or if you make the objective
function negative (in the python script).
