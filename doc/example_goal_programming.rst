Defining Multiple Objectives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    This example focuses on how to implement multiobjective optimization in
    RTC-Tools using Goal Programming. It assumes basic exposure to
    RTC-Tools. If you are a first-time user of RTC-Tools, see
    :doc:`example_basic`.

Goal programming is a way to satisfy (sometimes conflicting) goals by ranking
the goals by priority. The optimization algorithm will attempt to optimize each
goal one at a time, starting with the goal with the highest priority and moving
down through the list. Even if a goal cannot be satisfied, the goal programming
algorithm will move on when it has found the best possible answer. Goals can be
roughly divided into two types:

* As long as we satisfy the goal, we do not care by how much. If we cannot
  satisfy a goal, any lower priority goals are not allowed to increase the
  amount by which we exceed (which is equivalent to not allowing any change at
  all to the exceedance).
* We try to achieve as low a value as possible. Any lower priority goals are not
  allowed to result in an increase of this value (which is equivalent to not
  allowing any change at all).

In this example, we will be specifying two goals, on for each type. The higher
priority goal will be to maintian the water level of the storage element between
two levels. The lower priority goal will be to minimize the total volume pumped.

The Model
---------

.. note::

    This example uses the same hydraulic model as the MILP example. For a
    detalied explaination of the hydraulic model, including how to to formulate
    mixed integers in your model, see :doc:`example_mixed_integer`.

For this example, the model represents a typical setup for the dewatering of
lowland areas. Water is routed from the hinterland (modeled as discharge
boundary condition, right side) through a canal (modeled as storage element)
towards the sea (modeled as water level boundary condition on the left side).
Keeping the lowland area dry requires that enough water is discharged  to the
sea. If the sea water level is lower than the water level in the canal, the
water can be discharged to the sea via gradient flow through the orifice (or a
weir). If the sea water level is higher than in the canal, water must be pumped.

In OpenModelica Connection Editor, the model looks like this:

.. image:: images/orifice_vs_pump_openmodelica.png

In text mode, the Modelica model looks as follows (with annotation statements
removed):

.. literalinclude:: _build/mo/goal_programming.mo
  :language: modelica
  :lineno-match:

The Optimization Problem
------------------------

When using goal programming, the python script consists of the following blocks:

* Import of packages
* Declaration of Goals
* Declaration of the optimization problem class

  * Constructor
  * Declaration of constraint methods
  * Specification of Goals
  * Declaration of a ``priority_completed()`` method
  * Declaration of a ``pre()`` method
  * Declaration of a ``post()`` method
  * Additional configuration of the solver

* A run statement

Importing Packages
''''''''''''''''''

For this example, the import block is as follows:

.. literalinclude:: ../examples/goal_programming/src/example.py
  :language: python
  :lines: 1-8
  :lineno-match:

Declaring Goals
'''''''''''''''

Goals are defined as classes that inherit the ``Goal`` parent class. The
components of goals can be found in :doc:`multi_objective`.

First, we have a high priority goal to keep the water level within a minimum and
maximum:

.. literalinclude:: ../examples/goal_programming/src/example.py
  :language: python
  :pyobject: WaterLevelRangeGoal
  :lineno-match:

We also want to save energy, so we define a goal to minimize ``Q_pump``. This
goal has a lower priority.

.. literalinclude:: ../examples/goal_programming/src/example.py
  :language: python
  :pyobject: MinimizeQpumpGoal
  :lineno-match:

Optimization Problem
''''''''''''''''''''

Next, we construct the class by declaring it and inheriting the desired parent
classes.

.. literalinclude:: ../examples/goal_programming/src/example.py
  :language: python
  :pyobject: Example
  :lineno-match:
  :end-before: """

Constraints can be declared by declairing the ``path_constraints()`` method. Path
constraints are constraints that are applied every timestep. To set a constraint
at an individual timestep, define it inside the ``constraints()`` method.

Other parent classes also declare this method, so we call the ``super()`` method
so that we don't overwrite their behaviour.

.. literalinclude:: ../examples/goal_programming/src/example.py
  :language: python
  :pyobject: Example.path_constraints
  :lineno-match:

Now we pass in the goals. We want to apply our goals to every timestep, so we
use the ``path_goals()`` method. This is a method that returns a list of the goals
we defined above.

.. literalinclude:: ../examples/goal_programming/src/example.py
  :language: python
  :pyobject: Example.path_goals
  :lineno-match:

If all we cared about were the results, we could end our class declaration here.
However, it is usually helpful to track how the solution changes after
optimizing each priority level. To track these changes, we need to add three
methods.

The method ``pre()`` is already defined in RTC-Tools, but we would like to add
a line to it to create a variable for storing intermediate results. To do this,
we declare a new ``pre()`` method, call ``super(Example, self).pre()`` to ensure
that the original method runs unmodified, and add in a variable declaration to
store our list of intermediate results:

.. literalinclude:: ../examples/goal_programming/src/example.py
  :language: python
  :pyobject: Example.pre
  :lineno-match:

Next, we define the ``priority_completed()`` method to inspect and summerize the
results. These are appended to our intermediate results variable after each
priority is completed.

.. literalinclude:: ../examples/goal_programming/src/example.py
  :language: python
  :pyobject: Example.priority_completed
  :lineno-match:

We want some way to output our intermediate results. This is accomplished using
the ``post()`` method. Again, we nedd to call the ``super()`` method to avoid
overwiting the internal method.

.. literalinclude:: ../examples/goal_programming/src/example.py
  :language: python
  :pyobject: Example.post
  :lineno-match:

Finally, we want to apply some additional configuration, reducing the amount of
information the solver outputs:

.. literalinclude:: ../examples/goal_programming/src/example.py
  :language: python
  :pyobject: Example.solver_options
  :lineno-match:

Run the Optimization Problem
''''''''''''''''''''''''''''

To make our script run, at the bottom of our file we just have to call
the ``run_optimization_problem()`` method we imported on the optimization
problem class we just created.

.. literalinclude:: ../examples/goal_programming/src/example.py
  :language: python
  :lineno-match:
  :start-after: # Run

The Whole Script
''''''''''''''''

All together, the whole example script is as follows:

.. literalinclude:: ../examples/goal_programming/src/example.py
  :language: python
  :lineno-match:

Running the Optimization Problem
--------------------------------

Following the execution of the optimization problem, the  ``post()`` method
should print out the following lines::

    After finishing goals of priority 1:
    Level goal satisfied at 19 of 21 time steps
    Integral of Q_pump = 73.93

    After finishing goals of priority 2:
    Level goal satisfied at 19 of 21 time steps
    Integral of Q_pump = 52.22

As the output indicates, while optimizing for the priority 1 goal, no attempt
was made to minimize the integral of ``Q_pump``. The only objective was to
minimize the number of states in violation of the water level goal.

After optimizing for the priority 2 goal, the solver was able to find a solution
that reduced the integral of ``Q_pump`` without increasing the number of
timesteps where the water level exceded the limit.

Extracting Results
------------------

The results from the run are found in ``output/timeseries_export.csv``. Any
CSV-reading software can import it, but this is what the results look like when
plotted in Microsoft Excel:

.. note::

    TODO: Plot these results

.. image:: images/goal_example_resultplot.png
