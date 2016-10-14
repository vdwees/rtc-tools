Basics
======

.. autoclass:: rtctools.optimization.timeseries.Timeseries
    :members: __init__, times, values
    :show-inheritance:

.. autoclass:: rtctools.optimization.optimization_problem.OptimizationProblem
    :members: bounds, constant_inputs, constraints, control, control_at, delayed_feedback, der, der_at, ensemble_member_probability, ensemble_size, get_timeseries, history, initial_time, integral, interpolate, lookup_tables, objective, optimize, parameters, path_constraints, post, pre, seed, set_timeseries, solver_options, state, state_at, states_in, timeseries_at
    :show-inheritance:

.. autofunction:: rtctools.util.run_optimization_problem