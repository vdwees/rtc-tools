model Example
  // Elements
  Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Inflow inflow(Q = Q_in) annotation(Placement(visible = true, transformation(origin = {-55, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.SimpleRouting.Storage.Storage storage(Q_release = -PI.control, V(nominal=4e5)) annotation(Placement(visible = true, transformation(origin = {0, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Terminal outfall annotation(Placement(visible = true, transformation(origin = {55, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  PI_contoller PI(measured = storage.V, setpoint = storage_V_SP, Kp = 1.0, Ki = 0.0);
  // Inputs
  input Modelica.SIunits.VolumeFlowRate Q_in(fixed = true);
  input Modelica.SIunits.VolumeFlowRate storage_V_SP(fixed = true);
  input Modelica.SIunits.Volume V_storage_init(fixed = true);
  // Outputs
  output Modelica.SIunits.Volume V_storage = storage.V;
  output Modelica.SIunits.VolumeFlowRate Q_release = -PI.control;
equation
  connect(inflow.QOut, storage.QIn) annotation(Line(points = {{-47, 0}, {-10, 0}}));
  connect(storage.QOut, outfall.QIn) annotation(Line(points = {{8, 0}, {47, 0}}));
initial equation
  V_storage_init = storage.V;
  annotation(Diagram(coordinateSystem(extent = {{-148.5, -105}, {148.5, 105}}, initialScale = 0.1, grid = {5, 5})));
end Example;