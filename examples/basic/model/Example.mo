model Example
  Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Inflow inflow annotation(Placement(visible = true, transformation(origin = {-55, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.SimpleRouting.Storage.Storage storage annotation(Placement(visible = true, transformation(origin = {0, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Terminal outfall annotation(Placement(visible = true, transformation(origin = {55, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  input Modelica.SIunits.VolumeFlowRate Q_in(fixed = true);
  input Modelica.SIunits.VolumeFlowRate Q_release(fixed = false, min = 0.0, max = 6.5);
  output Modelica.SIunits.Volume V_storage;
equation
  connect(inflow.QOut, storage.QIn) annotation(Line(points = {{-47, 0}, {-10, 0}}));
  connect(storage.QOut, outfall.QIn) annotation(Line(points = {{8, 0}, {47, 0}}));
  storage.Q_release = Q_release;
  inflow.Q = Q_in;
  V_storage = storage.V;
  annotation(Diagram(coordinateSystem(extent = {{-148.5, -105}, {148.5, 105}}, initialScale = 0.1, grid = {5, 5})));
end Example;
