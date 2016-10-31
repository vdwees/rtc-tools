model Example
  Deltares.ChannelFlow.Hydraulic.Storage.Linear storage(A=1.0e6, H_b=0.0, HQ.H(min=0.0, max=0.5)) annotation(Placement(visible = true, transformation(origin = {32.022, -0}, extent = {{-10, -10}, {10, 10}}, rotation = -270)));
  Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Discharge discharge annotation(Placement(visible = true, transformation(origin = {60, -0}, extent = {{10, -10}, {-10, 10}}, rotation = 270)));
  Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Level level annotation(Placement(visible = true, transformation(origin = {-52.7, 0}, extent = {{-10, -10}, {10, 10}}, rotation = -270)));
  Deltares.ChannelFlow.Hydraulic.Structures.Pump pump annotation(Placement(visible = true, transformation(origin = {0, -20}, extent = {{10, -10}, {-10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.Structures.Pump orifice annotation(Placement(visible = true, transformation(origin = {0, 20}, extent = {{10, -10}, {-10, 10}}, rotation = 0)));
  input Modelica.SIunits.VolumeFlowRate Q_in(fixed=true);
  input Modelica.SIunits.Position H_sea(fixed=true);
  input Modelica.SIunits.VolumeFlowRate Q_pump(fixed=false, min=0.0, max=7.0);
  input Modelica.SIunits.VolumeFlowRate Q_orifice(fixed=false, min=0.0, max=10.0);
  /*input Boolean is_downhill;*/
  output Modelica.SIunits.Position storage_level;
  output Modelica.SIunits.Position sea_level;
equation
  connect(orifice.HQDown, level.HQ) annotation(Line(visible = true, origin = {-33.025, 10}, points = {{25.025, 10}, {-6.675, 10}, {-6.675, -10}, {-11.675, -10}}, color = {0, 0, 255}));
  connect(storage.HQ, orifice.HQUp) annotation(Line(visible = true, origin = {43.737, 48.702}, points = {{-3.715, -48.702}, {-3.737, -28.702}, {-35.737, -28.702}}, color = {0, 0, 255}));
  connect(storage.HQ, pump.HQUp) annotation(Line(visible = true, origin = {4.669, -31.115}, points = {{35.353, 31.115}, {35.331, 11.115}, {3.331, 11.115}}, color = {0, 0, 255}));
  connect(discharge.HQ, storage.HQ) annotation(Line(visible = true, origin = {46.011, -0}, points = {{5.989, 0}, {-5.989, -0}}, color = {0, 0, 255}));
  connect(pump.HQDown, level.HQ) annotation(Line(visible = true, origin = {-33.025, -10}, points = {{25.025, -10}, {-6.675, -10}, {-6.675, 10}, {-11.675, 10}}, color = {0, 0, 255}));
  discharge.Q = Q_in;
  level.H = H_sea;
  pump.Q = Q_pump;
  orifice.Q = Q_orifice;
  storage_level = storage.HQ.H;
  sea_level = H_sea;
  annotation(Diagram(coordinateSystem(extent = {{-148.5, -105}, {148.5, 105}}, preserveAspectRatio = true, initialScale = 0.1, grid = {5, 5})));
end Example;
