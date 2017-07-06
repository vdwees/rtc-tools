model PI_contoller
  input Real setpoint;
  input Real measured;
  parameter Real Kp = 1.0;
  parameter Real Ki = 0.0;
  Real error = setpoint - measured;
  Real integral;
  Real control = Kp * error + Ki * integral;
equation
  der(integral) = error;
initial equation
  integral = 0.0;
end PID_contoller;