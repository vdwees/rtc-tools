model TestModelWithInitial
	Real x;
	Real w(start=0.0, fixed=true);
	Real alias;

	parameter Real k = 1.0;

	parameter Real u_max;
	input Real u(fixed=false, min = -2, max = u_max);

	output Real y;

	output Real z;

	input Real x_delayed(fixed=false);

	output Real switched;

	input Real constant_input(fixed=true);
	output Real constant_output;

initial equation
	x = 1.1;

equation
	der(x) = k * x + u;
	der(w) = x;

	alias = x;

	y + x = 3.0;

	z = alias^2 + sin(time);

	if x > 0.5 then
		switched = 1.0;
	else
		switched = 2.0;
	end if;

	constant_output = constant_input;

end TestModelWithInitial;