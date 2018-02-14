model TestModel
	parameter Real x_start;
	Real x(start=x_start, fixed=true);
	Real w(start=0.0, fixed=true);
	Real alias;

	parameter Real k = 1.0;

	input Real u;
	output Real u_out;

	output Real y;

	output Real z;

	//TODO: Implement delayed variables and tests for them
	//input Real x_delayed;

	output Real switched;

	input Real constant_input;
	output Real constant_output;

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

	u_out = u + 1;

end TestModel;