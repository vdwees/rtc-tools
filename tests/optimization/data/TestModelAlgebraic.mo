model TestModelAlgebraic
	Real y;
	input Real u(fixed=false);

equation
	y + u = 1.0;

end TestModelAlgebraic;