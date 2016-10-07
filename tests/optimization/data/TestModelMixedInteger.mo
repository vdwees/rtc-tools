model TestModelMixedInteger
	input Boolean choice (fixed=false);
	Boolean other_choice;
	Real y;

equation
	if choice then
		y = 1.0;
		other_choice = false;
	else
		y = -1.0;
		other_choice = true;
	end if;

end TestModelMixedInteger;