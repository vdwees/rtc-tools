model TestModelMixedInteger
	input Boolean choice(fixed=false);
	Boolean other_choice;
	Real y;
equation
	y = choice + (choice - 1);
	other_choice = 1 - choice;
end TestModelMixedInteger;