from pymodelica.compiler import compile_fmu

# Compile the model for FMI v2.0 Co-Simulation (CS)
model_name = 'TestModel'
mo_files   = ['TestModel.mo']

compile_fmu(model_name,
	        mo_files,
	        version = 2.0,
	        target  = 'cs')
