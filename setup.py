def pathSetup():
	from pathlib import Path
	import sys
	PROJ_NAME = "optimal-stopping"

	# Locate the Project directory
	curr_dir = str(Path.cwd())
	start = curr_dir.find(PROJ_NAME)
	if start < 0:
		print("ERROR: Project directory not found")
		print("Make sure you have the correct project structure")
		print("and run the simulation from within the project")
	proj_pathname = curr_dir[:(start+len(PROJ_NAME))]

	# Create path to the project directory
	proj_path = Path(proj_pathname)

	# Add the project folder to PATH
	sys.path.append(proj_pathname)