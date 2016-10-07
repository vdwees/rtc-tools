from subprocess import call
import getpass
import sys
import os

user = getpass.getuser()
srcdir = os.path.normpath(os.path.join(os.getcwd(), '..', '..'))

arguments = \
    ['docker', 'run', \
     # Environment
     '-e', 'USER="{}"'.format(user), \
     # Mount local filesysteem
     '-v', '{}:/usr/src'.format(srcdir.replace('\\', '/').replace('C:', '/c')), \
     # Working directory
     '-w', '/usr/src/src',
     # Run interactively
     '-i', \
     # Select image
     '-t', 'rtctools-modelling-hsl',
     # Run nose with JModelica.org Python
     '/opt/JModelica/bin/jm_python.sh', \
     'setup.py', \
     'build_ext', '--inplace']

call(arguments)

run_apidoc = False
if run_apidoc:
    arguments = \
        ['docker', 'run', \
         # Environment
         '-e', 'USER="{}"'.format(user), \
         # Mount local filesysteem
         '-v', '{}:/usr/src'.format(srcdir.replace('\\', '/').replace('C:', '/c')), \
         # Working directory
         '-w', '/usr/src/src',
         # Run interactively
         '-i', \
         # Select image
         '-t', 'rtctools-modelling-hsl',
         # Run nose with JModelica.org Python
         '/opt/JModelica/bin/jm_python.sh', \
         '-m', 'sphinx.apidoc', \
         '-f', \
         '-o', '/usr/src/documentation/api', \
         '/usr/src/src/rtctools']

    call(arguments)

for backend in ['html']:
    arguments = \
        ['docker', 'run', \
         # Environment
         '-e', 'USER="{}"'.format(user), \
         # Mount local filesysteem
         '-v', '{}:/usr/src'.format(srcdir.replace('\\', '/').replace('C:', '/c')), \
         # Working directory
         '-w', '/usr/src/src', \
         # Run interactively
         '-i', \
         # Select image
         '-t', 'rtctools-modelling-hsl',
         # Run nose with JModelica.org Python
         '/opt/JModelica/bin/jm_python.sh', \
         '-m', 'sphinx', \
         '-b', backend, \
         '/usr/src/documentation/api', '/usr/src/documentation/api/_build/' + backend]

    arguments.extend(sys.argv[1:])

    call(arguments)
