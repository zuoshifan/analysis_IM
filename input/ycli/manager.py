"""This module contains the pipeline manager.

It is simply a driver for scripting together many pipeline modules.
"""

import os
import sys

from kiyopy import parse_ini

params_init = {
              # A list of modules that should be executed.  All entries of the
              # list should contain an executable accepting only parameter file
              # or dict.
              'modules' : [],
              'processes' : 1
              }


def execute(pipe_file_or_dict, feedback=0) :
    """Execute all the modules listed in the input file."""

    params, module_params = parse_ini.parse(pipe_file_or_dict, params_init, 
                                       prefix='pipe_',return_undeclared=True,
                                       feedback=feedback)
    
    for module in params['modules'] :
        # Module is either the python object that should be executed, or a
        # tuple, with the first element being the module and the second element
        # being a prefix replacement of the form ('p1_', 'p2_').  Before
        # executing the module, we rename all parameters begining with 'p1_'
        # to 'p2_'.
        if isinstance(module, tuple) :
            mod =  module[0]
            pars = dict(module_params)
            old_prefix = module[1][0]
            n = len(old_prefix)
            new_prefix = module[1][1]
            for key, value in module_params.iteritems() :
                if key[0:n] == old_prefix :
                    pars[new_prefix + key[n:]] = value
        else :
            mod = module
            pars = module_params
        if feedback > 1 :
            print 'Excuting analysis module: ' + str(mod)
        mod(pars, feedback=feedback).execute(params['processes'])

# If this file is run from the command line, execute the main function.
if __name__ == "__main__":
    import sys
    execute(str(sys.argv[1]))

