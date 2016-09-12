#!python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import click

from .instance import list_interfaces
from .utils import (CONTEXT_SETTINGS,
                    UNKNOWN_OPTIONS,
                    ExistingDirPath,
                    ExistingFilePath,
                    UnexistingFilePath,
                    RegularExpression,
                    PythonModule,
                    grouper)


# declare the CLI group
@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument('logdir', type=ExistingDirPath)
@click.option('-r', '--regex', type=RegularExpression(), default='*',
              help='Regular expression to be searched in each traceback.')
def search(logdir, regex):
    """Search for tracebacks content.

    Search for traceback inside a folder of nipype crash log files that match
    a given regular expression.

    Examples:\n
    nipype search nipype/wd/log -r '.*subject123.*'
    """
    from .crash_files import iter_tracebacks

    for file, trace in iter_tracebacks(logdir):
        if regex.search(trace):
            click.echo("-" * len(file))
            click.echo(file)
            click.echo("-" * len(file))
            click.echo(trace)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument('crashfile', type=ExistingFilePath,)
@click.option('-r', '--rerun', is_flag=True, flag_value=True,
              help='Rerun crashed node.')
@click.option('-d', '--debug', is_flag=True, flag_value=True,
              help='Enable Python debugger when re-executing.')
@click.option('-i', '--ipydebug', is_flag=True, flag_value=True,
              help='Enable IPython debugger when re-executing.')
@click.option('--dir', type=ExistingDirPath,
              help='Directory where to run the node in.')
def crash(crashfile, rerun, debug, ipydebug, directory):
    """Display Nipype crash files.

    For certain crash files, one can rerun a failed node in a temp directory.

    Examples:\n
    nipype crash crashfile.pklz\n
    nipype crash crashfile.pklz -r -i\n
    """
    from .crash_files import display_crash_file

    debug = 'ipython' if ipydebug else debug
    if debug == 'ipython':
        import sys
        from IPython.core import ultratb
        sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                             color_scheme='Linux',
                                             call_pdb=1)
    display_crash_file(crashfile, rerun, debug, directory)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument('pklz_file', type=ExistingFilePath)
def show(pklz_file):
    """Print the content of Nipype node .pklz file.

    Examples:\n
    nipype show node.pklz
    """
    from pprint import pprint
    from ..utils.filemanip import loadpkl

    pkl_data = loadpkl(pklz_file)
    pprint(pkl_data)


@cli.command(context_settings=UNKNOWN_OPTIONS)
@click.argument('module', type=PythonModule(), required=False)
@click.argument('interface', type=str, required=False)
@click.option('--list', is_flag=True, flag_value=True,
              help='List the available Interfaces inside the given module.')
@click.option('-h', '--help', is_flag=True, flag_value=True,
              help='Show help message and exit.')
@click.pass_context
def run(ctx, module, interface, list, help):
    """Run a Nipype Interface.

    Examples:\n
    nipype run nipype.interfaces.nipy --list\n
    nipype run nipype.interfaces.nipy ComputeMask --help
    """
    import argparse
    from .utils import add_args_options
    from ..utils.nipype_cmd import run_instance

    # print run command help if no arguments are given
    module_given = bool(module)
    if not module_given:
        click.echo(ctx.command.get_help(ctx))

    # print the list of available interfaces for the given module
    elif (module_given and list) or (module_given and not interface):
        iface_names = list_interfaces(module)
        click.echo('Available Interfaces:')
        for if_name in iface_names:
            click.echo('    {}'.format(if_name))

    # check the interface
    elif (module_given and interface):
        # create the argument parser
        description = "Run {}".format(interface)
        prog = " ".join([ctx.command_path,
                         module.__name__,
                         interface] + ctx.args)
        iface_parser = argparse.ArgumentParser(description=description,
                                               prog=prog)

        # instantiate the interface
        node = getattr(module, interface)()
        iface_parser = add_args_options(iface_parser, node)

        if not ctx.args:
            # print the interface help
            iface_parser.print_help()
        else:
            # run the interface
            args = iface_parser.parse_args(args=ctx.args)
            run_instance(node, args)


@cli.group()
def convert():
    """Export nipype interfaces to other formats."""
    pass


@convert.command(context_settings=CONTEXT_SETTINGS)
@click.option("-i", "--interface", type=str, required=True,
              help="Name of the Nipype interface to export.")
@click.option("-m", "--module", type=PythonModule(), required=True,
              help="Module where the interface is defined.")
@click.option("-o", "--output", type=UnexistingFilePath, required=True,
              help="JSON file name where the Boutiques descriptor will be written.")
@click.option("-t", "--ignored-template-inputs", type=str, multiple=True,
              help="Interface inputs ignored in path template creations.")
@click.option("-d", "--docker-image", type=str,
              help="Name of the Docker image where the Nipype interface is available.")
@click.option("-r", "--docker-index", type=str,
              help="Docker index where the Docker image is stored (e.g. http://index.docker.io).")
@click.option("-n", "--ignore-template-numbers", is_flag=True, flag_value=True,
              help="Ignore all numbers in path template creations.")
@click.option("-v", "--verbose", is_flag=True, flag_value=True,
              help="Enable verbose output.")
def boutiques(interface, module, output, ignored_template_inputs,
              docker_image, docker_index, ignore_template_numbers,
              verbose):
    """Nipype to Boutiques exporter.

    See Boutiques specification at https://github.com/boutiques/schema.
    """
    from nipype.utils.nipype2boutiques import generate_boutiques_descriptor

    # Generates JSON string
    json_string = generate_boutiques_descriptor(module,
                                                interface,
                                                ignored_template_inputs,
                                                docker_image,
                                                docker_index,
                                                verbose,
                                                ignore_template_numbers)

    # Writes JSON string to file
    with open(output, 'w') as f:
        f.write(json_string)
