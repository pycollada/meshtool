import sys
import argparse
from collections import defaultdict
import meshtool.filters as filters
from meshtool.filters.base_filters import FilterException, OpFilter, LoadFilter
import collada

def usage_exit(parser, s):
    parser.print_usage()
    sys.exit("meshtool: error: " + s)

class CustomAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not 'ordered_args' in namespace:
            setattr(namespace, 'ordered_args', [])
        previous = namespace.ordered_args
        previous.append((self.dest, values))
        setattr(namespace, 'ordered_args', previous)

class CustomFormatter(argparse.HelpFormatter):
    def add_arguments(self, actions):
        # gets rid of "optional arguments" prefix
        self.end_section()
        
        action_list = defaultdict(list)
        
        for action in actions:
            if not isinstance(action, CustomAction):
                continue
            filter_name = action.dest
            inst = filters.factory.getInstance(filter_name)
            
            action_list[inst.CATEGORY].append(action)
        
        order = ['Loading',
                 'Printing',
                 'Simplification',
                 'Optimizations',
                 'Meta',
                 'Operations',
                 'Saving']
        
        for section_name in order:
            loaders = action_list[section_name]
            self.start_section(section_name)
            for action in loaders:
                self.add_argument(action)
            self.end_section()
            
        self.start_section('')

def main():
    parser = argparse.ArgumentParser(
        description='Tool for manipulating mesh data using pycollada.',
        formatter_class=CustomFormatter,
        usage='meshtool --load_filter [--operation] [--save_filter]')
    for filter_name in filters.factory.getFilterNames():
        inst = filters.factory.getInstance(filter_name)
        parser.add_argument('--' + filter_name, required=False,
                            nargs=len(inst.arguments), help=inst.description,
                            metavar=tuple([arg.name for arg in inst.arguments]), action=CustomAction)
        
    args = parser.parse_args()
    
    if not 'ordered_args' in args:
        usage_exit(parser, "no arguments given")
    
    ordered_args = args.ordered_args

    (load_filter_name, load_filter_args) = ordered_args.pop(0)
    load_filter_inst = filters.factory.getInstance(load_filter_name)
    if load_filter_inst is None or not isinstance(load_filter_inst, LoadFilter):
        usage_exit(parser, "first argument must be a load filter")
    try:
        collada_inst = load_filter_inst.apply(*load_filter_args)
    except FilterException, e:
        sys.exit("Error: (argument %d) '%s': %s" % (1,load_filter_name,str(e)))
    if not isinstance(collada_inst, collada.Collada):
        sys.exit("Error: got an incorrect return value from filter (argument %d) '%s' " % (1, load_filter_name))

    for i, arg in enumerate(ordered_args):
        filter_name = arg[0]
        arguments = arg[1]
        inst = filters.factory.getInstance(filter_name)
        if inst is None or not isinstance(inst, OpFilter):
            usage_exit(parser, "specified filter (argument %d:'%s') is not an operation filter" % (i+1, filter_name))
        try:
            collada_inst = inst.apply(collada_inst, *arguments)
        except FilterException, e:
            sys.exit("Error: (argument %d) '%s': %s" % (i+1, filter_name, str(e)))
        if not isinstance(collada_inst, collada.Collada):
            sys.exit("Error: got an incorrect return value from filter (argument %d) '%s' " % (i+1, filter_name))

if __name__ == "__main__":
    main()
