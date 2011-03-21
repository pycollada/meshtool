import sys
import argparse
import meshtool.filters as filters
import collada

def usage_exit(parser, str):
    parser.print_usage()
    sys.exit("meshtool: error: " + str)

class CustomAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not 'ordered_args' in namespace:
            setattr(namespace, 'ordered_args', [])
        previous = namespace.ordered_args
        previous.append((self.dest, values))
        setattr(namespace, 'ordered_args', previous)

def main():
    parser = argparse.ArgumentParser(
        description='Mesh tool with various operations that can be performed.')    
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
    if load_filter_inst is None or not isinstance(load_filter_inst, filters.LoadFilter):
        usage_exit(parser, "first argument must be a load filter")
    try:
        collada_inst = load_filter_inst.apply(*load_filter_args)
    except filters.FilterException, e:
        sys.exit("Error: (argument %d) '%s': %s" % (1,load_filter_name,str(e)))
    if not isinstance(collada_inst, collada.Collada):
        sys.exit("Error: got an incorrect return value from filter (argument %d) '%s' " % (1, load_filter_name))

    for i, arg in enumerate(ordered_args):
        filter = arg[0]
        arguments = arg[1]
        inst = filters.factory.getInstance(filter)
        if inst is None or not isinstance(inst, filters.OpFilter):
            usage_exit(parser, "specified filter (argument %d:'%s') is not an operation filter" % (i+1, filter))
        try:
            collada_inst = inst.apply(collada_inst, *arguments)
        except filters.FilterException, e:
            sys.exit("Error: (argument %d) '%s': %s" % (i+1,filter,str(e)))
        if not isinstance(collada_inst, collada.Collada):
            sys.exit("Error: got an incorrect return value from filter (argument %d) '%s' " % (i+1, filter))

if __name__ == "__main__":
    main()
