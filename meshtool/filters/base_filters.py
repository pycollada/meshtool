from meshtool.args import *

class FilterException(Exception):
    """Exception message thrown by a filter"""
    pass

class Filter(object):
    """Base class for a filter"""
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.arguments = []

class LoadFilter(Filter):
    """Base class for a filter that loads a mesh"""
    def __init__(self, name, description):
        super(LoadFilter, self).__init__(name, description)
        self.arguments.append(FileArgument("file", "Path of the file to load"))
    def apply(self):
        """Must be overriden by subclasses. Returns Collada instance"""
        raise NotImplementedError()

class OpFilter(Filter):
    """Base class for a filter that takes a Collada instance as input and output"""
    def __init__(self, name, description):
        super(OpFilter, self).__init__(name, description)
    def apply(self, collada):
        """Must be overriden by subclasses. Returns Collada instance"""
        raise NotImplementedError()

class SaveFilter(OpFilter):
    """Base class for a filter that saves a mesh"""    
    def __init__(self, name, description):
        super(SaveFilter, self).__init__(name, description)
        self.arguments.append(FileArgument("file", "Path where the file should be saved to"))

class FilterFactory(object):
    """Factor for registering and retrieving filters"""
    def __init__(self):
        self.registrar = {}
        #keeping a list of names to preserve ordering
        self.nameList = []
    def register(self, name, filter_generator):
        self.registrar[name] = filter_generator
        self.nameList.append(name)
    def getInstance(self, name):
        if name in self.registrar:
            return self.registrar[name]()
        else:
            return None
    def getFilterNames(self):
        return self.nameList
