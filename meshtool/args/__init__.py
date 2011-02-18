class FilterArgument(object):
    """Base class for arguments to filters"""
    def __init__(self, name, description):
        self.name = name
        self.description = description
    def __str__(self):
        return "<FilterArgument (name=%s, desc=%s)>" % (self.name, self.description)

class FileArgument(FilterArgument):
    """Argument that should be a file path"""
    def __init__(self, name, description):
        super(FileArgument, self).__init__(name, description)
    def __str__(self):
        return "<FileArgument (name=%s, desc=%s)>" % (self.name, self.description)