try:
    import json
except ImportError:
    import simplejson as json

from itertools import groupby

def bare_tag(elem):
    return elem.tag.rsplit('}', 1)[-1]
    
def to_pod(xml):
    properties = {}
    
    #text of an element goes in $
    if xml.text is not None:
        properties['$'] = xml.text
        
    #attributes are prefixed with @
    for (key, val) in xml.attrib.iteritems():
        properties['@' + key] = val
    
    #children are entries keyed by their element name
    # if multiple elements have the same name, they become an array
    sorted_children = sorted([(bare_tag(e), e) for e in xml])
    for key, group in groupby(sorted_children, key=lambda t: t[0]):
        grouped_elements = list(group)
        
        if len(grouped_elements) > 1:
            properties[key] = [to_pod(e) for k,e in grouped_elements]
        else:
            properties[key] = to_pod(grouped_elements[0][1])
    
    return properties

def to_json(xml,**kargs):
    return json.JSONEncoder(**kargs).encode({bare_tag(xml): to_pod(xml)})
