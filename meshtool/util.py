import re
import unicodedata

try:
    import Image
except ImportError:
    from PIL import Image

try:
    import ImageDraw
except ImportError:
    from PIL import ImageDraw

try:
    import ImageOps
except ImportError:
    from PIL import ImageOps

try:
    import ImageFile
except ImportError:
    from PIL import ImageFile
# Following is a workaround for setting quality=95, optimize=1 when encoding JPEG
# Otherwise, an error is output when trying to save
# Taken from http://mail.python.org/pipermail/image-sig/1999-August/000816.html
# default is 64k, setting to 20MB to handle large textures
ImageFile.MAXBLOCK = 20 * 1024 * 1024

def to_unicode(s):
    try:
        return s.decode('utf8')
    except UnicodeDecodeError:
        return s.decode('latin-1')

_slugify_strip_re = re.compile(r'[^\w\s-]')
_slugify_hyphenate_re = re.compile(r'[-\s]+')
def slugify(value):
    """
    Normalizes string, removes non-alpha characters,
    and converts spaces to hyphens.
    
    From Django's "django/template/defaultfilters.py".
    """
    if not isinstance(value, unicode):
        value = unicode(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    value = unicode(_slugify_strip_re.sub('', value).strip())
    return _slugify_hyphenate_re.sub('-', value)
