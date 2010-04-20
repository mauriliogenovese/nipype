"""Miscellaneous file manipulation functions

"""
import os
import re
import shutil
from glob import glob
import logging
# The md5 module is deprecated in Python 2.6, but hashlib is only
# available as an external package for versions of python before 2.6.
# Both md5 algorithms appear to return the same result.
try:
    from hashlib import md5
except ImportError:
    from md5 import md5

try:
    # json included in Python 2.6
    import json
except ImportError:
    # simplejson is the json module that was included in 2.6 (I
    # believe).  Used here for Python 2.5
    import simplejson as json

import numpy as np

from nipype.utils.misc import is_container

fmlogger = logging.getLogger("filemanip")


class FileNotFoundError(Exception):
    pass

def split_filename(fname):
    """Split a filename into parts: path, base filename and extension.

    Parameters
    ----------
    fname : str
        file or path name

    Returns
    -------
    pth : str
        base path from fname
    fname : str
        filename from fname, without extension
    ext : str
        file extenstion from fname

    Examples
    --------
    >>> from nipype.utils.filemanip import split_filename
    >>> pth, fname, ext = split_filename('/home/data/subject.nii.gz')
    >>> pth
    '/home/data'

    >>> fname
    'subject'

    >>> ext
    '.nii.gz'

    """

    pth, fname = os.path.split(fname)
    tmp = '.none'
    ext = []
    while tmp:
        fname, tmp = os.path.splitext(fname)
        ext.append(tmp)
    ext.reverse()
    return pth, fname, ''.join(ext)

def fname_presuffix(fname, prefix='', suffix='', newpath=None, use_ext=True):
    """Manipulates path and name of input filename

    Parameters
    ----------
    fname : string
        A filename (may or may not include path
    prefix : string
        Characters to prepend to the filename
    suffix : string
        Characters to append to the filename
    newpath : string
        Path to replace the path of the input fname
    use_ext : boolean
        If True (default), appends the extension of the original file
        to the output name.

    Returns
    -------
    Absolute path of the modified filename

    >>> from nipype.utils.filemanip import fname_presuffix
    >>> fname = 'foo.nii.gz'
    >>> fname_presuffix(fname,'pre','post','/tmp')
    '/tmp/prefoopost.nii.gz'

    """
    pth, fname, ext = split_filename(fname)
    if not use_ext:
        ext = ''
    if newpath:
        pth = os.path.abspath(newpath)
    return os.path.join(pth, prefix+fname+suffix+ext)


def fnames_presuffix(fnames, prefix='', suffix='', newpath=None,use_ext=True):
    """Calls fname_presuffix for a list of files.
    """
    f2 = []
    for fname in fnames:
        f2.append(fname_presuffix(fname, prefix, suffix, newpath, use_ext))
    return f2

def hash_rename(filename, hash):
    """renames a file given original filename and hash
    and sets path to output_directory
    """
    path, name, ext = split_filename(filename)
    newfilename = ''.join((name,'_0x', hash, ext))
    return os.path.join(path, newfilename)
             

def check_forhash(filename):
    """checks if file has a hash in its filename"""
    if isinstance(filename,list):
        filename = filename[0]
    path, name = os.path.split(filename)
    if re.search('(_0x[a-z0-9]{32})', name):
        hash = re.findall('(_0x[a-z0-9]{32})', name)
        return True, hash
    else:
        return False, None


def hash_infile(afile, chunk_len=8192):
    """ Computes md5 hash of a file"""
    md5hex = None
    if os.path.isfile(afile):
        md5obj = md5()
        fp = file(afile, 'rb')
        while True:
            data = fp.read(chunk_len)
            if not data:
                break
            md5obj.update(data)
        fp.close()
        md5hex = md5obj.hexdigest()
    return md5hex

def hash_timestamp(afile):
    """ Computes md5 hash of the timestamp of a file """
    md5hex = None
    if os.path.isfile(afile):
        md5obj = md5()
        stat = os.stat(afile)
        md5obj.update(str(stat.st_size))
        md5obj.update(str(stat.st_ctime))
        md5obj.update(str(stat.st_mtime))
        md5hex = md5obj.hexdigest()
    return md5hex

def copyfile(originalfile, newfile, copy=False):
    """Copy or symlink ``originalfile`` to ``newfile``.

    Parameters
    ----------
    originalfile : str
        full path to original file
    newfile : str
        full path to new file
    copy : Bool
        specifies whether to copy or symlink files
        (default=False) but only for posix systems
         
    Returns
    -------
    None
    
    """
    if os.path.lexists(newfile):
        fmlogger.warn("File: %s already exists, overwriting with %s, copy:%d" \
            % (newfile, originalfile, copy))
    if os.name is 'posix' and not copy:
        if os.path.lexists(newfile):
            os.unlink(newfile)
        os.symlink(originalfile,newfile)
    else:
        try:
            shutil.copyfile(originalfile, newfile)
        except shutil.Error, e:
            fmlogger.warn(e.message)
        
    if originalfile.endswith(".img"):
        hdrofile = originalfile[:-4] + ".hdr"
        hdrnfile = newfile[:-4] + ".hdr"
        matofile = originalfile[:-4] + ".mat"
        if os.path.exists(matofile):
            matnfile = newfile[:-4] + ".mat"
            copyfile(matofile, matnfile, copy)
        copyfile(hdrofile, hdrnfile, copy)

def copyfiles(filelist, dest, copy=False):
    """Copy or symlink files in ``filelist`` to ``dest`` directory.

    Parameters
    ----------
    filelist : list
        List of files to copy.
    dest : path/files
        full path to destination. If it is a list of length greater
        than 1, then it assumes that these are the names of the new
        files. 
    copy : Bool
        specifies whether to copy or symlink files
        (default=False) but only for posix systems
         
    Returns
    -------
    None
    
    """
    outfiles = filename_to_list(dest)
    newfiles = []
    for i,f in enumerate(filename_to_list(filelist)):
        if len(outfiles) > 1:
            destfile = outfiles[i]
        else:
            destfile = fname_presuffix(f, newpath=outfiles[0])
        copyfile(f,destfile,copy)
        newfiles.insert(i,destfile)
    return newfiles

def filename_to_list(filename):
    """Returns a list given either a string or a list
    """
    if isinstance(filename,(str, unicode)):
        return [filename]
    elif isinstance(filename,list):
        return filename
    elif is_container(filename):
        return [x for x in filename]
    else:
        return None

def list_to_filename(filelist):
    """Returns a list if filelist is a list of length greater than 1, 
       otherwise returns the first element
    """
    if len(filelist) > 1:
        return filelist
    else:
        return filelist[0]

def cleandir(dir):
    """Cleans all nifti, img/hdr, txt and matfiles from dir"""
    filetypes = ['*.nii','*.nii.gz','*.txt','*.img','*.hdr','*.mat','*.json']
    for ftype in filetypes:
        for f in glob(os.path.join(dir,ftype)):
            os.remove(f)
        
def save_json(filename, data):
    """Save data to a json file
    
    Parameters
    ----------
    filename : str
        Filename to save data in.
    data : dict
        Dictionary to save in json file.

    """

    fp = file(filename, 'w')
    json.dump(data, fp, sort_keys=True, indent=4)
    fp.close()

def debuglog(inputlines,filename='/tmp/dbginputs.txt'):
    fp=open(filename,'at')
    fp.writelines(inputlines)
    fp.close()

def load_json(filename):
    """Load data from a json file

    Parameters
    ----------
    filename : str
        Filename to load data from.

    Returns
    -------
    data : dict
   
    """

    fp = file(filename, 'r')
    data = json.load(fp)
    fp.close()
    return data

def loadflat(infile, *args):
    """Load an npz file into a dict
    """
    data = np.load(infile)
    out = {}
    if args:
        outargs = np.setdiff1d(args,data.files)
        if outargs:
            raise IOError('File does not contain variables: '+str(outargs))
    for k in data.files:
        if k in args or not args:
            out[k] = [f for f in data[k].flat]
            if len(out[k])==1:
                out[k] = out[k].pop()
    return out
