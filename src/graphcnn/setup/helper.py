from graphcnn.helper import *
import time
import os

_default_location = 'datasets/'

def get_file_location(file):
    return _default_location + file

def locate_or_download_file(file, url):
    if os.path.isfile(_default_location + file) == False:
        print_ext('Downloading "%s", this might take a few minutes' % url)
        verify_dir_exists(os.path.dirname(_default_location + file) + '/')
        try:
            from urllib import urlretrieve
        except ImportError:
            # Support python 3
            from urllib.request import urlretrieve
        urlretrieve (url, _default_location + file)
        return file
        
def locate_or_extract_file(file, folder):
    if os.path.isdir(_default_location + folder) == False and os.path.isfile(_default_location + folder) == False:
        print_ext('Extracting "%s", this might take a few minutes' % file)
        if os.path.splitext(_default_location + file)[1] in ['.tar', '.tgz']:
            import tarfile
            tar = tarfile.open(_default_location + file)
            tar.extractall(os.path.dirname(_default_location + file))
            tar.close()
        elif os.path.splitext(_default_location + file)[1] == '.zip':
            import zipfile
            zip_ref = zipfile.ZipFile(_default_location + file, 'r')
            zip_ref.extractall(os.path.dirname(_default_location + file))
            zip_ref.close()
        else:
            print_ext('Cannot extract: Invalid extension name')
            
            