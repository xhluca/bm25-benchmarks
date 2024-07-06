import os
import fnmatch
import requests
from kaggle.api.kaggle_api_extended import KaggleApi

def kernels_output(self: KaggleApi, kernel: str, path: str, force=False, quiet=True, allowed_patterns=None):
    """ retrieve output for a specified kernel
            Parameters
        ==========
        kernel: the kernel to output
        path: the path to pull files to on the filesystem
        force: if output already exists, force overwrite (default False)
        quiet: suppress verbosity (default is True)
    """
    if kernel is None:
        raise ValueError('A kernel must be specified')
    if '/' in kernel:
        self.validate_kernel_string(kernel)
        kernel_url_list = kernel.split('/')
        owner_slug = kernel_url_list[0]
        kernel_slug = kernel_url_list[1]
    else:
        owner_slug = self.get_config_value(self.CONFIG_NAME_USER)
        kernel_slug = kernel

    if path is None:
        target_dir = self.get_default_download_dir('kernels', owner_slug,
                                                    kernel_slug, 'output')
    else:
        target_dir = path

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if not os.path.isdir(target_dir):
        raise ValueError(
            'You must specify a directory for the kernels output')

    response = self.process_response(
        self.kernel_output_with_http_info(owner_slug, kernel_slug))
    outfiles = []
    
    for item in response['files']:
        if allowed_patterns is not None:
            # check if the file matches any of the patterns, following bash-style pattern matching
            # https://docs.python.org/3/library/fnmatch.html
            match = any(fnmatch.fnmatch(item['fileName'], pattern) for pattern in allowed_patterns)
            if not match:
                continue
        
        outfile = os.path.join(target_dir, item['fileName'])
        outfiles.append(outfile)
        download_response = requests.get(item['url'])
        if force or self.download_needed(item, outfile, quiet):
            os.makedirs(os.path.split(outfile)[0], exist_ok=True)
            with open(outfile, 'wb') as out:
                out.write(download_response.content)
            if not quiet:
                print('Output file downloaded to %s' % outfile)

    log = response['log']
    if log:
        outfile = os.path.join(target_dir, kernel_slug + '.log')
        outfiles.append(outfile)
        with open(outfile, 'w') as out:
            out.write(log)
        if not quiet:
            print('Kernel log downloaded to %s ' % outfile)

    return outfiles