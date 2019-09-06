import errno
import os
from os.path import dirname, realpath, join
from pathlib import Path


def file_exists(given_filepath, my_dir):
    """
    Check if the file exists as specified in argument, or try to find
    it using the local path of the script
    :param given_filepath:
    :return: The path where the file is or None if it couldn't be found
    """
    if os.path.exists(given_filepath) is True:
        filepath = given_filepath
    else:
        new_filepath = os.path.join(my_dir, given_filepath)
        if os.path.exists(new_filepath) is True:
            filepath = new_filepath
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), new_filepath)
    return filepath


def valid_output_name(filename, path, extension=None):
    """
    Builds a valid name with the encoder metadata the date.
    Returns The filename if the name is valid and file does not exists,
            None otherwise.
    """
    path = file_exists(path, dirname(realpath(__file__)))
    if extension:
        base_filepath = join(path, filename) + '.{}'.format(extension)
    else:
        base_filepath = join(path, filename)
    output_filepath = base_filepath
    idx = 1
    while Path(output_filepath).is_file() is True:
        if extension:
            output_filepath = join(
                path, filename) + '_{:d}.{}'.format(
                    idx, extension)
        else:
            output_filepath = join(path, filename + '_{}'.format(idx))
        idx += 1
    return output_filepath