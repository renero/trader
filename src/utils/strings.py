import numpy as np
from termcolor import colored


def letter_in_string(string, letter):
    """Given a string, determine if that string contains the letter.
    Parameters:
      - string: a sequence of letters
      - letter: a string character.

    Return values:
      - index position if found,
      - -1 if ValueError exception is raised.
    """
    try:
        return string.index(letter)
    except ValueError:
        return -1


def which_string(strings, letter, group_index=0):
    """Return the string and position within that string where the letter
    passed as argument is found.

    Arguments:
      - strings: an array of strings.
      - letter: a single character to be found in strings

    Return values:
      - Tuple with index of string containing the letter, and position within
        the string. In case the letter is not found, both values are -1.
    """
    if len(strings) == 0:
        return -1, -1
    pos = letter_in_string(strings[0], letter)
    if pos != -1:
        return group_index, pos
    else:
        return which_string(strings[1:], letter, group_index + 1)


def previous(objects_array: object, pos: int):
    """
    Return the object at pos - 1 position, only if pos != 0, otherwise
    returns None
    """
    if pos == 0:
        return None
    else:
        return objects_array[pos - 1]


def print_bin_predictions_match(y_test, yhat):
    for i in range(y_test.shape[0]):
        ix = colored(str(f'{i:02d} |'), 'grey')
        sep = colored('|', 'grey')
        y = int(y_test[i][0])
        pred = f"{yhat[i][0]:.02f}"
        true_pred = (yhat[i][0] >= 0.5 and y == 1) or (
                yhat[i][0] < 0.5 and y == 0)
        color = "green" if true_pred else "red"
        pred = colored(pred, color)
        if i % 9 == 0:
            print(f"\n{ix} ", end='')
        print(f"{y} {sep} {pred} {sep} ", end="")
    print(colored("\n", "white"))


def print_progbar(percent: float, max: int = 20, do_print=True,
                  **kwargs: str) -> str:
    """ Prints a progress bar of max characters, with progress up to
    the passed percentage

    :param percent: the percentage of the progress bar completed
    :param max: the max width of the progress bar
    :param do_print: print the progbar or not
    :param **kwargs: optional arguments to the `print` method.

    Example
    -------

    >>> print_progbar(0.65)
    >>> "[=============·······]"

    """
    done = int(np.floor(percent * 20))
    remain = max - done
    pb = "[" + "=" * done + "·" * remain + "]"
    if do_print is True:
        print(pb, sep="", **kwargs)
    else:
        return pb

