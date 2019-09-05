from random import randint, seed


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

    Retun values:
      - Tuple with index of string containing the letter, and position within
        the string. In case the letter is not found, both values are -1.
    """
    if len(strings) == 0:
        return (-1, -1)
    pos = letter_in_string(strings[0], letter)
    if pos != -1:
        return (group_index, pos)
    else:
        return which_string(strings[1:], letter, group_index+1)


def random_tick_group(ticks, max_len):
    """
    Return a series of ticks of length 'max_len', starting on a random position.
    :param ticks: the dataframe of ticks to extract the random series from
    :param max_len: the maximum length of the series to be taken away
    :return: the dataframe of length max_len extracted
    """
    # seed(1971)
    start = randint(0, ticks.shape[0] - max_len - 1)
    end = start + max_len
    return ticks.iloc[start:end]
