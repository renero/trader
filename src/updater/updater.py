import sys

from u_dictionary import UDictionary
from update import Update as update


def main(argv):
    params = UDictionary(args=argv)
    update(params, params.action)


if __name__ == "__main__":
    main(sys.argv)
