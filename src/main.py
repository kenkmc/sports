"""Clean minimal runner for the MVP app.

This file purposefully contains only a tiny startup check to verify the
workspace imports are working and to avoid pulling in the large GUI /
CV thread while we stabilize the repository.
"""

import sys
import os


def main():
    print('Minimal main: file is clean.')
    print('Python executable:', sys.executable)
    print('Working directory:', os.getcwd())


if __name__ == '__main__':
    main()
