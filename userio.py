"User Interaction Helper Library"
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import input
from future import standard_library
standard_library.install_aliases()

def confirm(message_str):
    "Ask confirmation from the console"
    while True:
        response = input("{} (Y/n)".format(message_str))
        if response.lower() == 'y':
            return True
        elif response.lower() == 'n':
            return False
        print("Please type 'Y' or 'N'")

if __name__ == "__main__":
    print(confirm("test confirm"))
