"User Interaction Helper Library"

def confirm(message_str):
    "Ask confirmation from the console"
    while True:
        response = input("{} (Y/n)".format(message_str))
        if response.lower() == 'y':
            return True
        elif response.lower() == 'n':
            return False
        print("Please type 'Y' or 'N'")
