import datetime
from pathlib import Path

UNAUTHORIZED_PLATES_FILE = Path(__file__).with_name("unauthorized_plates.txt")

def write(message, print_to_terminal=True):
    time = datetime.datetime.now()
    line = message + " " + str(time)
    with open(UNAUTHORIZED_PLATES_FILE, "a") as f:
        f.write(line + "\n")
    if print_to_terminal:
        print(line, flush=True)
    return True
