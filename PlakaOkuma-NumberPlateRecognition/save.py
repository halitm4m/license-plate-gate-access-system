import datetime
from pathlib import Path

DATA_FILE = Path(__file__).with_name("data.txt")

def write(message, print_to_terminal=True):
    time = datetime.datetime.now()
    line = message + " " + str(time)
    with open(DATA_FILE, "a") as f:
        f.write(line + "\n")
    if print_to_terminal:
        print(line, flush=True)
    return True
