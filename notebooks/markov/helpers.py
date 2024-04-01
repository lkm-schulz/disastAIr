import sys

def print_over(text: str):
    move_and_clear_line()
    print(text)


def move_and_clear_line(lines=1):
    for _ in range(lines):
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')