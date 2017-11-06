from datetime import datetime

# TODO: use Python Loggger
def log(message):
    print('[{0}] {1}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message))

def flatten(iterable):
    return [j for i in iterable for j in i]

# Borrowed shamelessly from here:
# https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression
def union_maps(maps):
    result = {}
    for dictionary in maps:
        result.update(dictionary)
    return result