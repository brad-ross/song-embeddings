from datetime import datetime

# TODO: use Python Loggger
def log(message):
    print('[{0}] {1}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message))