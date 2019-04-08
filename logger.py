import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger("yourlogfilename.txt")
print("Hello world !") # this is should be saved in yourlogfilename.txt
print(sys.getdefaultencoding())
a = '你是我的人'
print(a)
