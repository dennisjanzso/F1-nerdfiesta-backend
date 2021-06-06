import subprocess
import random
import string

def getTicket():
     return str(''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)))

def removeFile(filename):
    subprocess.run(["rm", "cache/" + filename + ".png"], capture_output=False)

