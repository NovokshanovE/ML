from base_1 import *
from advenced_1_v2 import *
import time
def main():
    part = input()
    t = time.time()
    if(part=='base'):
        uniform()
        chebishev()
        uniform_chebishev()
    elif (part == 'tsk1'):
        tsk1_2(True)
    elif(part == 'tsk3'):
        tsk3(tsk1_2(False))
    elif(part == "all"):
        tsk3(tsk1_2(True))
    t -= time.time()
    print(t)
if __name__ == '__main__':
    main()