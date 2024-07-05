#!/usr/bin/env python
n=68

for i in range(0,5):
    num=int(input("Guess a number: "))
    if num==n:
        print("Wonderful, you got the no.")
        break
    else:
        if i<4:
            print("Try again, you still have " + str(4-i) +  " trails")
        else:
            print("You lost, good bye")
