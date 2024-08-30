#The project is by Mohannad, Hamza and Laith for Dr. Alaa, we hope you like it
print("Welcome to the math exam")
print("The exam will be 3 levels")
print("Let's start at Level 1")

score1 = 0
Questions_L1=["2x4", "3x5", "2x6", "2x8", "4x5"]
Answers_L1=["8","15","12","16","20"]

for i in range(len(Questions_L1)):
    print(Questions_L1[i], " = ", end="")
    answer = int(input())
    if answer == int(Answers_L1[i]):
        score1 += 1



print("\nTest 1 Results:")
print("You answered " + str(score1) + " out of five questions correctly.")

print("Let's start at Level 2")
print("Level 2 will be more difficult than Level 1. Be fully prepared")

score2 = 0

Questions_L2=["5X9", "8X7", "6X11", "7x9", "7x12"]
Answers_L2=["45","56","66","63","84"]

for i in range(len(Questions_L2)):
    print(Questions_L2[i], " = ", end="")
    answer = int(input())
    if answer == int(Answers_L2[i]):
        score2 += 1    
    
print("\nTest 2 Results:")
print("You answered " + str(score2) + " out of ten questions correctly.")

print("Let's start at Level 3")
print("Well done for reaching level 3. Get ready because this is the last level and the hardest level. Good luck.")

score3 = 0

Questions_L3=["7X6+12", "5X8-6", "7X9+17", "7X7+15", "5X9+23"]
Answers_L3=["54","34","80","64","68"]

for i in range(len(Questions_L3)):
    print(Questions_L3[i], " = ", end="")
    answer = int(input())
    if answer == int(Answers_L3[i]):
        score3 += 1    
    
print("\nTest 3 Results:")
print("You answered " + str(score3) + " out of Fifteen questions correctly.")

print("If you arrive here and answer all the questions correctly, I congratulate you, you are an expert in mathematics")
print("goodbye")
#The project is sponsored by Dr. Alaa. I hope you like it