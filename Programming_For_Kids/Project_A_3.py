#The project is by Mohannad, Hamza and Laith for Dr. Alaa, we hope you like it
import random

def Exam(Questions,Answers):
    print("Let's start at Level 1")

    # Create a list of indices
    number_questions = list(range(0, len(Questions)))
    # Generate a random permutation of the indices
    random_permutation = random.sample(number_questions, len(number_questions))

    score = 0
    
    for i in range(0, len(Questions)):
        print(Questions[random_permutation[i]], " = ", end="")
        answer = int(input())
        if answer == int(Answers[random_permutation[i]]):
            score += 1
    
    return score

# Main program 

print("Welcome to the math exam")
print("The exam will be 3 levels")
print("Let's start at Level 1")

Questions_L1=["2x4", "3x5", "2x6", "2x8", "4x5"]
Answers_L1=["8","15","12","16","20"]
score1= Exam(Questions_L1,Answers_L1)

print("\nTest 1 Results:")
print("You answered " + str(score1) + " out of five questions correctly.")

print("Let's start at Level 2")
print("Level 2 will be more difficult than Level 1. Be fully prepared")

Questions_L2=["5X9", "8X7", "6X11", "7x9", "7x12"]
Answers_L2=["45","56","66","63","84"]
score2= Exam(Questions_L2,Answers_L2)

print("\nTest 2 Results:")
print("You answered " + str(score2) + " out of five questions correctly.")

print("Let's start at Level 3")
print("Well done for reaching level 3. Get ready because this is the last level and the hardest level. Good luck.")

Questions_L3=["7X6+12", "5X8-6", "7X9+17", "7X7+15", "5X9+23"]
Answers_L3=["54","34","80","64","68"]
score3= Exam(Questions_L3,Answers_L3)

print("\nTest 3 Results:")
print("You answered " + str(score3) + " out of five questions correctly.")

print("If you arrive here and answer all the questions correctly, I congratulate you, you are an expert in mathematics")
print("goodbye")
#The project is sponsored by Dr. Alaa. I hope you like it