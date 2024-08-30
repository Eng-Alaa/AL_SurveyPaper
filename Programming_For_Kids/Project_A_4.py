import random

def Exam(Questions_list, Answers_list, Attempts):
    print("Let's start the exam")

    # Create a list of indices
    number_questions = list(range(0, len(Questions_list)))
    # Generate a random permutation of the indices
    random_permutation = random.sample(number_questions, len(number_questions))

    score = 0
    user_answers = []

    for i in range(len(Questions_list)):
        print(Questions_list[random_permutation[i]], " = ", end="")
        for j in range(Attempts):
            answer = input().lower()
            if answer == Answers_list[random_permutation[i]].lower():
                print("Correct answer")
                score += 1
                user_answers.append(answer)
                break
            else:
                if j < Attempts - 1:
                    print("Incorrect answer, try again")
                else:
                    print("Correct answer is:", Answers_list[random_permutation[i]])
                    user_answers.append(answer)

    return score, user_answers

def display_solutions(Questions_list, User_answers, Answers_list):
    print("The Solutions Of The Exam")
    for i in range(len(Questions_list)):
        print(Questions_list[i])
        print("Your answer is: ", User_answers[i])
        print("Correct Answer is: ", Answers_list[i])

# Main program
print("Welcome to the math exam")
print("The exam will have 3 levels")

# Level 1
print("Let's start at Level 1")
Questions_L1 = ["2x4", "3x5", "2x6", "2x8", "4x5"]
Answers_L1 = ["8", "15", "12", "16", "20"]
Attempts_L1 = 2
score1, user_answers_1 = Exam(Questions_L1, Answers_L1, Attempts_L1)

print("\nTest 1 Results:")
print("You answered " + str(score1) + " out of five questions correctly.")

# Level 2
print("\nLet's start at Level 2")
print("Level 2 will be more difficult than Level 1. Be fully prepared")
Questions_L2 = ["5X9", "8X7", "6X11", "7x9", "7x12"]
Answers_L2 = ["45", "56", "66", "63", "84"]
Attempts_L2 = 2
score2, user_answers_2 = Exam(Questions_L2, Answers_L2, Attempts_L2)

print("\nTest 2 Results:")
print("You answered " + str(score2) + " out of five questions correctly.")

# Level 3
print("\nLet's start at Level 3")
print("Well done for reaching level 3. Get ready because this is the last level and the hardest level. Good luck.")
Questions_L3 = ["7X6+12", "5X8-6", "7X9+17", "7X7+15", "5X9+23"]
Answers_L3 = ["54", "34", "80", "64", "68"]
Attempts_L3 = 2
score3, user_answers_3 = Exam(Questions_L3, Answers_L3, Attempts_L3)

print("\nTest 3 Results:")
print("You answered " + str(score3) + " out of five questions correctly.")

print("\nIf you arrived here and answered all the questions correctly, I congratulate you, you are an expert in mathematics")
print("Goodbye")

print("\nDisplaying solutions:")
display_solutions(Questions_L1, user_answers_1, Answers_L1)
display_solutions(Questions_L2, user_answers_2, Answers_L2)
display_solutions(Questions_L3, user_answers_3, Answers_L3)