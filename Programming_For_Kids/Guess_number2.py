import random

# Initialize the score
score = 20

# Level 1 (5 trials, hint if guess is higher or lower)
print("Level 1: Guess the number (5 trials, with hints)")
n = random.randint(1, 100)
for i in range(0, 5):
    num = int(input("Guess a number: "))
    if num == n:
        print("Wonderful, you got the number!")
        score += 50
        break
    elif num < n:
        print("Your guess is lower than the number. Try again.")
    else:
        print("Your guess is higher than the number. Try again.")
    score -= 4
else:
    print(f"You lost. The number was {n}.")

print(f"Your score for Level 1: {score}")

# Level 2 (7 trials, no hints)
print("\nLevel 2: Guess the number (7 trials, no hints)")
n = random.randint(1, 100)
for i in range(0, 7):
    num = int(input("Guess a number: "))
    if num == n:
        print("Wonderful, you got the number!")
        score += 100
        break
    else:
        print("Wrong guess, try again.")
    score -= 4
else:
    print(f"You lost. The number was {n}.")

print(f"Your score for Level 2: {score}")

# Level 3 (3 trials, no hints)
print("\nLevel 3: Guess the number (3 trials, no hints)")
n = random.randint(1, 100)
for i in range(0, 3):
    num = int(input("Guess a number: "))
    if num == n:
        print("Wonderful, you got the number!")
        score += 150
        break
    else:
        print("Wrong guess, try again.")
    score -= 4
else:
    print(f"You lost. The number was {n}.")

print(f"Your score for Level 3: {score}")