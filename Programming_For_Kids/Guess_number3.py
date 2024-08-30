import random

def guess_the_number(level):
    # Initialize the score
    score = 20
    trials = {1: (5, True), 2: (7, False), 3: (3, False)}
    
    if level not in trials:
        print("Invalid level. Please choose 1, 2, or 3.")
        return

    max_trials, hints = trials[level]
    n = random.randint(1, 100)

    print(f"Level {level}: Guess the number ({max_trials} trials, {'with hints' if hints else 'no hints'})")

    for i in range(max_trials):
        num = int(input("Guess a number: "))
        if num == n:
            print("Wonderful, you got the number!")
            score += 50 * (level)  # Increase points based on level
            break
        elif hints:
            if num < n:
                print("Your guess is lower than the number. Try again.")
            else:
                print("Your guess is higher than the number. Try again.")
        else:
            print("Wrong guess, try again.")
        
        score -= 4
    else:
        print(f"You lost. The number was {n}.")

    print(f"Your score for Level {level}: {score}")

# Example usage:
guess_the_number(1)  # Call for Level 1
#guess_the_number(2)  # Call for Level 2
#guess_the_number(3)  # Call for Level 3