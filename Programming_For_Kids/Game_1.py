import pygame
import random

# Initialize Pygame
pygame.init()

# Set the window size
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))

# Set the window title
pygame.display.set_caption("Simple Game")

# Set the background color to black
background_color = (0, 0, 0)

# Set the colors and scores for the top rectangles
rectangle_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
                    (0, 255, 255), (128, 128, 128), (255, 128, 0), (128, 0, 128), (0, 128, 128)]
rectangle_scores = [str(random.randint(0, 100)) for _ in range(10)]
rectangle_active = [True] * 10

# Set the dimensions for the rectangles
rectangle_width = (window_width - 90) // 10
rectangle_height = 30
rectangle_spacing = 10

# Set the dimensions and position for the main triangle
main_triangle_size = 50
main_triangle_x = window_width // 2 - main_triangle_size // 2
main_triangle_y = window_height - main_triangle_size

# Set the main triangle's speed
main_triangle_speed = 5

# Set the shot properties
shot_radius = 5
shot_speed = 10
shot_x = main_triangle_x + main_triangle_size // 2
shot_y = main_triangle_y
shot_active = False

# Set the player's score
player_score = 0

# Set the game timer
game_duration = 30  # 30 seconds
start_time = pygame.time.get_ticks()
game_over = False

# Game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            shot_x = main_triangle_x + main_triangle_size // 2
            shot_y = main_triangle_y
            shot_active = True

    # Handle key presses
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        main_triangle_x -= main_triangle_speed
    if keys[pygame.K_RIGHT]:
        main_triangle_x += main_triangle_speed

    # Ensure the main triangle stays within the window boundaries
    main_triangle_x = max(0, min(window_width - main_triangle_size, main_triangle_x))

    # Move the shot
    if shot_active:
        shot_y -= shot_speed
        if shot_y < 0:
            shot_active = False

    # Check for collisions between the shot and the top rectangles
    for i in range(len(rectangle_colors)):
        if rectangle_active[i]:
            x = i * (rectangle_width + rectangle_spacing) + rectangle_spacing
            y = rectangle_spacing
            rect = pygame.Rect(x, y, rectangle_width, rectangle_height)
            if shot_active and rect.collidepoint(shot_x, shot_y):
                # Remove the rectangle and update the player's score
                rectangle_active[i] = False
                player_score += int(rectangle_scores[i])
                shot_active = False
                break

    # Check if the game time is up
    current_time = pygame.time.get_ticks() - start_time
    if current_time >= game_duration * 1000:
        game_over = True

    # Fill the background with black color
    window.fill(background_color)

    # Draw the top rectangles
    for i in range(len(rectangle_colors)):
        if rectangle_active[i]:
            x = i * (rectangle_width + rectangle_spacing) + rectangle_spacing
            y = rectangle_spacing
            pygame.draw.rect(window, rectangle_colors[i], (x, y, rectangle_width, rectangle_height))
            font = pygame.font.Font(None, 24)
            text = font.render(rectangle_scores[i], True, (255, 255, 255))
            text_rect = text.get_rect(center=(x + rectangle_width // 2, y + rectangle_height // 2))
            window.blit(text, text_rect)

    # Draw the main triangle
    points = [(main_triangle_x, main_triangle_y),
              (main_triangle_x + main_triangle_size, main_triangle_y),
              (main_triangle_x + main_triangle_size // 2, main_triangle_y - main_triangle_size)]
    pygame.draw.polygon(window, (255, 255, 255), points)

    # Draw the shot
    if shot_active:
        pygame.draw.circle(window, (255, 0, 0), (int(shot_x), int(shot_y)), shot_radius)

    # Display the player's score and the remaining time
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {player_score}", True, (255, 255, 255))
    score_rect = score_text.get_rect(bottomleft=(10, window_height - 10))
    window.blit(score_text, score_rect)

    time_left = max(0, game_duration - current_time // 1000)
    time_text = font.render(f"Time: {time_left:02d}", True, (255, 255, 255))
    time_rect = time_text.get_rect(bottomleft=(10, window_height - 50))
    window.blit(time_text, time_rect)

    # Check if the game is over
    if game_over:
        # Display a game over message
        game_over_text = font.render("Game Over!", True, (255, 255, 255))
        game_over_rect = game_over_text.get_rect(center=(window_width // 2, window_height // 2))
        window.blit(game_over_text, game_over_rect)
        pygame.display.update()
        pygame.time.wait(3000)  # Wait for 3 seconds before quitting
        running = False

    # Update the display
    pygame.display.update()

# Quit Pygame
pygame.quit()