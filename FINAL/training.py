import pygame
import csv

WINWIDTH = 800
WINHEIGHT = 800

WIN = pygame.display.set_mode((WINWIDTH, WINHEIGHT))
pygame.display.set_caption("NUMS DATA COLLECTION")

#colors
GRAY = (211,211,211)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 153)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
LIGHTBLUE = (173, 216, 230)

SCALING = 32
BORDERWIDTH = SCALING // 4
BOUNDS = (WINWIDTH//SCALING, WINHEIGHT//SCALING)



def draw_borders(walls):
    WIN.fill((255, 255, 255))
    verticals = []
    for i in range(1, 100):
        verticals.append(pygame.Rect(i * SCALING, 0, BORDERWIDTH, WINHEIGHT))
    for vertical in verticals:
        pygame.draw.rect(WIN, GRAY, vertical)

    horizontals = []
    for i in range(1, 100):
        horizontals.append(pygame.Rect(0, i * SCALING, WINWIDTH, BORDERWIDTH))
    for horizontal in horizontals:
        pygame.draw.rect(WIN, GRAY, horizontal)

    for wall in walls:
        color_square(wall[0], wall[1], BLUE)


def color_square(a, b, color):     #a is the row, b is the column
    if a != 0 and b != 0:
        pygame.draw.rect(WIN, color, pygame.Rect((a * SCALING)+BORDERWIDTH, (b * SCALING)+BORDERWIDTH, SCALING-BORDERWIDTH, SCALING-BORDERWIDTH))
    elif a == 0 and b != 0:
        pygame.draw.rect(WIN, color, pygame.Rect((a * SCALING), (b * SCALING)+BORDERWIDTH, SCALING, SCALING-BORDERWIDTH))
    elif a != 0 and b == 0:
        pygame.draw.rect(WIN, color, pygame.Rect((a * SCALING)+BORDERWIDTH, (b * SCALING), SCALING-BORDERWIDTH, SCALING))
    else:
        pygame.draw.rect(WIN, color, pygame.Rect((a * SCALING), (b * SCALING), SCALING, SCALING))


def isInWalls(coord, walls):
    nums = coord.split(".")
    for wall in walls:
        if int(nums[0]) == wall[0] and int(nums[1]) == wall[1]:
            return True
    return False


def main():
    run = True
    pathFound = False
    FPS = 144
    clock = pygame.time.Clock()

    walls = []
    mouseStatus = [0, 0, 0]
    mode = None

    keyNums = [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]
    
    fieldnames = ['num']
    tempString = ""
    for i in range(25):
        for j in range(25):
            tempString = f"{i}.{j}"
            fieldnames.append(tempString)
    
    while run:
        clock.tick(FPS)

        # if not pathFound:
        #     draw_borders(walls)

        draw_borders(walls)

        for event in pygame.event.get():
            mouseButtons = pygame.mouse.get_pressed()
            mouseStatus.append(mouseButtons[0])
            mouseStatus.pop(0)
            pos = pygame.mouse.get_pos()
            wall = (pos[0]//SCALING, pos[1]//SCALING)
            if mouseStatus[-1] != mouseStatus[-2]:
                if mouseStatus[-1] == 1:
                    if wall in walls:
                        mode = "remove"
                    else:
                        mode = "add"
                else:
                    mode = None
            if mode == "add":
                if wall not in walls:
                   walls.append(wall)
            elif mode == "remove":
                if wall in walls:
                    walls.remove(wall)

            if event.type == pygame.KEYDOWN:
                if event.key in keyNums:
                    # keyNums.index(event.key)
                    # walls.clear()

                    with open(r'data/numbers_test.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                        tempDict = {}
                        

                        for i in range(1, len(fieldnames)):
                            if isInWalls(fieldnames[i], walls):
                                tempDict[fieldnames[i]] = 1
                            else:
                                tempDict[fieldnames[i]] = 0

                        tempDict["num"] = keyNums.index(event.key)
                        
                        writer.writerow(tempDict)

                    walls.clear()
                        



            if event.type == pygame.QUIT:
                run = False




        pygame.display.update()

    pygame.quit()



if __name__ == "__main__":
    main()
