# Python code to implement Conway's Game Of Life
# Also has implemented statistic about some of the known shapes that appear in this game
# It also detects them in any orientation or state
# The board is treated as if it was connected by the boarders
# Author Luis Castillo
import argparse
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapes import *



# setting up the values for the grid
ON = 255
OFF = 0
vals = [ON, OFF]
iteration = 0


def randomGrid(width, height):
 
    """returns a grid of NxN random values"""
    return np.random.choice(vals, width* height, p=[0.2, 0.8]).reshape(width, height)
 
def addGlider(i, j, grid):
 
    """adds a glider with top left cell at (i, j)"""
    glider = np.array([[0,    0, 255],
                       [255,  0, 255],
                       [0,  255, 255]])
    grid[i:i+3, j:j+3] = glider

def addFileInput(cells,grid):
    for live_cell in cells:
        grid[int(live_cell[0]), int(live_cell[1])] = ON
    return grid

def rotateMatCheck(matrix,shape,rotations):
    rotMat = matrix
    for i in range(rotations):
        rotMat = np.rot90(rotMat)
        isSame = rotMat == shape
        if(isSame.all()):
            return True

    return False

def update(frameNum, img, grid, width,height,res):
    global iteration 
    counterShapes = {'Glider': 0,'Block': 0,'Beehive': 0,'Loaf': 0,'Boat': 0,'Tub': 0,'Blinker': 0,'Toad': 0,'Spaceship': 0,'Beacon': 0,}
    
    # copy grid since we require 8 neighbors
    # for calculation and we go line by line
    newGrid = grid.copy()
    for i in range(width):
        for j in range( height):
            neigh = int(( grid[(i-1)%width , (j-1)%height] +  grid[i, (j-1)%height]      + grid[(i+1)%width, (j-1)%height] +
                         grid[(i-1)%width , j]              +                            grid[(i+1)%width, j]+ 
                         grid[(i-1)%width , (j+1)%height] + grid[i, (j+1)%height]       + grid[(i+1)%width , (j+1)%height])/255)
            
            # apply Conway's rules
            if grid[i, j]  == ON:
                if (neigh < 2) or (neigh > 3):
                    newGrid[i, j] = OFF
            else:
                if neigh == 3:
                    newGrid[i, j] = ON
    # we use the new grid 
    # get matrixes to compare with the different shapes
    for i in range(width-6):
        for j in range(height-7):
            mat_4 = np.array(newGrid[i:i + 4, j:j + 4])
            mat_5 = np.array(newGrid[i:i + 5, j:j + 5])
            mat_6 = np.array(newGrid[i:i + 6, j:j + 6])
            mat_7 = np.array(newGrid[i:i + 7, j:j + 7])
            
            if mat_4.shape == (4, 4):
                isBlock = mat_4 == block
                if isBlock.all(): counterShapes['Block'] += 1

            if mat_5.shape == (5, 5):
                isTub = mat_5 == tub
                isBlinker = rotateMatCheck(mat_5,blinker,2)
                isBoat = rotateMatCheck(mat_5,boat,4)
                isGlider_1 = rotateMatCheck(mat_5,glider_1,4)
                isGlider_2 = rotateMatCheck(mat_5,glider_2,4)
                isGlider_3 = rotateMatCheck(mat_5,glider_3,4)
                isGlider_4 = rotateMatCheck(mat_5,glider_4,4)
                if isTub.all():  counterShapes['Tub'] += 1
                if isBoat: counterShapes['Boat'] += 1
                if isBlinker: counterShapes['Blinker'] +=1
                if isGlider_1 or isGlider_2 or isGlider_3 or isGlider_4:
                    counterShapes['Glider'] += 1

            if mat_6.shape == (6, 6):
                isBeehive = rotateMatCheck(mat_6,behive,2)
                isToad_1 = rotateMatCheck(mat_6,toad_1,2)
                isToad_2 = rotateMatCheck(mat_6,toad_2,4)
                isBeacon_1 = rotateMatCheck(mat_6,beacon_1,2)
                isBeacon_2 = rotateMatCheck(mat_6,beacon_2,2)
                isLoaf = rotateMatCheck(mat_6,loaf,4)
                if isToad_1 or isToad_2:  counterShapes['Toad'] += 1
                if isBeacon_1 or isBeacon_2: counterShapes['Beacon'] += 1
                if isLoaf:  counterShapes['Loaf'] += 1
                if isBeehive : counterShapes['Beehive'] += 1

            if mat_7.shape == (7, 7):
                isSpaceship_1 = rotateMatCheck(mat_7,spaceship_1,4)
                isSpaceship_2 = rotateMatCheck(mat_7,spaceship_2,4)
                isSpaceship_3 = rotateMatCheck(mat_7,spaceship_2,4)
                isSpaceship_4 = rotateMatCheck(mat_7,spaceship_2,4)
                if isSpaceship_1 or isSpaceship_2 or isSpaceship_3 or isSpaceship_4:
                    counterShapes['Spaceship'] += 1


    shapestextformat = [ "Block      ","Beehive    ", "Loaf       ", "Boat       ", "Tub        ", "Blinker    ", "Toad       ", "Beacon     ", "Glider     ", "LWspaceship"]
    nameshapes = ['Block','Beehive','Loaf','Boat','Tub','Blinker','Toad','Beacon','Glider','Spaceship']

    shapestotal = sum(counterShapes.values())
    #prevent division by zero
    if(shapestotal == 0):
        auxNotZero = 1
    else:
        auxNotZero = 0
    shapesPercent = {
    'Glider': int(100*counterShapes['Glider']/(shapestotal+auxNotZero)),
    'Block': int(100*counterShapes['Block']/(shapestotal+auxNotZero)),
    'Beehive': int(100*counterShapes['Beehive']/(shapestotal+auxNotZero)),
    'Loaf': int(100*counterShapes['Loaf']/(shapestotal+auxNotZero)),
    'Boat': int(100*counterShapes['Boat']/(shapestotal+auxNotZero)),
    'Tub': int(100*counterShapes['Tub']/(shapestotal+auxNotZero)),
    'Blinker': int(100*counterShapes['Blinker']/(shapestotal+auxNotZero)),
    'Toad': int(100*counterShapes['Toad']/(shapestotal+auxNotZero)),
    'Spaceship': int(100*counterShapes['Spaceship']/(shapestotal+auxNotZero)),
    'Beacon': int(100*counterShapes['Beacon']/(shapestotal+auxNotZero)),
    }


    res.write("Iteration:" + str(iteration) +"\n")
    res.write("---------------------------\n")
    res.write("Name      Count      Percent\n")
    res.write("---------------------------\n")
    aux = 0
    for shape in shapestextformat:
        res.write( str(shape)+"    "+ str(counterShapes[str(nameshapes[aux])]) +"       " +  str(shapesPercent[str(nameshapes[aux])])+"\n")
        aux+=1
    res.write("---------------------------\n")
    res.write("Total         "+ str(shapestotal) +"           \n")
    res.write("---------------------------\n\n")

    # update data
    iteration +=1
    img.set_data(newGrid)
    grid[:] = newGrid[:]
    return img

def getneighbors(radius, row_number, column_number, grid):
     return [[grid[i][j] if  i >= 0 and i < len(grid) and j >= 0 and j < len(grid[0]) else 0
                for j in range(column_number-1-radius, column_number+radius)]
                    for i in range(row_number-1-radius, row_number+radius)]


# main() function
def main():
 
    # Command line args are in sys.argv[1], sys.argv[2] ..
    # sys.argv[0] is the script name itself and can be ignored
    # parse arguments
    parser = argparse.ArgumentParser(description="Runs Conway's Game of Life simulation.")
    
    # add arguments
    parser.add_argument('--grid-width', dest='width', required=False)
    parser.add_argument('--grid-height', dest='height', required=False)
    parser.add_argument('--iterations', dest='iterations', required=False)
    args = parser.parse_args()



    f = open("input5.txt", "r")
    w,h = f.readline().split()
    width = int(w)
    height = int(h)
    it = f.readline()
    iterations = int(it)
    content = f.readlines()
    cells = []
    for line in content:
        x,y = line.split()
        cells.append([x,y])


    # set grid size
    if args.width and int(args.width) > 8:
        width = int(args.width)
    if args.height and int(args.height) > 8:
        height = int(args.height)

    if args.iterations and int(args.iterations) > 1:
        iterations = int(args.iterations)

    # set animation update interval
    updateInterval = 50
 
    # declare grid
    grid = np.array([])
    # fill the grid with zeros
    grid = np.zeros(width*height).reshape(width, height)

    # populate grid with random on/off -
    # more off than on
    #grid = randomGrid(width, height)

    grid = addFileInput(cells,grid)
    #addGlider(20, 20, grid)

    #Set output initial text
    today = date.today() 
    res = open("output5.txt","w")
    res.write("Simulation at "+str(today)+"\n")
    res.write("Universe size "+str(width)+ " x "+ str(height)+"\n")
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest')
    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, width,height,res ),
                                  frames=int(iterations),
                                  interval=updateInterval,
                                  save_count=50,
                                  repeat=False)
        
    plt.show()
    res.close()


if __name__ == '__main__':
    main()