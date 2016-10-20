import turtle as t
RIGHTANGLE=90
SPACE=10

def basic(width):
    t.forward(width)
    t.right(RIGHTANGLE)
    t.forward(3*width)
    t.right(RIGHTANGLE)
    t.forward(width)

def space_maker():
    t.up()
    t.forward(SPACE)
    t.down()

def basic2(width):
    t.forward(width)
    t.right(RIGHTANGLE)
    t.forward(width)
    t.left(RIGHTANGLE)
    t.forward(width)
    t.left(RIGHTANGLE)
    t.forward(width)
    t.right(RIGHTANGLE)
    t.forward(width)

def recursive_Program2(level,width):
    #
    if level>1:
        t.left(180)
        recursive_Program2(level-1,width)
        t.right(180)

    basic2(width)
    if level>1:
        t.left(RIGHTANGLE)
        recursive_Program2(level-1,width)
        t.right(RIGHTANGLE)


    t.right(RIGHTANGLE)
    #space_maker()
    basic2(width)
    #space_maker()
    t.right(RIGHTANGLE)

    if level>1:
        t.right(180)
        recursive_Program2(level-1,width)
        t.right(180)
    basic2(width)

    t.right(RIGHTANGLE)
    #space_maker()
    if level>1:
        t.right(180)
        recursive_Program2(level-1,width)
        t.right(180)
    basic2(width)
    #space_maker()
    t.right(RIGHTANGLE)


def recursive_Program(level,width):
    #
    if level>1:
        t.left(180)
        recursive_Program(level-1,width)
        t.right(180)

    basic(width)
    if level>1:
        t.left(RIGHTANGLE)
        recursive_Program(level-1,width)
        t.right(RIGHTANGLE)

    t.right(RIGHTANGLE)
    basic(width)
    t.right(RIGHTANGLE)
    if level>1:
        t.right(180)
        recursive_Program(level-1,width)
        t.right(180)
    basic(width)
    t.right(RIGHTANGLE)
    if level>1:
        t.right(180)
        recursive_Program(level-1,width)
        t.right(180)
    basic(width)
    t.right(RIGHTANGLE)

#t.right(45)
#recursive_Program(3,1,25)
recursive_Program2(2,25)


t.mainloop()