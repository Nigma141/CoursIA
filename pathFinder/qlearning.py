# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
from tkinter import *

stop = False


def initialization():
    # parametres
    global reward, gain, n, Q, r, block_square, window, heightY, widthX, size_of_square, posX, posY, canvas
    reward = 100
    gain = 0.5
    n = 8  # largeur de notre carre
    Q = np.zeros((np.square(n), 4))
    r = np.zeros((np.square(n), 1))
    r[np.square(n) - 1] = reward
    block_square = np.zeros(np.square(n));
    window = Tk()
    heightY = n
    widthX = n
    size_of_square = 60
    posX = 0
    posY = 0

    # Création du
    canvas = Canvas(window, height=size_of_square * heightY, width=size_of_square * widthX, background='white')
    for j in range(heightY):
        for i in range(widthX):
            canvas.create_rectangle(i * size_of_square, j * size_of_square, (i + 1) * size_of_square,
                                    (j + 1) * size_of_square, fill="white")

    canvas.create_rectangle(posX,posY, size_of_square, size_of_square, fill="red")

    # Modifier Q pour donner les mouvements possibles


# main function:

def moveSquare():
    global reward, gain, n, Q, r, block_square, size_of_square, posX, posY, stop


    if stop == False:
        canvas.create_rectangle(posX, posY, size_of_square, size_of_square, fill="white")
        window.after = (10, moveSquare)
    stop = False


def blockSquare(event):
    global size_of_square, block_square

    x_pos = event.x
    y_pos = event.y

    a = int(x_pos / size_of_square)
    b = int(y_pos / size_of_square)

    block_square[n * b + a] = -1
    canvas.create_rectangle(a * size_of_square, b * size_of_square, (a + 1) * size_of_square, (b + 1) * size_of_square,
                            fill="black")
    # a chaque fois modifier Q pour dire les mouvements possible

def stopSquare():
    global stop
    stop = True


initialization()

ButtonStart = Button(window, text="Start", command=moveSquare)
ButtonStart.pack(side=TOP, padx=5, pady=5)

Button_Stop = Button(window, text="Stop", command=stopSquare)
Button_Stop.pack(side=TOP, padx=5, pady=5)

canvas.bind("<Button-1>", blockSquare)
canvas.pack()
window.mainloop()









