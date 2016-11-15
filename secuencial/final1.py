from PIL import Image
import numpy as np
import math
import time
np.set_printoptions(threshold=np.nan)
result=[]
result2=[]
# img = Image.open("lena512.bmp")
# a = np.array(img.convert('P', palette=Image.ADAPTIVE, colors=255))
# blockLengthX = np.argmin(a[0]==a[0,0])
# blockLengthY = np.argmin(a[:,0]==a[0,0])
# result = a[::blockLengthX, ::blockLengthY]

# img2 = Image.open("lena512.bmp")
# b = np.array(img2.convert('P', palette=Image.ADAPTIVE, colors=256))
# blockLengthX = np.argmin(b[0]==b[0,0])
# blockLengthY = np.argmin(b[:,0]==b[0,0])
# result2 = b[::blockLengthX, ::blockLengthY]

# x2 =len(result)
# y2=len(result[0])

def load_dataset(filename,result):
    with open(filename) as f:
        for line in f:
            line = line.split(",")
            if line:
                line = [int(i) for i in line]
                result.append(line)
    return result


load_dataset("dataSet64x64.txt",result)
load_dataset("dataSet64x64.txt", result2)



x2 = len(result)/16-1
y2 = len(result[0])/16-1


matrixResult = [[0 for x in range(x2 + 1)] for y in range(y2 + 1 )]  # matriz x2 * y2
def crearMacro(fIni,fFin,cIni,cFin,matrix): 
    return [row[cIni:cFin] for row in matrix[fIni:fFin]]


def compararMacro(macro1, macro2,x,y):
    resta = 0
    for i in range(0,15):
        for j in range(0,15):
            proceso = macro1[x+i][y+j] - macro2[x+i][x+j]
            resta += math.fabs(proceso)
        
    return resta


def saveMatrix(valor1,valor2,p,n):
    palabra = "MB"
    matrixResult [valor1][valor2] = palabra + "(" + str(valor1) + "," + str(valor2) + ")" + "=" + "(" + str(p) + "," + str(n) + ")"

def compararBmp(result,result2):
    tiempo = time.time()
    filas1 =len(result)-15
    columnas1=len(result[0])-15
    filas2 = len(result2)-15
    columnas2 = len(result2[0])-15
    valorMenor = 100
    nuevah = 0
    nuevak = 0
    macro16 = 0

    for i in range(0,filas1,16):
        for j in range(0,columnas1,16):
            macro1 = crearMacro(i,i+16,j,j+16,result)
            for h in xrange(filas2):
                for k in xrange(columnas2):
                    macro2 = crearMacro(h,h+16,k,k+16,result2)
                    criterioParada = compararMacro(macro1,macro2,0,0)
                    if (criterioParada == 0):
                        saveMatrix(i/16,j/16,h,k)
                    if (criterioParada != 0):
                        if(criterioParada < valorMenor):
                            valorMenor = criterioParada
                            nuevah = h
                            nuevak = k
                            saveMatrix(i/16,j/16,nuevah,nuevak)

    print matrixResult
    print (time.time() - tiempo, "segundos")
# MAIN

compararBmp(result,result2)

# FIN MAIN
