from PIL import Image
from mpi4py import MPI
import numpy as np
import math
import time
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

np.set_printoptions(threshold=np.nan)
result=[]
result2=[]


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
    print matrixResult

"""def compararBmp(result,result2):
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
            print macro1
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


    print (time.time() - tiempo, "segundos")"""

tiempo = time.time()
# MAIN

if rank==0:
    filas1 =len(result)-15
    columnas1=len(result[0])-15
    filas2 = len(result2)-15
    columnas2 = len(result2[0])-15
    valorMenor = 100
    nuevah = 0
    nuevak = 0
    macro16 = 0
    node = 1

    for i in range(0,filas1,16):
        for j in range(0,columnas1,16):
            if(node == nprocs):
                node = 1
            data_send = i
            destination_process = node
            comm.send(data_send,dest=destination_process)
            data_send = j
            destination_process= node
            comm.send(data_send,dest=destination_process)
            data_send = filas2
            destination_process= node
            comm.send(data_send,dest=destination_process)
            data_send = columnas2
            destination_process= node
            comm.send(data_send,dest=destination_process)
            data_send = matrixResult
            destination_process= node
            comm.send(data_send,dest=destination_process)
            """ source_process = node
            data_received=comm.recv(source=source_process)"""
            node += 1
print matrixResult
print (time.time() - tiempo, "segundos")

if rank!=0:
    source_process = 0
    data_received=comm.recv(source=source_process)
    nuevai = data_received
    source_process = 0
    data_received=comm.recv(source=source_process)
    nuevaj = data_received
    source_process = 0
    data_received=comm.recv(source=source_process)
    nuevafilas2 = data_received - 1
    source_process = 0
    data_received=comm.recv(source=source_process)
    nuevacolumnas2 = data_received - 1
    macro1 = crearMacro(nuevai,nuevai+16,nuevaj,nuevaj+16,result)
    for h in xrange(nuevafilas2):
        for k in xrange(nuevacolumnas2):
            macro2 = crearMacro(h,h+16,k,k+16,result2)
            criterioParada = compararMacro(macro1,macro2,0,0)
            if (criterioParada == 0):
                saveMatrix(nuevai/16,nuevaj/16,h,k)
                if (criterioParada != 0):
                    if(criterioParada < valorMenor):
                        valorMenor = criterioParada
                        nuevah = h
                        nuevak = k
                        saveMatrix(nuevai/16,nuevaj/16,nuevah,nuevak)
    """data_send = "Hola, list"
    destination_process= 0
    comm.send(data_send,dest=destination_process)"""

print matrixResult
# compararBmp(result,result2)

# FIN MAIN
