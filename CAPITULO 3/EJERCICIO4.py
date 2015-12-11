#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image,ImageDraw
import codecs
charset = "utf-8"

def readfile(filename):
  f=codecs.open(filename,encoding="utf-8", errors="ignore")
  lines=[line for line in f]
  
  # La primera linea son los títulos de columna
  colnames=lines[0].strip().split('\t')[1:]
  rownames=[]
  data=[]
  for line in lines[1:]:
    p=line.strip().split('\t')
    # La primera columna en cada fila es el nombre de la fila
    rownames.append(p[0])
    # Los datos para esta fila es lo que queda de la fila
    data.append([float(x) for x in p[1:]])
  return rownames,colnames,data
from math import sqrt

def euclidean(v1,v2):
  return sqrt(sum([(v1[i]-v2[i])**2 for i in range(len(v1))]))
#==========================================================================
def pearson(v1,v2):
  # Sumas simples
  sum1=sum(v1)
  sum2=sum(v2)

  n = float(len(v1)) # esto es necesario por que sino la division es entera
  
  # Sums of the squares
  sum1Sq=sum([pow(v,2) for v in v1])
  sum2Sq=sum([pow(v,2) for v in v2])	
  
  # Sum of the products
  pSum=sum([v1[i]*v2[i] for i in range(len(v1))])
  
  # Calculate r (Pearson score)
  num=pSum-(sum1*sum2/n)
  den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
  if den==0: return 0

  return 1.0-num/den

class bicluster:
  def __init__(self,vec,left=None,right=None,distance=0.0,id=None):
    self.left=left
    self.right=right
    self.vec=vec
    self.id=id
    self.distance=distance
#rows filas
def hcluster(rows,distance=pearson):
  distances={}
  currentclustid=-1

  # Los clusters son inicialmente las filas
  clust=[bicluster(rows[i],id=i) for i in range(len(rows))]

  while len(clust)>1:
    lowestpair=(0,1)
    closest=distance(clust[0].vec,clust[1].vec)

    # Iterando atraves de cada par viendo la distancia mas pequeña 
    for i in range(len(clust)):
      for j in range(i+1,len(clust)):
        # distances is the cache of distance calculations
        if (clust[i].id,clust[j].id) not in distances:
          distances[(clust[i].id,clust[j].id)]=distance(clust[i].vec,clust[j].vec)

        d=distances[(clust[i].id,clust[j].id)]
        if d<closest:
          closest=d
          lowestpair=(i,j)
          

    # calcular el promedio de los dos grupos
    mergevec=[
    (clust[lowestpair[0]].vec[i]+clust[lowestpair[1]].vec[i])/2.0 
    for i in range(len(clust[0].vec))]

    # crear el nuevo grupo
    newcluster=bicluster(mergevec,left=clust[lowestpair[0]],
                         right=clust[lowestpair[1]],
                         distance=closest,id=currentclustid)

    # identificadores de racimo que no están en el conjunto original son negativos
    currentclustid-=1
    del clust[lowestpair[1]]
    del clust[lowestpair[0]]
    clust.append(newcluster)

  return clust[0]
#muestra las cluster que hemos obtenidos
def printclust(clust,labels=None,n=0):
  # Identar para hacer un esquema jerarquico
  for i in range(n): print ' ',
  if clust.id<0:
    # un ID negativo significa que esta es una rama 
    print '-'#, clust.distance
  else:
    # un ID positivo significa que es un punto final
    if labels==None: print clust.id
    else: print labels[clust.id] #, clust.distance

  # Ahora imprimimos las ramas derecha e izquierda
  if clust.left!=None: printclust(clust.left,labels=labels,n=n+1)
  if clust.right!=None: printclust(clust.right,labels=labels,n=n+1)

def getheight(clust):
#¿Es este un punto final? A continuación, la altura es de sólo 1
  if clust.left==None and clust.right==None: return 1

  # De lo contrario, la altura es la misma de las alturas de
  # cada rama
  return getheight(clust.left)+getheight(clust.right)

def getdepth(clust):
  # La distancia de un punto final es 0.0
  if clust.left==None and clust.right==None: return 0

  # La distancia de una rama es el mayor de sus dos lados
  # además de su propia distancia
  return max(getdepth(clust.left),getdepth(clust.right))+clust.distance


def drawdendrogram(clust,labels,jpeg='clusters.jpg'):
  # altura y la anchura
  h=getheight(clust)*20
  w=1200
  depth=getdepth(clust)

  # anchura es fija, por lo que las distancias de escala en consecuencia
  scaling=float(w-150)/depth

  # Crear una nueva imagen con un fondo blanco
  img=Image.new('RGB',(w,h),(255,255,255))
  draw=ImageDraw.Draw(img)

  draw.line((0,h/2,10,h/2),fill=(255,0,0))    

  # Dibuja el primer nodo
  drawnode(draw,clust,10,(h/2),scaling,labels)
  img.save(jpeg,'JPEG')

def drawnode(draw,clust,x,y,scaling,labels):
  if clust.id<0:
    h1=getheight(clust.left)*20
    h2=getheight(clust.right)*20
    top=y-(h1+h2)/2
    bottom=y+(h1+h2)/2
    # Longitud de la línea
    ll=clust.distance*scaling
    # Línea vertical de este grupo a los niños    
    draw.line((x,top+h1/2,x,bottom-h2/2),fill=(255,0,0))    
    
    # Línea horizontal al punto izquierdo
    draw.line((x,top+h1/2,x+ll,top+h1/2),fill=(255,0,0))    

    #Línea horizontal de artículo correcto
    draw.line((x,bottom-h2/2,x+ll,bottom-h2/2),fill=(255,0,0))        

    # Llame a la función para dibujar los nodos de izquierda y derecha    
    drawnode(draw,clust.left,x+ll,top+h1/2,scaling,labels)
    drawnode(draw,clust.right,x+ll,bottom-h2/2,scaling,labels)
  else:   
    # Si esto es un punto final, sacar la etiqueta del elemento
    draw.text((x+5,y-7),labels[clust.id],(0,0,0))

def rotatematrix(data):
  newdata=[]
  for i in range(len(data[0])):
    newrow=[data[j][i] for j in range(len(data))]
    newdata.append(newrow)
  return newdata

import random
###==========================PREGUNTA 4=============================================
###=================================================================================
###=================================================================================
def kcluster(rows,distance=pearson,k=4):
  # Determinar el valor mínimo y máximo para cada punto
  ranges=[(min([row[i] for row in rows]),max([row[i] for row in rows])) 
  for i in range(len(rows[0]))]

  # Crear k centroides colocadas al azar
  clusters=[[random.random()*(ranges[i][1]-ranges[i][0])+ranges[i][0] 
  for i in range(len(rows[0]))] for j in range(k)]
  
  lastmatches=None
  for t in range(100):
    print 'Iteration %d' % t
    bestmatches=[[] for i in range(k)]
    
    # Encuentra que centroide es el más cercano para cada fila
    for j in range(len(rows)):
      row=rows[j]
      bestmatch=0
      for i in range(k):
        d=distance(clusters[i],row)
        if d<distance(clusters[bestmatch],row): bestmatch=i
      bestmatches[bestmatch].append(j)

    # Si los resultados son los mismos que la última vez, esto es completa
    if bestmatches==lastmatches: break
    lastmatches=bestmatches
    
    # Mover los centroides a la media de sus miembros
    for i in range(k):
      avgs=[0.0]*len(rows[0])
      if len(bestmatches[i])>0:
        for rowid in bestmatches[i]:
          for m in range(len(rows[rowid])):
            avgs[m]+=rows[rowid][m]
        for j in range(len(avgs)):
          avgs[j]/=len(bestmatches[i])
        clusters[i]=avgs
      
  return bestmatches
###=================================================================================
###=================================================================================
###=================================================================================
def tanamoto(v1,v2):
  c1,c2,shr=0,0,0
  
  for i in range(len(v1)):
    if v1[i]!=0: c1+=1 # in v1
    if v2[i]!=0: c2+=1 # in v2
    if v1[i]!=0 and v2[i]!=0: shr+=1 # en ambos
  
  return 1.0-(float(shr)/(c1+c2-shr))

def scaledown(data,distance=pearson,rate=0.01):
  n=len(data)

  # Las distancias reales entre cada par de artículos
  realdist=[[distance(data[i],data[j]) for j in range(n)] 
             for i in range(0,n)]

  # Aleatoriamente inicializar los puntos de partida de las ubicaciones en 2D
  loc=[[random.random(),random.random()] for i in range(n)]
  fakedist=[[0.0 for j in range(n)] for i in range(n)]
  
  lasterror=None
  for m in range(0,1000):
    # Encuentra distancias proyectadas
    for i in range(n):
      for j in range(n):
        fakedist[i][j]=sqrt(sum([pow(loc[i][x]-loc[j][x],2) 
                                 for x in range(len(loc[i]))]))
  
    # puntos Move
    grad=[[0.0,0.0] for i in range(n)]
    
    totalerror=0
    for k in range(n):
      for j in range(n):
        if j==k: continue
        # El error es diferencia porcentual entre las distancias
        errorterm=(fakedist[j][k]-realdist[j][k])/realdist[j][k]
        
        # Cada punto tiene que ser alejado de o hacia el otro
        # punto en proporción a la cantidad de errores que tiene
        grad[k][0]+=((loc[k][0]-loc[j][0])/fakedist[j][k])*errorterm
        grad[k][1]+=((loc[k][1]-loc[j][1])/fakedist[j][k])*errorterm

        # No pierda de vista el error total
        totalerror+=abs(errorterm)
    print totalerror

    # Si la respuesta empeoró moviendo los puntos, hemos terminado
    if lasterror and lasterror<totalerror: break
    lasterror=totalerror
    
    # Mueva cada uno de los puntos por los que aprenden veces la tasa del gradiente
    for k in range(n):
      loc[k][0]-=rate*grad[k][0]
      loc[k][1]-=rate*grad[k][1]

  return loc

def draw2d(data,labels,jpeg='mds2d.jpg'):
  img=Image.new('RGB',(2000,2000),(255,255,255))
  draw=ImageDraw.Draw(img)
  for i in range(len(data)):
    x=(data[i][0]+0.5)*1000
    y=(data[i][1]+0.5)*1000
    draw.text((x,y),labels[i],(0,0,0))
  img.save(jpeg,'JPEG')  
