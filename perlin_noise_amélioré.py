# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math
try :
    import pygame
except :
    pass
import sys
import os
import imageio
import numpy as np


noir = (0,0,0)
blanc = (255,255,255)
gris_fonce = (100,100,100)
gris_clair = (200,200,200)
bleu_fonce = (0,0,129)
bleu_demi_clair = (0,0,200)
bleu_clair = (0,0,255)

vert_clair = (31,225,113)
vert_fonce = (31,129,113)
jaune = (225,218,85)



nb = 1000003 #un nombre premier
u = [1]
a = 14125421
b = 1
m = 100000000
for k in range(nb-1):
    u.append((a*u[-1]+b)%m)

def triplet(l):
    #l est une liste à 3 éléments
    return l[0],l[1],l[2]

def gen(seed):
    return(u[seed%nb]/m)


def vecteur(A,B):
    return([B[i]-A[i] for i in range(len(A))])

def pscalaire(v1,v2):
    if len(v1) != len(v2):
        print("Les deux vecteurs ne sont pas de la même taille !")
        print(v1,v2)
    return sum([v1[i]*v2[i] for i in range(len(v1))])

def somme(v1,v2):
    return [x+y for x,y in zip(v1,v2)]
    

def produit(t,v):
    return([t*x for x in v])

def norme(v):
    return np.sqrt(pscalaire(v,v))

def unitaire(v):
    return produit(1/norme(v),v)

def pvectoriel(u,v):
    x,y,z = u[0],u[1],u[2]
    xx,yy,zz= v[0],v[1],v[2]
    return [y*zz - yy*z , z*xx - zz*x, x*yy - xx*y]


def unitaire_aleatoire(seed):
    theta = 2*np.pi*gen(seed)
    return [np.cos(theta),np.sin(theta)]
    
def vecteur_normal(courbe,point,h=0.001):
    x,y = point[0],point[1]
    dfdx = (courbe([x+h,y])-courbe([x-h,y]))/(2*h)
    dfdy = (courbe([x,y+h])-courbe([x,y-h]))/(2*h)
    return(unitaire([-dfdx,-dfdy,1]))

def injection(i,j):
    return (math.floor(j+(((i+j)*(i+j+1))/2)))

def polynome(t):
    return(3*t**2-2*t**3)

def interpolation(a1,a2,t):
    return(a1+polynome(t)*(a2-a1))
    
    
def perlin_noise(seed,wave_length,taille):
    #wave_length correspont à la distance entre deux "points" de la grille de gradient.
    #taille est la taille de la grille que l'on veut calculer. On est obligé de la spécifier 
    #car les calculs doivent bien s'arrêter. 
    #print(28*"-"+"precalcul"+28*"-")

    """
    tab_gradient = [[0 for _ in range(taille)] for _ in range(taille)]
    cpt = 0
    for i in range(taille):
        for j in range(taille):
            tab_gradient[i][j] = unitaire_aleatoire(seed*injection(i,j))
            cpt += 1
            #if cpt%taille == 0:
                #print("progression : {}%".format(int(100*cpt/(taille**2))))
    """
   
    def res(x,y):
        ix0 = int(x/wave_length)
        x0 = ix0*wave_length
        ix1 = ix0 + 1
        x1 = ix1*wave_length
        iy0 = int(y/wave_length)
        y0 = iy0*wave_length
        iy1 = iy0 + 1
        y1 = iy1*wave_length
        
        g00 = unitaire_aleatoire(seed*injection(ix0,iy0))
        g10 = unitaire_aleatoire(seed*injection(ix1,iy0))
        g01 = unitaire_aleatoire(seed*injection(ix0,iy1))
        g11 = unitaire_aleatoire(seed*injection(ix1,iy1))

        bg = [x0,y0]
        bd = [x1,y0]
        hg = [x0,y1]
        hd = [x1,y1]

            
        gradient_bg = g00
        gradient_bd = g10
        gradient_hg = g01
        gradient_hd = g11
                        
        v = x,y
            
        sbg = pscalaire(vecteur(bg,v),gradient_bg)
        sbd = pscalaire(vecteur(bd,v),gradient_bd)
        shg = pscalaire(vecteur(hg,v),gradient_hg)
        shd = pscalaire(vecteur(hd,v),gradient_hd)
            
        tx = (x-x0)/wave_length
        ty = (y-y0)/wave_length
        interpb = interpolation(sbg,sbd,tx)
        interph = interpolation(shg,shd,tx)
            
        maximum = wave_length / np.sqrt(2)
            
        return ((interpolation(interpb,interph,ty)/maximum)+1)/2.
    
    return res




tab_couleur = [bleu_fonce,bleu_fonce,bleu_demi_clair,jaune,vert_clair,vert_fonce,gris_fonce,gris_clair,blanc,blanc]
def couleur(valeur):
    return tab_couleur[min(9,math.floor(10*valeur))]


def generate(seed,taille_map,wave_length):
    #on pourra prendre par exemple taille_map = 300 et wave_length = [100,50,25]

     
    pas = 1. #la taille d'une case
    
    continents = perlin_noise(4*seed,wave_length[0])
    isthmes = perlin_noise(2*seed,wave_length[1])
    rochers = perlin_noise(seed,wave_length[2])

    bruit = lambda x,y : (4*continents(x,y) + 2*isthmes(x,y)+rochers(x,y)) /7.


    tab = [[0. for _ in range(taille_map)] for _ in range(taille_map)]


    def transformation(x):
        return (polynome(x))**2


    print(30*"-"+"calcul"+30*"-")
    cpt = 0
    for ix in range(taille_map):
        for iy in range(taille_map):
            tab[taille_map-1-iy][ix] = transformation(bruit(pas*ix,pas*iy))
            cpt += 1
            #if cpt%taille_map == 0 :
                #print("progression : {} %".format(int(100*cpt/(taille_map**2))))

    return tab


def afficher_map(carte,taille_fen = 600,fermer=True):

    taille_map = len(carte)
    dim_fen = taille_fen,taille_fen
    taille_case = int(taille_fen/taille_map)
    
    pygame.init()
    fen = pygame.display.set_mode(dim_fen)

    for i in range(taille_map):
        for j in range(taille_map):
            case = pygame.Rect(j*taille_case,i*taille_case,taille_case,taille_case)
            pygame.draw.rect(fen,couleur(carte[i][j]),case)
            pygame.display.flip()

    if fermer :
        pygame.quit()
        sys.exit()
    else :
        launched = True
        while launched :
            for event in pygame.event.get():
                if event.type == pygame.QUIT :
                    pygame.quit()
                    sys.exit()
                    launched = False



def sauver_image(carte,nom):
    taille_map = len(carte)
    img = np.array([[couleur(carte[i][j]) for j in range(taille_map)] for i in range(taille_map)])
    imageio.imwrite("images/"+nom+".png",img)



def view(seed,wave_length,radius,n=100,pointview=[0.,0.,60.], pointsun = [0,0,-1], vectorview= [0,1,-1.],view_precision=100,distanceecran=0.5,taille_fen=600,show_image=True,save_image=False,name="img"):
    #n est la taille de l'écran en nombre de cases. On suppose que l'écran a une taille de 1m.
    #viewprecision est la précision pour le calcul des intersections.                                                                                       

    altmax = 50.
    taille_ecran = 1.
    pas = 1. #la taille d'une case
    
    continents = perlin_noise(4*seed,wave_length[0],radius+wave_length[0])
    isthmes = perlin_noise(2*seed,wave_length[1],radius+wave_length[0])
    rochers = perlin_noise(seed,wave_length[2],radius+wave_length[0])
    
    def transformation(x):
        return (polynome(x))**2
    
    bruit = lambda r : altmax*transformation((4*continents(r[0],r[1]) + 2*isthmes(r[0],r[1])+rochers(r[0],r[1])) /7.)

    u = unitaire(vectorview)

    vx = [1,0,0]
    vy = [0,1,0]
    vz = [0,0,1]

    proj = somme(u,produit(-pscalaire(u,vz),vz))
    theta = np.arccos(pscalaire(proj,vx)/norme(proj)) #On ne pourra pas couvrir tout l'espace par la vision ! (a améliorer ...)
    vxx = [-np.sin(theta),np.cos(theta),0]
    vyy = pvectoriel(vxx,u)

    pas = taille_ecran / n

    mat = [[blanc for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            print(i,j)
            uu = somme(produit(distanceecran,vectorview),somme(produit(-taille_ecran/2 + (j+1/2)*pas,vxx),produit(taille_ecran/2 - (i+1/2)*pas,vyy)))
            u = unitaire(uu)

            indice = -1
            for k in range(view_precision+1):
                viewpos = somme(pointview,produit(k*radius/view_precision,u))
                if bruit(viewpos[:2]) > viewpos[2] :
                    indice = k
                    break

            if indice == -1 :
                mat[i][j] = bleu_clair
            else : 
                point = somme(pointview,produit(indice*radius/view_precision,u))
                vnormal = unitaire(vecteur_normal(bruit,point))
                vsoleil = unitaire(vecteur(point,pointsun))
                basiccolor = couleur(bruit(point[:2])/altmax)

                if basiccolor == bleu_fonce or basiccolor == bleu_demi_clair :
                    mat[i][j] = basiccolor
                else :
                    mat[i][j] = triplet(produit(abs(pscalaire(vnormal,vsoleil)),list(basiccolor)))

    dim_fen = taille_fen,taille_fen
    taille_case = int(taille_fen/n)

    
    if save_image :
        imageio.imwrite("saves/{}.png".format(name),np.array(mat))

    if show_image : 
        pygame.init()
        fen = pygame.display.set_mode(dim_fen)

        for i in range(n):
            for j in range(n):
                case = pygame.Rect(j*taille_case,(n-1-i)*taille_case,taille_case,taille_case)
                pygame.draw.rect(fen,mat[i][j],case)
        pygame.display.flip()

        
        launched = True
        while launched :
            for event in pygame.event.get():
                if event.type == pygame.QUIT :
                    pygame.quit()
                    sys.exit()
                    launched = False



for i in range(10,20):

    pos = [120.,120.,60.]
    seed = i
    taille_map = 200
    wave_length = [60,20,10]
    n=200
    radius = 200
    view_precision = 50

    view(seed,wave_length,radius,n,pointview=pos,view_precision=view_precision,show_image=False,save_image=True,name="img{}".format(str(i)))



#carte = generate(seed,taille_map,wave_length)
#afficher_map(carte,fermer=False)

#nom = "testimage5"
#sauver_image(carte,nom)
