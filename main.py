import cpu_proj
import gpu_proj



def start():
    x = int(input("Taper 1 pour le rendu côté CPU et taper 2 pour le rendu côté GPU : "))
    if (x==1):
        n = int(input("nombre d'itération n = "))
        win =cpu_proj.Window(700,700,"fenetre")
        
    
        octa = cpu_proj.Octaedre()
        octas = [octa]

    
        win.render(octas,n)
    elif (x==2):
        win = gpu_proj.Window(700,700,"fenetre")

        sc = gpu_proj.Screen()
        win.render(sc)
    else:
        pass
        
start()