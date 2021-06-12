

import glfw
from OpenGL.GL import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
import math
from copy import deepcopy
from copy import copy

#vertex receives the vertices and the colors then sends colors to fragment
# it sets the position
# uniform is a type that can be set in the main(here python) program
vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
layout(location = 2) in vec3 a_normal;

uniform mat4 mvp; //model view projection (p*v*m)
uniform mat4 model; //matrice model
uniform vec3 light_pos; //position de la source lumineuse
uniform vec3 camPos; //position de la camera

out vec3 v_color;
out vec3 Normal;
out vec3 FragPos;
out vec3 LightPosition;
out vec3 camPosition;

void main()
{
     
     
    
    gl_Position = mvp * vec4(a_position,1.0);
    v_color = a_color;
    Normal = mat3(transpose(inverse(model))) * a_normal;
    FragPos = vec3(model * vec4(a_position,1.0));
    LightPosition = light_pos;
    camPosition = camPos;
}
"""

fragment_src = """
# version 330

in vec3 v_color;
in vec3 Normal;
in vec3 FragPos;
in vec3 LightPosition;
in vec3 camPosition;

out vec4 out_color;



void main()
{
     //normalisation des vecteurs 
     vec3 norm = normalize(Normal);
     vec3 lightDir = normalize(LightPosition - FragPos);
     //couleur de la lumière
     vec3 lightColor = vec3(1,1,1);
     
     //lumière ambiente
     float coeffAmbient = 0.2;
     vec3 light_ambient = coeffAmbient*lightColor;
     
     //lumière diffuse
     float diff = max(dot(norm, lightDir),0.0);
     vec3 light_diffuse = diff*lightColor;
     
     //lumière séculaire
     float coeffSpecular = 0.5;
     vec3 viewDir = normalize(camPosition - FragPos);
     vec3 reflectDir = reflect(-lightDir, norm);
     float spec = pow(max(dot(viewDir, reflectDir),0.0),4);
     vec3 light_specular = coeffSpecular * spec * lightColor;
     
     //résultats de l'ensemble des lumières formant l'illumination de Phongs
     vec3 result = (light_ambient + light_diffuse + light_specular) * v_color;
     
     
 
    out_color = vec4(result, 1.0);
}



"""

class Window:

    def __init__(self, width, height,title):
        #init library
        if not glfw.init():
            raise Exception("cant init glfw")

        #create window
        self._win = glfw.create_window(width, height, title, None, None)

        if not self._win:
            glfw.terminate()
            raise Exception("cant create window")

        #set window position
        glfw.set_window_pos(self._win, 400, 400)

        # allows the resize of the window
        #glfw.set_window_size_callback(self._win, window_resize)

        # make the context current
        glfw.make_context_current(self._win)
        
        self.ProjectionMatrix(width, height)
        
    def ProjectionMatrix(self,width,height):
        fov = 45
        aspect_ratio = width / height
        near_clip = 0.1
        far_clip = 300

        #create a perspective matrix
        self.projection = pyrr.matrix44.create_perspective_projection_matrix(
                fov,
                aspect_ratio,
                near_clip,
                far_clip
                )

        glViewport(0, 0, width, height)
        
    def initViewMatrix(self,eye=[0,0,70],target=[0,0,0]):
        eye=np.array(eye)
        target=np.array(target)
        up=np.array([0,1,0])
        self.ViewMatrix = pyrr.matrix44.create_look_at(eye,target,up)
        
    def render(self,octas,n):
        octas = fractOcta(octas,n)
        self.initViewMatrix(eye = [0,0,(2**(n+1) - 1)*2])
        #print(self.ViewMatrix())
        #color of the window
        glClearColor(0.1, 0.1, 0.1, 1)
        t = 1
        #need it or depth perception is weird
        glEnable(GL_DEPTH_TEST)
        #glDisable(GL_CULL_FACE)
        
        
        while not glfw.window_should_close(self._win):
            glfw.poll_events()
            
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            
            eyeCam = [0,(2**n)*np.cos(0.25*glfw.get_time()),(2**(n+2) - 2)*(np.cos(0.5*glfw.get_time())-2)]
            self.initViewMatrix(eye = eyeCam)
            v=self.ViewMatrix
            v = np.matmul(pyrr.matrix44.create_from_y_rotation(0.5*glfw.get_time()),v)
            p=self.projection
            vp=np.matmul(v,p)
            #adapte la projection si la taille de la fenêtre change
            w, h = glfw.get_framebuffer_size(self._win)
            self.ProjectionMatrix(w,h)
           
            for o in octas:
                light_pos = [2**n,(2**n+1)*np.sqrt(2) - np.sqrt(2),0]
                mvp = np.matmul(o.modelMatrix, vp)
                mvp = np.matmul( pyrr.matrix44.create_from_translation([0,(2**n - 1)*np.sqrt(2),0]),mvp)
                o.Shader.draw(mvp, light_pos,o.modelMatrix,eyeCam)
                
            glfw.swap_buffers(self._win)
            
        self.CloseWindow()

        
    def CloseWindow(self):
        glfw.terminate()
        
        
class Octaedre:
    
    def __init__(self):
        self.modelMatrix = pyrr.matrix44.create_identity()
        
        
        self.vertices = [ 1, 0.0, 1, 0.6, 0.196078, 0.8, np.sqrt(2),0.0,np.sqrt(2),
                          1, 0.0, -1, 0.6, 0.196078, 0.8, np.sqrt(2),0.0,-np.sqrt(2),
                         -1, 0.0, -1, 0.6, 0.196078, 0.8, -np.sqrt(2),0.0,-np.sqrt(2),
                         -1, 0.0, 1, 0.6, 0.196078, 0.8, -np.sqrt(2),0.0,np.sqrt(2),
                          0.0, np.sqrt(2), 0.0, 1, 0, 0, 0.0,1.0,0.0,
                          0.0, -np.sqrt(2), 0.0, 1, 0, 0, 0.0,-1.0,0.0]

        self.indices = [0,4,1,
                        1,4,2,
                        2,4,3,
                        3,4,0,
                        0,5,1,
                        1,5,2,
                        2,5,3,
                        3,5,0]
        
        
        self.Shader=colorShader(self.vertices,self.indices)

    
        
    #permet de une translation de l'octaèdre
    def doTranslation(self,vecteur):
        self.modelMatrix = np.matmul(pyrr.matrix44.create_from_translation(vecteur), self.modelMatrix)

class colorShader:
    
    def __init__(self,vertices,indices):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices =np.array(indices, dtype=np.uint32)
        self.shader = self.createShader(vertex_src,fragment_src)
        self.createBuffers()
        glUseProgram(self.shader)
    
    #création du shader
    def createShader(self, vs, fs):
        return OpenGL.GL.shaders.compileProgram(compileShader(vs, GL_VERTEX_SHADER),
                       compileShader(fs, GL_FRAGMENT_SHADER))
     #création des buffers   
    def createBuffers(self):
        
        #create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        #create 1 VBO buffer
        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        # Send data to buffer //each element 4 bytes float vertices.nbytes : len(vertices) * 4
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices,GL_STATIC_DRAW)
        
        # Element Buffer Object : the link between vertices
        EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices,GL_STATIC_DRAW)
        
        #position = glGetAttribLocation(shader, "a_position") DONT NEED IT ANYMOER WHEN ADD layout(location = n) in vertex shader
        glEnableVertexAttribArray(0)
        #The last arg is where you start in the buffer (vertices array)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(0))

        #color = glGetAttribLocation(shader, "a_color")
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(12))
        
        #normal = glGetAttribLocation(shader, "a_normal")
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(24))
        
    def draw(self, mvp, light_pos, model,eyeCam):
        
        #relation entre la matrice uniform mvp de glsl avec la matrice mvp de openGL python
        transformLoc = glGetUniformLocation(self.shader, "mvp")
        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, mvp)
        #relation entre le vecteur de la position de la lumière uniform de glsl avec celui  de openGL python
        transformLoc2 = glGetUniformLocation(self.shader, "light_pos")
        glUniform3fv(transformLoc2, 1, light_pos)
         #relation entre la matrice model uniform de glsl avec celle de openGL python
        transformLoc3 = glGetUniformLocation(self.shader, "model")
        glUniformMatrix4fv(transformLoc3, 1,GL_FALSE, model)
        #relation entre le vecteur de la position de la camera uniform de glsl avec celui de openGL python
        transformLoc4 = glGetUniformLocation(self.shader, "camPos")
        glUniform3fv(transformLoc4, 1, eyeCam)
        
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

def main():
    win = Window(700,700,"fenetre")
    
    
    octa = Octaedre()
    octas = [octa]

    
    win.render(octas,3)
    
#fonction fractal récursive générant et positionnant les différents octaèdres formant l'objet mathématique      
def fractOcta(Oc,n):
    if n == 0 :
        return Oc
    else:
        Oc = fractOcta(Oc, n-1) #appel récursif
        #on récupère le nombe d'octaèdre de l'étape n-1 pour placer ceux de l'étape n
        a = 2**n
        b = 2**(n-1)
        l = len(Oc) #taille de la liste contenant les octaèdres
        
        for i in range(l):
            #on effectue les différentes translations a effectué aux octaèdres à l'étape n pour former
            #notre objet mathématique
            tmp = copy(Oc[i])
            tmp.doTranslation([b,-b*np.sqrt(2),b])
           
            Oc.append(tmp)
            
            tmp = copy(Oc[i])
            tmp.doTranslation([b,-b*np.sqrt(2),-b])
           
            Oc.append(tmp)

            tmp = copy(Oc[i])
            tmp.doTranslation([-b,-b*np.sqrt(2),b])
           
            Oc.append(tmp)

            tmp = copy(Oc[i])
            tmp.doTranslation([-b,-b*np.sqrt(2),-b])
            
            Oc.append(tmp)

            tmp = copy(Oc[i])
            tmp.doTranslation([0,-a*np.sqrt(2),0])
            
            Oc.append(tmp)

        return Oc     
        

if __name__ == "__main__":
    main()