# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:38:31 2021

@author: jeome
"""

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

uniform mat4 mvp; //model view projection (p*v*m)


out vec3 v_color;

void main()
{
    gl_Position = mvp * vec4(a_position,1.0);
    v_color = a_color;
}
"""

fragment_src = """
# version 330

in vec3 v_color;

out vec4 out_color;

void main()
{
    out_color = vec4(v_color, 1.0);
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
        glfw.set_window_pos(self._win, 200, 200)

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
        
    def initViewMatrix(self,eye=[0,0,2]):
        eye=np.array(eye)
        target=np.array([0,0,0])
        up=np.array([0,1,0])
        self.ViewMatrix = pyrr.matrix44.create_look_at(eye,target,up)
        
    def render(self,octas):
        #self.initViewMatrix()
        #color of the window
        glClearColor(0.1, 0.1, 0.1, 1)

        #need it or depth perception is weird
        glEnable(GL_DEPTH_TEST)
        #glDisable(GL_CULL_FACE)
        
        v=self.ViewMatrix
        p=self.projection
        vp=np.matmul(v,p)
        
        while not glfw.window_should_close(self._win):
            glfw.poll_events()
            
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            
            #adapte la projection si la taille de la fenêtre change
            w, h = glfw.get_framebuffer_size(self._win)
            self.ProjectionMatrix(w,h)
            for o in octas:
                o.actualizeRotation()
                o.actualizeModel()
                mvp = np.matmul(o.modelMatrix, vp)
                o.Shader.draw(mvp)
                
            glfw.swap_buffers(self._win)
            
        self.CloseWindow()

        
    def CloseWindow(self):
        glfw.terminate()
        
        
class Octaedre:
    
    def __init__(self):
        self.rotationMatrix = pyrr.matrix44.create_identity()
        self.actualizeTranslation((0,0,0))
        self.actualizeScale((1,1,1))

        self.modelMatrix = pyrr.matrix44.create_identity()
        self.translationMatrix = pyrr.matrix44.create_identity()
        self.rotationMatrix = pyrr.matrix44.create_identity()
        self.ScaleMatrix = pyrr.matrix44.create_identity()
        
        self.vertices = [ 1, 0.0, 1, 1.0, 0.0, 0.0,
                          1, 0.0, -1, 0.0, 1.0, 0.0,
                         -1, 0.0, -1, 0.0, 0.0, 1.0,
                         -1, 0.0, 1, 1.0, 1.0, 0.0,
                          0.0, np.sqrt(2), 0.0, 1.0, 1.0, 1.0,
                          0.0, -np.sqrt(2), 0.0, 1.0, 0.0, 1.0]

        self.indices = [0,4,1,
                        1,4,2,
                        2,4,3,
                        3,4,0,
                        0,5,1,
                        1,5,2,
                        2,5,3,
                        3,5,0]
        
        
        self.Shader=colorShader(self.vertices,self.indices)

    def changeTranslation(self,translationmMat):
        self.translationMatrix = translationmMat
    
    def actualizeTranslation(self, vecteur):
        self.translationMatrix = pyrr.matrix44.create_from_translation(vecteur)
        self.vecteurT = vecteur
        
    def ajoutTranslation(self,vecteur):
            V = [self.vecteurT[0]+vecteur[0],self.vecteurT[1]+vecteur[1],self.vecteurT[2]+vecteur[2]]
            self.actualizeTranslation(V)
        
    def actualizeScale(self, k):
        self.ScaleMatrix = pyrr.matrix44.create_from_scale(k)
    
    def actualizeRotation(self):
        #rotx = pyrr.matrix44.create_from_x_rotation(0.5 * glfw.get_time())
        rotx = pyrr.matrix44.create_identity()
        roty = pyrr.matrix44.create_from_y_rotation(0.08 * glfw.get_time())
        self.rotationMatrix = pyrr.matrix44.multiply(rotx,roty)

    def Rotation(self,x,y,z):
        #rotx = pyrr.matrix44.create_from_x_rotation(0.5 * glfw.get_time())
        rotx = pyrr.matrix44.create_from_x_rotation(x)
        roty = pyrr.matrix44.create_from_y_rotation(y)
        rotz = pyrr.matrix44.create_from_z_rotation(z)
        rot = np.matmul(rotx,np.matmul(roty,rotz))
        if self.rotationMatrix is not None:
            self.rotationMatrix = np.matmul(self.rotationMatrix , rot)
        else:
            self.rotationMatrix = rot

    def actualizeModel(self):
        self.modelMatrix=np.matmul(self.translationMatrix,np.matmul(self.ScaleMatrix,self.rotationMatrix))

class colorShader:
    
    def __init__(self,vertices,indices):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices =np.array(indices, dtype=np.uint32)
        self.shader = self.createShader(vertex_src,fragment_src)
        self.createBuffers()
        glUseProgram(self.shader)
        
    def createShader(self, vs, fs):
        return OpenGL.GL.shaders.compileProgram(compileShader(vs, GL_VERTEX_SHADER),
                       compileShader(fs, GL_FRAGMENT_SHADER))
        
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
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

        #color = glGetAttribLocation(shader, "a_color")
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        
    def draw(self, mvp):
        
        transformLoc = glGetUniformLocation(self.shader, "mvp")
        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, mvp)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

def main():
    win = Window(700,700,"fenetre")
    win.initViewMatrix(eye=[0,-20,40])
    
    #if not win.Window:
    #    return
    octa = Octaedre()
    #octa2 = Octaedre()
    #octa2.actualizeTranslation([0,-2*np.sqrt(2),0])
    #octa3 = Octaedre()
    #octa3.actualizeTranslation([1,-np.sqrt(2),1])
    #octas = [octa,octa2,octa3]
    octas = [octa]
    #octas = fractOcta(octas,2)
    octas = fractOcta2(octas,3,3)
    a = (1,2)
    #print(a[1]+a[1])
    win.render(octas)
    

def fractOcta(Oc,n):
    if n == 0 :
        return Oc
    else:
        Oc = fractOcta(Oc, n-1)
        o0 = deepcopy(Oc)
       #print(o0)
        a = 2**n
        b = 2**(n-1)
        for i in range(1,6,1):
            oi = deepcopy(o0)
            #print(oi)
            if i == 1:
                for o in oi:
                   
                    o.ajoutTranslation([b,-b*np.sqrt(2),b])
                    o.actualizeModel()
                   
                    Oc.append(o)
                
            elif i==2:
                for o in oi:
                    
                    o.ajoutTranslation([b,-b*np.sqrt(2),-b])
                    o.actualizeModel()
                    
                    Oc.append(o)
                
            elif i==3:
                for o in oi:
                   
                    o.ajoutTranslation([-b,-b*np.sqrt(2),b])
                    o.actualizeModel()
                    
                    Oc.append(o)
                
            elif i==4:
                for o in oi:
                   
                    o.ajoutTranslation([-b,-b*np.sqrt(2),-b])
                    o.actualizeModel()
                    
                    Oc.append(o)
               
            else:
                for o in oi:
                    
                    o.ajoutTranslation([0,-a*np.sqrt(2),0])
                    o.actualizeModel()
                    
                    Oc.append(o)
        
        return Oc
        
def fractOcta2(Oc,n,s):
    if n == 0 :
        return Oc
    else:
        Oc = fractOcta2(Oc, n-1,s)
        a = 2**n
        b = 2**(n-1)
        l = len(Oc)
        for i in range(l):

            tmp = copy(Oc[i])
            tmp.ajoutTranslation([b,-b*np.sqrt(2),b])
            tmp.actualizeModel()
            Oc.append(tmp)
            
            tmp = copy(Oc[i])
            tmp.changeTranslation(Oc[i].translationMatrix)
            tmp.ajoutTranslation([b,-b*np.sqrt(2),-b])
            tmp.actualizeModel()
            Oc.append(tmp)

            tmp = copy(Oc[i])
            tmp.changeTranslation(Oc[i].translationMatrix)
            tmp.ajoutTranslation([-b,-b*np.sqrt(2),b])
            tmp.actualizeModel()
            Oc.append(tmp)

            tmp = copy(Oc[i])
            tmp.changeTranslation(Oc[i].translationMatrix)
            tmp.ajoutTranslation([-b,-b*np.sqrt(2),-b])
            tmp.actualizeModel()
            Oc.append(tmp)

            tmp = copy(Oc[i])
            tmp.changeTranslation(Oc[i].translationMatrix)
            tmp.ajoutTranslation([0,-a*np.sqrt(2),0])
            tmp.actualizeModel()
            Oc.append(tmp)

        return Oc     
        

if __name__ == "__main__":
    main()