#vec3 iResolution = vec3(WIDTH, HEIGHT, 0);

import glfw
from OpenGL.GL import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
import math

#vertex receives the vertices and the colors then sends colors to fragment
# it sets the position
# uniform is a type that can be set in the main(here python) program
vertex_src = """
# version 330

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec2 in_texcoord;

out vec2 fragCoord;

void main()
{
    gl_Position = vec4(in_position.x, in_position.y, in_position.z, 1.0); 
    fragCoord = in_texcoord;
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
        
        #self.ProjectionMatrix(width, height)
        
        
    def render(self,screen):
        #self.initViewMatrix()
        #color of the window
        glClearColor(0.1, 0.1, 0.1, 1)

        #need it or depth perception is weird
        glEnable(GL_DEPTH_TEST)
        #glDisable(GL_CULL_FACE)
 
        while not glfw.window_should_close(self._win):
            glfw.poll_events()
            
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            
            #adapte la projection si la taille de la fenêtre change
            w, h = glfw.get_framebuffer_size(self._win)
            
        

            screen.Shader.draw()

            glfw.swap_buffers(self._win)
            
        self.CloseWindow()

        
    def CloseWindow(self):
        glfw.terminate()
        
        
class Screen:
    
    def __init__(self):
        self.vertices = [ -1,1,0,
                           1,1,0,
                           -1,-1,0,
                           1,-1,0
        ]
        self.tex = [ 0.0, 1.0,
                     1.0, 1.0,
                     0.0, 0.0,
                     1.0, 0.0 
        ]

        self.indices = [0,1,2,1,2,3]
        
        self.Shader=colorShader(self.vertices, self.tex, self.indices)
        #print(self.Shader.vertices)



class colorShader:
    
    def __init__(self,vertices, tex, indices):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.tex = np.array(tex, dtype=np.float32)
        self.indices =np.array(indices, dtype=np.uint32)
        self.shader = self.createShader(vertex_src,fragment_src)
        self.createBuffers()
        glUseProgram(self.shader)
        
    def createShader(self, vs, fs):
        fs_file=open('fs.txt','r')
        FRAGMENT_SHADER = fs_file.read()
        fs_file.close()
        return OpenGL.GL.shaders.compileProgram(compileShader(vs, GL_VERTEX_SHADER),
                       compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER))
        
    def createBuffers(self):
        
        #create VAO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
        
        #create 1 VBO buffer
        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        # Send data to buffer //each element 4 bytes float vertices.nbytes : len(vertices) * 4
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices,GL_STATIC_DRAW)

        #create 1 VBO buffer
        VBO2 = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO2)
        # Send data to buffer //each element 4 bytes float vertices.nbytes : len(vertices) * 4
        glBufferData(GL_ARRAY_BUFFER, self.tex.nbytes, self.tex,GL_STATIC_DRAW)
        
        # Element Buffer Object : the link between vertices
        EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices,GL_STATIC_DRAW)
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        #position = glGetAttribLocation(shader, "a_position") DONT NEED IT ANYMOER WHEN ADD layout(location = n) in vertex shader
        glEnableVertexAttribArray(0)
        #The last arg is where you start in the buffer (vertices array)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO2)
        #color = glGetAttribLocation(shader, "a_color")
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        
    def draw(self):

        SCALE = 2.0
        C_X = 1.0
        C_Y = 0.0
        C_Z = 0.0

        ROT1_X = 0.0 #20.0 * np.sin(glfw.get_time())
        ROT1_Y = 0.0 #60.0
        ROT1_Z = 0.0

        ROT2_X = 0.0 #20.0 * np.sin(glfw.get_time())
        ROT2_Y = 0.0 #20.0
        ROT2_Z = 0.0


        Loc = glGetUniformLocation(self.shader, "SCALE")
        glUniform1f(Loc, SCALE)
        Loc = glGetUniformLocation(self.shader, "C_X")
        glUniform1f(Loc, C_X)
        Loc = glGetUniformLocation(self.shader, "C_Y")
        glUniform1f(Loc, C_Y)
        Loc = glGetUniformLocation(self.shader, "ROT1_X")
        glUniform1f(Loc, ROT1_X)
        Loc = glGetUniformLocation(self.shader, "ROT1_Y")
        glUniform1f(Loc, ROT1_Y)
        Loc = glGetUniformLocation(self.shader, "ROT1_Z")
        glUniform1f(Loc, ROT1_Z)
        Loc = glGetUniformLocation(self.shader, "ROT2_X")
        glUniform1f(Loc, ROT2_X)
        Loc = glGetUniformLocation(self.shader, "ROT2_Y")
        glUniform1f(Loc, ROT2_Y)
        Loc = glGetUniformLocation(self.shader, "ROT2_Z")
        glUniform1f(Loc, ROT2_Z)

        




        resLoc = glGetUniformLocation(self.shader, "Resolution")
        glUniform2fv(resLoc, 1, (700.0,700.0))
        
        time = glfw.get_time()
        timeLoc = glGetUniformLocation(self.shader, "time")
        glUniform1f(timeLoc, time)

        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

def main():
    win = Window(700,700,"fenetre")

    sc = Screen()
    win.render(sc)
      
        

if __name__ == "__main__":
    main()