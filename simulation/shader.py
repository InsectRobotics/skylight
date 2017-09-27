#!/usr/bin/env python

''' Description:
This module loads and compiles OpenGL shader programs

# Link our shader program - "the linking
#step involves making the connections between the input variables from one shader to the
#output variables of another, and making the connections between the other input/output
#variables of a shader to appropriate locations in the OpenGL environment."
#program = link_shader_program(program)

__author__ = 'Jan Stankiewicz'
__date__ = '14/06/2016'
__credits__ = 'blank'


'''

import definitions
from os import path
from OpenGL.GL import *     # todo: load only relevant parts here?
from OpenGL.GL.shaders import *

# Todo: work out why importing wgl causes parent script to throw error
#from OpenGL import WGL as wgl


class ShaderClass(object):

    def __init__(self,
                 vertex_shader_name,
                 fragment_shader_name,
                 geometry_shader_name = None,
                 file_path=definitions.SHADER_PATH,#"../compound_eye/shaders"
                 ):

        self.vertex_shader_name = vertex_shader_name
        self.fragment_shader_name = fragment_shader_name
        self.geometry_shader_name = geometry_shader_name

        # assert wgl.wglGetCurrentContext() != None, 'No OpenGL context is open. You must create a context before making a shader class'

        v_file = path.join(file_path, vertex_shader_name)
        f_file = path.join(file_path, fragment_shader_name)

        vertex_string = open(v_file, 'r')
        fragment_string = open(f_file, 'r')


        vertex_shader = self.compile_vertex_shader(vertex_string)

        fragment_shader = self.compile_fragment_shader(fragment_string)

        #print 'geo shader is called: ', self.geometry_shader_name

        if geometry_shader_name is not None:
            g_file = path.join(file_path, geometry_shader_name)
            geometry_string = open(g_file, 'r')
            geometry_shader = self.compile_geometry_shader(geometry_string)
            self.program = self.create_shader_program(vertex_shader, fragment_shader,geometry_shader)

        else:
            self.program = self.create_shader_program(vertex_shader, fragment_shader)

    def compile_vertex_shader(self,source):
        """Compile a vertex shader from source."""
        #print source
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, source)
        glCompileShader(vertex_shader)
        # check compilation error
        result = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
        if not(result):
            raise RuntimeError(glGetShaderInfoLog(vertex_shader))
        return vertex_shader

    def compile_geometry_shader(self,source):
        """Compile a vertex shader from source."""
        geometry_shader = glCreateShader(GL_GEOMETRY_SHADER)
        glShaderSource(geometry_shader, source)
        glCompileShader(geometry_shader)
        # check compilation error
        result = glGetShaderiv(geometry_shader, GL_COMPILE_STATUS)
        if not(result):
            raise RuntimeError(glGetShaderInfoLog(geometry_shader))
        return geometry_shader

    def compile_fragment_shader(self,source):
        """Compile a vertex shader from source."""
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, source)
        glCompileShader(fragment_shader)
        # check compilation error
        result = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
        if not(result):
            raise RuntimeError(glGetShaderInfoLog(fragment_shader))
        return fragment_shader

    def create_shader_program(self,vertex_shader, fragment_shader, g_shader = None):
        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        if type(g_shader) is long:
            glAttachShader(program, g_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)
        # check linking error
        result = glGetProgramiv(program, GL_LINK_STATUS)
        if not(result):
            raise RuntimeError(glGetProgramInfoLog(program))
        return program

    def link_shader_program(self,program):
        """Create a shader program with from compiled shaders."""
        glLinkProgram(program)
        # check linking error
        result = glGetProgramiv(program, GL_LINK_STATUS)
        if not(result):
            raise RuntimeError(glGetProgramInfoLog(program))
        return program

# if __name__ == "__main__":
#
#     'test class initialisation'
#     pygame.init()
#     pygame.display.set_mode((1080,720), pygame.DOUBLEBUF | pygame.OPENGL)
#     test = ShaderClass('passthrough.vert','passthrough.frag')


class ComputeClass(object):

    def __init__(self,
                 compute_shader_name,
                 file_path= definitions.SHADER_PATH, #"../compound_eye/shaders",
                 shader_string=None,
                 ):

        self.compute_shader_name = compute_shader_name

        # cs_file = path.join(file_path, compute_shader_name)
        # cs_string = open(cs_file, 'r')
        # print file_path

        if shader_string is None:
            cs_file = path.join(file_path, compute_shader_name)
            cs_string = open(cs_file, 'r')
           # print file_path

        else:
            cs_string = shader_string

        self.program = glCreateProgram()

        # create the shader and add the shader string, then compile
        compute_shader = glCreateShader(GL_COMPUTE_SHADER)
        glShaderSource(compute_shader,cs_string)
        glCompileShader(compute_shader)

        # check for compile errors
        result = glGetShaderiv(compute_shader, GL_COMPILE_STATUS)
        if not result:
            raise RuntimeError(glGetShaderInfoLog(compute_shader))

        # attach shader to the program and link
        glAttachShader(self.program,compute_shader)
        glLinkProgram(self.program)

        # check for link errors
        result = glGetProgramiv(self.program, GL_LINK_STATUS)
        if not(result):
            raise RuntimeError(glGetProgramInfoLog(self.program))

        # set program as active
        glUseProgram(self.program)