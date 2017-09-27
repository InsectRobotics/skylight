from OpenGL.GL import *
from PIL.Image import fromarray

import numpy as np
import os


class CubeFrameBuffer(object):
    ''' this class is used to store a framebuffer object intended to store a cubemap used to map the environment of an
    opengl mesh.
    Note. A seperate texture is required for each attachment (e.g. colour, depth, stencil). Since we are using a cubemap
    this must have 6 layers or GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS will be called
    '''
    # todo - can pyopengl VBO be used more here?
    # todo - cater for both RGB and RGBA configurations
    # todo - add frame buffer error code module

    def __init__(self,
                 cube_resolution,
                 faces,
                 texture_name = None,
                 ):

        ####################################################### setup frame buffer
        self.fb_name = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fb_name)

        ####################################################### setup colour texture
        if texture_name is None:
            self.texture_name = glGenTextures(1)
        else:
            self.texture_name = texture_name

        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texture_name)
        #
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        #
        # todo - how about higher colour ranges here?
        dummy = np.zeros((cube_resolution*cube_resolution,4),dtype=np.byte) * 0.5


        for face in faces:
            glTexImage2D(face,
                         0,
                         GL_RGBA8,                                    # best to specifiy precision explicitlty here, otherwise the driver will choose this
                         cube_resolution,cube_resolution,
                         0,
                         GL_RGBA,
                         GL_UNSIGNED_BYTE,
                         dummy)

        #glFramebufferTexture2d is the other option but need to do each cubemap face?
        glFramebufferTexture(GL_FRAMEBUFFER,                      # 1. fbo target: GL_FRAMEBUFFER (existing/bound framebuffer)
                             GL_COLOR_ATTACHMENT0,                  # 2. attachment point
                             self.texture_name,                              # texture ID
                             0)                                             # mipmap level of texture to attach.



        # ####################################################### setup depth texture - use 2D array
        # have kept this as could be useful for cases when not using a cubemap
        # self.texture2 = glGenTextures(1)
        # #glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture2)
        # glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture2)
        # glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        # glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        # glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        # glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        # glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        #
        # glTexImage3D(GL_TEXTURE_2D_ARRAY,
        #     0,
        #     GL_DEPTH_COMPONENT24,
        #     cube_resolution,
        #     cube_resolution,
        #     6,
        #     0,
        #     GL_DEPTH_COMPONENT,
        #     GL_UNSIGNED_INT,
        #     None)
        #
        # glFramebufferTexture(GL_FRAMEBUFFER,
        #     GL_DEPTH_ATTACHMENT,
        #     self.texture2,
        #     0)

         # ####################################################### setup depth texture - use cubemap
        self.texture2 = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texture2)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

        for face in faces:
            glTexImage2D(face,
                         0,
                         GL_DEPTH_COMPONENT24,                                    # best to specifiy precision explicitlty here, otherwise the driver will choose this. todo What size to use though
                         cube_resolution,cube_resolution,
                         0,
                         GL_DEPTH_COMPONENT,
                         GL_UNSIGNED_INT,
                         dummy)

        glFramebufferTexture(GL_FRAMEBUFFER,
            GL_DEPTH_ATTACHMENT,
            self.texture2,
            0)

        # useful line for checking depth buffer
        #im_depth = glReadPixels(0, 0, 512,512, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT)
        #print im_depth

        #print 'Frame buffer status is: ', glCheckFramebufferStatus(GL_FRAMEBUFFER)

    def Reload(self):
        # todo - think this can be deleted now?
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glEnableVertexAttribArray(self.vertex_shader_attachment_ID)
        #glEnableVertexAttribArray(self.ConeVertsID)
        glVertexAttribPointer(
           self.vertex_shader_attachment_ID,                 # // attribute 0. No particular reason for 0, but must match the layout in the shader.
           3,                 # // size
           GL_FLOAT,          # // type
           GL_FALSE,          # // normalized?
           0,                 # // stride
           None                  # // array buffer offset
        )


class LoadQuads(object):

    def __init__(self,
                 vertex_shader_attachment_ID,
                 uv_shader_attachment_ID
                 ):
        ################################################################Setup quads
        self.vertex_attribute_1 = glGenVertexArrays(1)
        glBindVertexArray(self.vertex_attribute_1)

        quad_data = np.array(
            (-1,-1,0,
              1,-1,0,
              -1,1,0,
              -1,1,0,
              1,-1,0,
              1,1,0
            ),np.float32
            )

        # generate VBO, bind and copy the data
        vertex_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER,
                     quad_data.nbytes,          # Specifies the size in bytes of the buffer object's new data store.
                     quad_data,
                     GL_STATIC_DRAW)

        # 1st attribute buffer : vertices
        glEnableVertexAttribArray(vertex_shader_attachment_ID)

        #glEnableVertexAttribArray(self.vertex_attribute_1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
        glVertexAttribPointer(
           vertex_shader_attachment_ID,                 # // attribute 0. No particular reason for 0, but must match the layout in the shader.
           3,                 # // size
           GL_FLOAT,          # // type
           GL_FALSE,          # // normalized?
           0,                 # // stride
           None                  # // array buffer offset
        )

        ######################################## UV buffer
        uv_data = np.array(
            (
            0,0,
            1,0,
            0,1,
            0,1,
            1,0,
            1,1,
            ),np.float32
            )

        # generate UV buffer
        uvbuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, uvbuffer)
        glBufferData(GL_ARRAY_BUFFER, uv_data.nbytes, uv_data, GL_STATIC_DRAW)

        # attribute buffer : uv
        glEnableVertexAttribArray(uv_shader_attachment_ID)
        glBindBuffer(GL_ARRAY_BUFFER, uvbuffer)

        glVertexAttribPointer(
            uv_shader_attachment_ID,                                #// attribute. No particular reason for 1, but must match the layout in the shader.
            2,                                #// size
            GL_FLOAT,                         #// type
            GL_FALSE,                         #// normalized?
            0,                                #// stride
            None                              #// array buffer offset, JS: important to use None rather than 0
            )
        glBindVertexArray(0)

    def clean_up(self):

        glDisableVertexAttribArray(self.vertex_attribute_1)
        #todo: clean up all gl objects


class HabitatMesh(object):

    def __init__(self,
                 meshfilename,
                 meshfilepath,
                 ):

        self.meshfilepath = meshfilepath
        self.meshfilename = meshfilename
        self.components =[]

        #cached = os.path.join(self.meshfilepath, meshfilename )
        cache_file = os.path.join(self.meshfilepath, self.meshfilename )
        if os.path.isfile(cache_file):
            cached = np.load(cache_file)
            verticies = cached['v']
            colours = cached['c']
            faces = cached['i']
        else:
            print ('error - no file', cache_file, ' in the directory ', self.meshfilepath)

        self.facenumber = faces.shape[0]

        # print mesh info to screen
        #print 'loaded ', faces.shape, ' faces and ', verticies.shape, ' verticies and ', colours.shape, ' colours'
        #print 'data types are : ', faces.dtype, ' for faces and ', verticies.dtype, ' for verticies and ', colours.dtype, ' for colours'
        #print 'max colour value is : ', np.amax(colours[1])


        self.VertexArrayID = glGenVertexArrays(1)
        glBindVertexArray(self.VertexArrayID)
        ######################################################################## vertex buffer
        # Todo class for buffer object
        # Vertex buffer
        # generate VBO, bind and copy the data
        vertexbuffers = glGenBuffers(2)
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffers[0])


        glBufferData(GL_ARRAY_BUFFER,
                     verticies.nbytes,          # Specifies the size in bytes of the buffer object's new data store.
                     verticies,
                     GL_STATIC_DRAW)

        self.bytenumber = faces.shape

        # 1st attribute buffer : vertices
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffers[0])
        glVertexAttribPointer(
           0,                 # // attribute 0. No particular reason for 0, but must match the layout in the shader.
           3,                 # // size
           GL_FLOAT,          # // type
           GL_FALSE,          # // normalized?
           0,                 # // stride
           None                  # // array buffer offset
        )


        ################################################################## colour buffer
        colorbuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer)
        glBufferData(GL_ARRAY_BUFFER,
                     colours.nbytes,
                     colours,
                     GL_STATIC_DRAW)

        #// 2nd attribute buffer : colors
        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer)
        glVertexAttribPointer(
            1,                               # // attribute. No particular reason for 1, but must match the layout in the shader.
            3,                               # // size
            GL_FLOAT,                        # // type
            GL_FALSE,                        # // normalized?
            0,                               # // stride
            None                         # // array buffer offset
            )
        ################################################################### Element Buffer

        self.elementbuffer= vertexbuffers[1]
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexbuffers[1])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                        faces.nbytes,
                        faces,
                        GL_STATIC_DRAW)

        glBindVertexArray(0)


class ConeVerts(object):

    def __init__(self,
                 vertex_shader_attachment_ID,
                 ):

        self.vertex_shader_attachment_ID = vertex_shader_attachment_ID

        self.ConeVertsID = glGenVertexArrays(1)
        glBindVertexArray(self.ConeVertsID)

        # generate verts for a cone using the fan strip vertex method
        # Always create VAO after the OpenGL context is open
        # todo, get rid of the +0.1 in the line below, currently required to make cone fully connected
        thetas = np.linspace(0,2*np.pi+0.1,66,endpoint=True)
        radius = 0.3
        xs = radius*np.sin(thetas)
        ys = radius*np.cos(thetas)
        zs = np.ones_like(xs)
        cone_vertex_buffer_data = np.asarray(zip(xs,ys,zs),dtype=np.float32)
        cone_vertex_buffer_data = np.insert(cone_vertex_buffer_data,0,np.array((0,0,0),dtype=np.float32)).reshape(-1,3) # insert the cone tip to start of array
        self.buffer_len = cone_vertex_buffer_data.shape[0]

        # generate VBO, bind and copy the data
        self.vertex_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER,
                     cone_vertex_buffer_data.nbytes,          # Specifies the size in bytes of the buffer object's new data store.
                     cone_vertex_buffer_data,
                     GL_STATIC_DRAW)
        glEnableVertexAttribArray(self.vertex_shader_attachment_ID)
        #glEnableVertexAttribArray(self.ConeVertsID)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glVertexAttribPointer(
           self.vertex_shader_attachment_ID,                 # // attribute 0. No particular reason for 0, but must match the layout in the shader.
           3,                 # // size
           GL_FLOAT,          # // type
           GL_FALSE,          # // normalized?
           0,                 # // stride
           None                  # // array buffer offset
        )
        glBindVertexArray(0)


def create_cubemap(textureID=None):
    from sky import cubebox, ChromaticitySkyModel
    from datetime import datetime
    from ephem import city

    if textureID is None:
        textureID = glGenTextures(1)

    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID)

    # initialise sky
    obs = city("Edinburgh")
    obs.date = datetime.now()
    sky = ChromaticitySkyModel(observer=obs, nside=1)
    sky.generate()

    # create cubebox parts
    L_left, DOP_left, AOP_left = cubebox(sky, "left")
    L_front, DOP_front, AOP_front = cubebox(sky, "front")
    L_right, DOP_right, AOP_right = cubebox(sky, "right")
    L_back, DOP_back, AOP_back = cubebox(sky, "back")
    L_top, DOP_top, AOP_top = cubebox(sky, "top")
    L_bottom, DOP_bottom, AOP_bottom = cubebox(sky, "bottom")

    L_front_RGBX = np.rot90(np.rot90(
        np.concatenate((L_front, np.ones((L_front.shape[0], L_front.shape[1], 1))), axis=-1)
    ))
    L_back_RGBX = np.rot90(np.rot90(
        np.concatenate((L_back, np.ones((L_back.shape[0], L_back.shape[1], 1))), axis=-1)
    ))
    L_top_RGBX = np.rot90(
        np.concatenate((L_top, np.ones((L_top.shape[0], L_top.shape[1], 1))), axis=-1)
    )
    L_bottom_RGBX = np.rot90(
        np.concatenate((L_bottom, np.ones((L_bottom.shape[0], L_bottom.shape[1], 1))), axis=-1)
    )
    L_right_RGBX = np.rot90(np.rot90(
        np.concatenate((L_right, np.ones((L_right.shape[0], L_right.shape[1], 1))), axis=-1)
    ))
    L_left_RGBX = np.rot90(np.rot90(
        np.concatenate((L_left, np.ones((L_left.shape[0], L_left.shape[1], 1))), axis=-1)
    ))

    imgs = [
        fromarray(np.uint8(L_front_RGBX * 255), "RGBX"),
        fromarray(np.uint8(L_back_RGBX * 255), "RGBX"),
        fromarray(np.uint8(L_top_RGBX * 255), "RGBX"),
        fromarray(np.uint8(L_bottom_RGBX * 255), "RGBX"),
        fromarray(np.uint8(L_right_RGBX * 255), "RGBX"),
        fromarray(np.uint8(L_left_RGBX * 255), "RGBX")
    ]

    faces = [
        GL_TEXTURE_CUBE_MAP_POSITIVE_X,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
    ]

    for image, face in zip(imgs, faces):
        ix = image.size[0]
        iy = image.size[1]
        img = image.tobytes("raw", "RGBX", 0, -1)

        glTexImage2D(face, 0, GL_RGBA8, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, img)

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    return textureID


def axis_rotation_matrix(direction, angle):
     """
     Create a rotation matrix corresponding to the rotation around a general
     axis by a specified angle.

     R = dd^T + cos(a) (I - dd^T) + sin(a) skew(d)

     Parameters:

         angle : float a
         direction : array d
     """
     d = np.array(direction, dtype=np.float32)
     d /= np.linalg.norm(d)

     eye = np.eye(3, dtype=np.float32)
     ddt = np.outer(d, d)
     skew = np.array([[    0,  d[2],  -d[1]],
                      [-d[2],     0,  d[0]],
                      [d[1], -d[0],    0]], dtype=np.float32)

     mtx = ddt + np.cos(angle) * (eye - ddt) + np.sin(angle) * skew
     return np.transpose(mtx)
