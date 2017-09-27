import pygame
import numpy as np
import scipy
import time

from shader import *
from cube_verts import cube_verts
from camera import Camera
from utils import *
from definitions import SKYBOX_VAO_LOCATION


class Simulator(object):

    def __init__(self, insect_eyes,
                 display_w=1080, display_h=720, cube_resolution=64, acceptance_angles=np.deg2rad(3),
                 translation_speed=0.1, mouse_sensitivity=0.001,
                 key_front='w', key_back='s', key_right='d', key_left='a', key_up='o', key_down='l'):

        self.eyes = insect_eyes
        self.translation_speed = translation_speed
        self.mouse_sensitivity = mouse_sensitivity
        self.key = {
            "front": key_front,
            "back": key_back,
            "right": key_right,
            "left": key_left,
            "up": key_up,
            "down": key_down
        }
        faces = [GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
                 GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
                 GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z]

        # initialise OpenGL context
        pygame.init()
        pygame.display.set_mode((display_w, display_h), pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

        # load cubemapping program
        cubemap_program = ShaderClass("cubemap_gs.vert", "cubemap_gs.frag", geometry_shader_name="cubemap_gs.glsl")
        glUseProgram(cubemap_program.program)
        view_handle = glGetUniformLocation(cubemap_program.program, "view")
        projection_handle = glGetUniformLocation(cubemap_program.program, "projection")
        scale_handle = glGetUniformLocation(cubemap_program.program, "scale")

        # create skybox
        skybox_program = ShaderClass("skybox_simple.vert", "skybox_simple.frag")
        glUseProgram(skybox_program.program)

        skybox_view_handle = glGetUniformLocation(skybox_program.program, "view")
        skybox_projection_handle = glGetUniformLocation(skybox_program.program, "projection")
        skybox_resolution_handle = glGetUniformLocation(skybox_program.program, "resolution")
        glUniform2f(skybox_resolution_handle, display_w, display_h)

        glActiveTexture(GL_TEXTURE1)
        skyboxTexture_ID = create_cubemap()
        skybox_texture = glGetUniformLocation(skybox_program.program, "skybox")
        glUniform1i(skybox_texture, 1)
        glActiveTexture(GL_TEXTURE0)

        rot = np.pi / 2
        matricies = [
            axis_rotation_matrix(np.array((0, 1, 0)), rot * 3),  # correct for 1 way
            axis_rotation_matrix(np.array((0, 1, 0)), -3 * rot),
            axis_rotation_matrix(np.array((1, 0, 0)), -rot),
            axis_rotation_matrix(np.array((1, 0, 0)), rot),
            np.eye(3, dtype=np.float32),  # correct side
            axis_rotation_matrix(np.array((0, 1, 0)), 2 * rot),  # correct for 1 way
        ]

        ##################################################### Setup skybox VAO
        skyboxVAO = glGenVertexArrays(1)
        glBindVertexArray(skyboxVAO)
        # generate VBO and copy cube verticies into it
        skyboxVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO)
        skyboxVertices = cube_verts() * 20
        glBufferData(GL_ARRAY_BUFFER, skyboxVertices.nbytes, skyboxVertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(SKYBOX_VAO_LOCATION)
        glVertexAttribPointer(SKYBOX_VAO_LOCATION, 3, GL_FLOAT, GL_FALSE, 4 * 3, None)  # stride could also be 12

        ################################################################ clean up
        glBindVertexArray(0)  # disconnect vao to finalise - this is important!
        glActiveTexture(GL_TEXTURE0)

        # Handle eye type =======================================================
        cs = ComputeClass('computeshade_pointsampler.glsl')  # todo - move this to full eye model
        eye_model_file = np.load(definitions.EYE_GEOMETRY_PATH + '/' + compound_eye_file)
        view_dirs = eye_model_file['rdirs']

        print ('comp eye file ' + compound_eye_file)
        wm_file_name = definitions.EYE_GEOMETRY_PATH + '/weightmaps/wm_' + compound_eye_file[0:-4] + '_' + str(
            np.rad2deg(acceptance_angles)) + '_' + str(cube_resolution) + '.npz'
        print ('wm file ' + wm_file_name)
        if os.path.isfile(wm_file_name):
            print ('using previously calculated weightmap file ', wm_file_name)
        else:
            make_weightmap.make_weightmap(cube_resolution=cube_resolution,
                                          receptor_dirs=view_dirs,
                                          acceptance_angles=acceptance_angles,  # 0.1, #745329,#acceptance_angles,
                                          output_fileneme_str=wm_file_name,
                                          clip_thresh=1e-5
                                          )

        weight_file = np.load(wm_file_name)  # 'drosophila_rdirs.npz')
        view_dirs = weight_file['receptor_dirs']  # todo , get rid of this line?
        rweights = scipy.sparse.csc_matrix(
            (weight_file['spm_data'], weight_file['spm_indices'], weight_file['spm_indptr']),
            shape=weight_file['spm_shape'])

        ########################### process view directions
        view_dirs = np.stack((view_dirs[:, 0], view_dirs[:, 1], view_dirs[:, 2], np.ones(view_dirs.shape[0])), axis=1)
        view_dirs = view_dirs.astype(np.float32)

        view_dirs = view_dirs
        new_dirs = rotate_vector_list(view_dirs, (1, 0, 0), -90)

        response_ssbo = ShaderStorageBufferClass(
            view_dirs.shape[0],  # todo - generalise this or remove magic numbers
            4,  # todo - generalise this or remove magic number
            5,  # attachment point - todo define this with variable
        )

        ################################################################### load unrotated views
        new_dirs_ssbo = ShaderStorageBufferClass(
            new_dirs.shape[0],
            new_dirs.shape[1],
            4,  # attachment point - todo define this with variable
        )
        # view_dirs_ssbo.write_data(view_dirs.newbyteorder('>'))       # todo - need to verify this JS1
        new_dirs_ssbo.write_data(new_dirs)

        ##################################################################################### cone
        cone_program = ShaderClass('cone_vs.vert', 'cone_fs.frag')
        glUseProgram(cone_program.program)

        # get uniform locations
        clr = glGetUniformLocation(cone_program.program, "FragColor")
        centre_loc = glGetUniformLocation(cone_program.program, "cone_centre")
        CONE_VERTS_BUFFER_ID = 8
        cone_verticies = ConeVerts(CONE_VERTS_BUFFER_ID)

        # todo - add below to eye data models
        lon, lat = eul2geo(view_dirs[:, 0], view_dirs[:, 1], view_dirs[:, 2])
        # rescale between -1 and 1 for opengl window size
        lon = scale_linear_bycolumn(lon, high=1.0, low=-1.0)
        lat = scale_linear_bycolumn(lat, high=1.0, low=-1.0)
        cone_centres = np.column_stack((lon, lat))
        cone_centres = np.asarray(cone_centres, dtype=np.float32)

        ################################################################################################# end cone

        cube_face_data_len = 4 * cube_resolution * cube_resolution  # todo remove magic no.
        img_pointer = np.zeros(cube_face_data_len, dtype=np.uint8)

        reds_all = np.empty(6 * cube_resolution * cube_resolution, dtype=np.float32)
        greens_all = np.empty(6 * cube_resolution * cube_resolution, dtype=np.float32)
        blues_all = np.empty(6 * cube_resolution * cube_resolution, dtype=np.float32)

        glUseProgram(cubemap_program.program)  # set as program
        cube_frame_buffer = CubeFrameBuffer(cube_resolution, faces)

        quads = LoadQuads(quad_verts, quad_uv)

        # Setup mesh environment
        mesh = HabitatMesh(mesh_str, meshfilepath=definitions.MESH_PATH)
        self.camera = Camera()
        Projection = self.camera.perspective_projection_matrix(fovy=90, aspect=1)  # TODO: aspect ratio
        pos = (0, 0, -1)
        target = (1, 1, -1)
        View = self.camera.getLookAtMatrix(pos, target)

        # todo why isn't the scale working? Try adding into camera module?
        scale = np.eye(4, dtype=np.float32) * (1)
        scale[3, 3] = 1
        glUniformMatrix4fv(view_handle, 1, GL_FALSE, View)
        glUniformMatrix4fv(projection_handle, 1, GL_FALSE, Projection)
        glUniformMatrix4fv(scale_handle, 1, GL_FALSE, scale)

        # skybox
        glUseProgram(skybox_program.program)  # set as program
        glUniformMatrix4fv(skybox_view_handle, 1, GL_FALSE, View)
        glUniformMatrix4fv(skybox_projection_handle, 1, GL_FALSE, Projection)

        self.step_count = 0

        # initialise timers
        self.t_start = 0.0
        self.t_skybox = 0.0
        self.t_render = 0.0
        self.t_response = 0.0
        self.t_display = 0.0
        self.t_total = 0.0
        self.clock = pygame.time.Clock()

    def visual_processing(self):

        self.t_start = time.time()

        ################################################################### Setup frame buffer
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glBindFramebuffer(GL_FRAMEBUFFER, cube_frame_buffer.fb_name)
        glViewport(0, 0, cube_resolution, cube_resolution)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


        ################################################################### Render the skybox (if required)
        glUseProgram(skybox_program.program) # set as program
        glActiveTexture(GL_TEXTURE0)

        glBindVertexArray(skyboxVAO)
        cam_view = camera.get_view_matrix(translate=False)
        eyes_up = rot2homo(axis_rotation_matrix(np.array((1, 0, 0)), -np.pi/2))
        box_view = np.dot(eyes_up, cam_view)
        for idx, face in enumerate(faces):

                # todo - optimise this transfer - upload matricies as UBO in the initiaisation phase
                mat = matricies[idx]
                mat = rot2homo(mat)
                rot_mat = np.dot(box_view, mat)
                glUniformMatrix4fv(skybox_view_handle, 1, GL_FALSE, rot_mat)
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, face, cube_frame_buffer.texture_name, 0)
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, face, cube_frame_buffer.texture2, 0)
                glDrawArrays(GL_TRIANGLES, 0, 36)
        glBindVertexArray(0)


        t_skybox = time.time() - t_start

        ######################################################################### render scene
        glUseProgram(cubemap_program.program)
        #glBindTexture(GL_TEXTURE_CUBE_MAP,self.cube_frame_buffer.texture_name)
        View = camera.get_view_matrix()
        glUniformMatrix4fv(view_handle, 1, GL_FALSE, View)
        glBindVertexArray(mesh.VertexArrayID)
          #bind our buffer to the active texture
        glBindTexture(GL_TEXTURE_CUBE_MAP, cube_frame_buffer.texture_name)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, cube_frame_buffer.texture_name, 0)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, cube_frame_buffer.texture2, 0)
        glDrawElements(
                         GL_TRIANGLES,      #// mode
                         mesh.facenumber*3,    #// count
                         GL_UNSIGNED_INT,   #// type
                         None           #// element array buffer offset
                        )
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, cube_frame_buffer.texture_name)
        t_render = time.time() - t_skybox - t_start


        ############################################################## Draw quads to screen
        if display_output:
            if eye_type == 'equirectangular':

                glUseProgram(quad_program.program)
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                glViewport(0, 0, display_w, display_h)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glBindVertexArray(quads.vertex_attribute_1)
                glDrawArrays(GL_TRIANGLES, 0, 6)  # Starting from vertex 0; 3 vertices total -> 1 triangle
                glBindVertexArray(0)

                glActiveTexture(GL_TEXTURE0)

                self.t_response = time.time() - self.t_render - self.t_start

            if eye_type == 'compoundeye':

                if use_compute_shader:

                    glUseProgram(cs.program)
                    glDispatchCompute(10, 1, 1)
                    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)  # todo - work out exactly how this works#
                    eye_output = response_ssbo.read_data()
                    eye_output_colours = eye_output[:, 0:3]

                    # print 'responses size: ', self.eye_output_colours.shape
                    # colours = np.random.rand(self.cone_centres.shape[0],3)

                else:

                    for idx, face in enumerate(faces):
                        glGetTexImage(face,
                                      0,
                                      GL_RGBA,
                                      GL_UNSIGNED_BYTE,           # or unsigned byte?
                                      img_pointer,
                                      )

                        start_idx = idx * (cube_face_data_len / 4)
                        end_idx = (idx + 1) * (cube_face_data_len / 4)

                        # we read the memory from the buffer backwards, using steps of 4 for each channel (R,G,B,A)
                        reds_all[start_idx:end_idx] = img_pointer[-4::-4]
                        greens_all[start_idx:end_idx] = img_pointer[-3::-4]
                        blues_all[start_idx:end_idx] = img_pointer[-2::-4]

                    responsesR = rweights * reds_all
                    responsesG = rweights * greens_all  # sparse matrix times vector = vector
                    responsesB = rweights * blues_all

                    glUseProgram(cs.program)
                    eye_output_colours = np.stack((responsesR, responsesG, responsesB), 1) / 255.
                    #self.response_ssbo.write_data(self.eye_output)
                    #print 'responses size: ', self.eye_output_colours.shape
                    #print 'responses type: ', type(self.eye_output_colours[0,0])
                    #print ('eye output shape is: ', self.eye_output_colours.shape)

                self.t_response = time.time() - self.t_render - self.t_start


                ################################################################ draw cones
                glUseProgram(cone_program.program)
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                glEnable(GL_DEPTH_TEST)
                glDepthFunc(GL_LEQUAL)

                glViewport(0, 0, display_w, display_h)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                glBindVertexArray(cone_verticies.ConeVertsID)
                for idx in range(cone_centres.shape[0]):
                    if use_compute_shader:
                        #colour = self.eye_output_colours.shape
                        colour = np.asarray(eye_output_colours[idx, :], dtype=np.float32)
                    else:
                        colour = np.asarray(eye_output_colours[idx, :], dtype=np.float32)

                    cc = np.asarray(cone_centres[idx, :], dtype=np.float32)
                    glUniform3f(clr, colour[0], colour[1], colour[2])
                    glUniform2f(centre_loc, cc[0], cc[1])
                    glDrawArrays(GL_TRIANGLE_FAN, 0, cone_verticies.buffer_len)
                glBindVertexArray(0)
                ############################################################ cone.end


        # finish step
        self.t_display = time.time() - self.t_response - self.t_start

        # press Esc to quit function
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()

        # complete metrics
        self.step_count += 1
        self.t_total = time.time() - self.t_start
        self.clock.tick()
        pygame.display.set_caption("fps: " + str(self.clock.get_fps()))

    def step(self):
        pressed = dict((chr(i), int(v)) for i, v in enumerate(pygame.key.get_pressed()) if i < 256)
        mouse_dx, mouse_dy = pygame.mouse.get_rel()

        self.camera.rx -= mouse_dx * self.mouse_sensitivity
        # self.camera.ry += mouse_dy * self.mouse_sensitivity

        # translate to new position
        vx, vy, vz = self.camera.forward_vector()
        strafe = self.translation_speed * (pressed[self.key["left"]] - pressed[self.key["right"]])
        self.__translate_camera(vx, vy, vz, strafe)

        vx, vy, vz = self.camera.right_vector()
        forward = self.translation_speed * (pressed[self.key["front"]] - pressed[self.key["back"]])
        self.__translate_camera(vx, vy, vz, forward)

        vx, vy, vz = self.camera.up_vector()
        up = self.translation_speed * (pressed[self.key["down"]] - pressed[self.key["up"]])
        self.__translate_camera(vx, vy, vz, up)

        # update display
        self.visual_processing()
        pygame.display.flip()
        pygame.event.pump()

    def __translate_camera(self, vx, vy, vz, direction):
        self.camera.x += vx * direction
        self.camera.y += vy * direction
        self.camera.z += vz * direction
