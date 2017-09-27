'''
This module provides the functions required in order to configure an opengl camera for visualising 3D scenes

'''

import numpy as np
from utils import normalize
#TODO add functionality to remain a fixed distance above the ground

class Camera(object):

    def __init__(self,
                x = 0,
                y = 0,
                z = 0,
                rx = 0,  # horizontal angle
                ry = -np.pi/2,  # vertical angle - must initialise less than pi
                rz = 0,
                mat = np.eye(4, dtype=np.float32),
                 ):

        self.x = x
        self.y = y
        self.z = z
        self.rx = rx     # horizontal angle
        self.ry = ry    #vertical angle - must initialise less than pi
        self.rz = rz
        self.mat = mat

    def get_view_matrix(self, translate=True):

        self.mat = np.eye(4, dtype=np.float32)
        self.mat = self.rotate((np.cos(self.rx), 0, np.sin(self.rx)), self.ry)
        self.mat = self.rotate((0, -1, 0), -self.rx)
        if translate:
            self.mat = self.translate((-self.x, -self.y, -self.z))
        return self.mat

    def translate(self, value):
        x, y, z = value
        matrix = np.matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [x, y, z, 1]
        ], dtype=np.float32)
        return np.dot(matrix, self.mat)

    def rotate(self, vector, angle):
        # todo - is this column or row major?
        x, y, z = normalize(vector)
        s = np.sin(angle)
        c = np.cos(angle)
        m = 1 - c
        matrix = np.matrix([
            [m * x * x + c, m * x * y - z * s, m * z * x + y * s, 0],
            [m * x * y + z * s, m * y * y + c, m * y * z - x * s, 0],
            [m * z * x - y * s, m * y * z + x * s, m * z * z + c, 0],
            [0, 0, 0, 1]
        ])
        return np.dot(matrix, self.mat)

    def forward_vector(self):
        vx = np.sin(self.rx - np.pi/2) * (-np.sin(self.ry))
        vy = np.cos(self.rx - np.pi/2) * (-np.sin(self.ry))
        vz = np.cos(self.ry)
        return (vx, vy, vz)

    def right_vector(self):
        vx = -np.sin(self.rx)
        vy = -np.cos(self.rx)
        vz = np.cos(self.ry)
        return (vx, vy, vz)

    def up_vector(self):
        # derived from http://www.opengl-tutorial.org/beginners-tutorials/tutorial-6-keyboard-and-mouse/
        return np.cross(self.right_vector(),self.forward_vector())

    def getLookAtMatrix(self, eye, lookat, up=(0, 0, 1)):
        # source: https://sourceforge.net/p/pyopengl/mailman/message/34655108/
        ez = np.subtract(eye, lookat)
        ez = ez / np.linalg.norm(ez)

        ex = np.cross(up, ez)
        ex = ex / np.linalg.norm(ex)

        ey = np.cross(ez, ex)
        ey = ey / np.linalg.norm(ey)

        rmat = np.eye(4)
        rmat[0][0] = ex[0]
        rmat[0][1] = ex[1]
        rmat[0][2] = ex[2]

        rmat[1][0] = ey[0]
        rmat[1][1] = ey[1]
        rmat[1][2] = ey[2]

        rmat[2][0] = ez[0]
        rmat[2][1] = ez[1]
        rmat[2][2] = ez[2]

        tmat = np.eye(4)
        tmat[0][3] = -eye[0]
        tmat[1][3] = -eye[1]
        tmat[2][3] = -eye[2]

        return np.dot(rmat, tmat).transpose()


    def perspective_projection_matrix(self, fovy=90, aspect=1.0, near=0.01, far=100, dtype=np.float32):

        """Creates perspective projection matrix.
        .. seealso:: http://www.songho.ca/opengl/gl_projectionmatrix.html
        .. seealso:: http://www.opengl.org/sdk/docs/man2/xhtml/gluPerspective.xml
        .. seealso:: http://www.geeks3d.com/20090729/howto-perspective-projection-matrix-in-opengl/
        :param float fovy: field of view in y direction in degrees
        :param float aspect: aspect ratio of the view (width / height)
        :param float near: distance from the viewer to the near clipping plane (only positive)
        :param float far: distance from the viewer to the far clipping plane (only positive)
        :rtype: numpy.array
        :return: A projection matrix representing the specified perpective.
        """
        ymax = near * np.tan(fovy * np.pi / 360.0)
        xmax = ymax * aspect
        return self.create_perspective_projection_matrix_from_bounds(-xmax, xmax, -ymax, ymax, near, far)

    def create_perspective_projection_matrix_from_bounds(
        self,
        left,
        right,
        bottom,
        top,
        near,
        far,
        dtype=np.float32
        ):
        """Creates a perspective projection matrix using the specified near
        plane dimensions.
        :param float left: The left of the near plane relative to the plane's centre.
        :param float right: The right of the near plane relative to the plane's centre.
        :param float top: The top of the near plane relative to the plane's centre.
        :param float bottom: The bottom of the near plane relative to the plane's centre.
        :param float near: The distance of the near plane from the camera's origin.
            It is recommended that the near plane is set to 1.0 or above to avoid rendering issues
            at close range.
        :param float far: The distance of the far plane from the camera's origin.
        :rtype: numpy.array
        :return: A projection matrix representing the specified perspective.
        .. seealso:: http://www.gamedev.net/topic/264248-building-a-projection-matrix-without-api/
        .. seealso:: http://www.glprogramming.com/red/chapter03.html
        """

        """
        E 0 A 0
        0 F B 0
        0 0 C D
        0 0-1 0
        A = (right+left)/(right-left)
        B = (top+bottom)/(top-bottom)
        C = -(far+near)/(far-near)
        D = -2*far*near/(far-near)
        E = 2*near/(right-left)
        F = 2*near/(top-bottom)
        """
        A = (right + left) / (right - left)
        B = (top + bottom) / (top - bottom)
        C = -(far + near) / (far - near)
        D = -2. * far * near / (far - near)
        E = 2. * near / (right - left)
        F = 2. * near / (top - bottom)

        return np.array((
            (  E, 0., 0., 0.),
            ( 0.,  F, 0., 0.),
            (  A,  B,  C,-1.),
            ( 0., 0.,  D, 0.),
        ), dtype=dtype)

    def ortho_projection_matrix(self, fovy=90, aspect=1, near=0.01, far=100, dtype=np.float32):

        """Creates perspective projection matrix.
        .. seealso:: http://www.songho.ca/opengl/gl_projectionmatrix.html
        .. seealso:: http://www.opengl.org/sdk/docs/man2/xhtml/gluPerspective.xml
        .. seealso:: http://www.geeks3d.com/20090729/howto-perspective-projection-matrix-in-opengl/
        :param float fovy: field of view in y direction in degrees
        :param float aspect: aspect ratio of the view (width / height)
        :param float near: distance from the viewer to the near clipping plane (only positive)
        :param float far: distance from the viewer to the far clipping plane (only positive)
        :rtype: numpy.array
        :return: A projection matrix representing the specified perpective.
        """
        ymax = near * np.tan(fovy * np.pi / 360.0)
        xmax = ymax * aspect
        return self.create_perspective_projection_matrix_from_bounds(-xmax, xmax, -ymax, ymax, near, far)

    def create_ortho_projection_matrix_from_bounds(
        self,
        left,
        right,
        bottom,
        top,
        near,
        far,
        dtype=np.float32
    ):
        """Creates a perspective projection matrix using the specified near
        plane dimensions.
        :param float left: The left of the near plane relative to the plane's centre.
        :param float right: The right of the near plane relative to the plane's centre.
        :param float top: The top of the near plane relative to the plane's centre.
        :param float bottom: The bottom of the near plane relative to the plane's centre.
        :param float near: The distance of the near plane from the camera's origin.
            It is recommended that the near plane is set to 1.0 or above to avoid rendering issues
            at close range.
        :param float far: The distance of the far plane from the camera's origin.
        :rtype: numpy.array
        :return: A projection matrix representing the specified perspective.
        .. seealso:: http://www.gamedev.net/topic/264248-building-a-projection-matrix-without-api/
        .. seealso:: http://www.glprogramming.com/red/chapter03.html
        """

        """
        E 0 A 0
        0 F B 0
        0 0 C D
        0 0-1 0
        A = (right+left)/(right-left)
        B = -(top+bottom)/(top-bottom)
        C = -(far+near)/(far-near)
        D = -2*far*near/(far-near)
        E = 2/(right-left)
        F = 2*near/(top-bottom)
        """
        A = -(right + left) / (right - left)
        B = (top + bottom) / (top - bottom)
        C = -2. / (far - near)
        D = (-(far + near)) / (far - near)
        E = 2. * near / (right - left)
        F = 2. * near / (top - bottom)

        return np.array((
            (E , 0., 0., 0.),
            (0., F , 0., 0.),
            (0., 0., C , 0.),
            (A , B , D , 1.),
        ), dtype=dtype)


#if __name__ == '__main__':





