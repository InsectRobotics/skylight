import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
IMAGES_PATH = os.path.join(ROOT_DIR, 'skylight/data/images')
EYE_GEOMETRY_PATH = os.path.join(ROOT_DIR, 'pyeye/compound_eye/eye_geometry')
MESH_PATH = os.path.join(ROOT_DIR, 'pyeye/data')

# VAO locations
SKYBOX_VAO_LOCATION = 10