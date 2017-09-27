
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SHADER_PATH = os.path.join(ROOT_DIR, 'skylight/simulation/shaders')
IMAGES_PATH = os.path.join(ROOT_DIR, 'skylight/data/images')
EYE_GEOMETRY_PATH = os.path.join(ROOT_DIR, 'skylight/compoundeye/geometry')
MESH_PATH = os.path.join(ROOT_DIR, 'skylight/data')

# VAO locations
SKYBOX_VAO_LOCATION = 10