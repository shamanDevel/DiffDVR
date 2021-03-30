import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

def renderCamera(
        initial_viewport : np.ndarray,
        target_viewport : np.ndarray,
        current_viewport : np.ndarray,
        reference_sphere_radius : float,
        eye_positions : np.ndarray,
        axes : Axes3D):
  """
  Viewport format: shape 3*3 with
    viewport[0,:] is the eye position
    viewport[1,:] is the right vector
    viewport[2,:] is the up vector
  :param initial_viewport:
  :param target_viewport:
  :param current_viewport:
  :param reference_sphere_radius:
  :param eye_positions: eye positions of shape N*3
  :param axes:
  :return:
  """

  assert (initial_viewport.shape == (3,3))
  assert (target_viewport.shape == (3, 3))
  assert (current_viewport.shape == (3, 3))

  # make sphere
  if reference_sphere_radius is not None:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = reference_sphere_radius * np.outer(np.cos(u), np.sin(v))
    y = reference_sphere_radius * np.outer(np.sin(u), np.sin(v))
    z = reference_sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    axes.plot_surface(x,y,z, alpha=0.2, shade=True)

  def drawViewport(viewport, color, len=0.2):
    if viewport is None: return
    eye = viewport[0, :]
    right = viewport[1, :]
    up = viewport[2, :]
    front = np.cross(up, right)
    axes.quiver(
      [eye[0]]*3, [eye[1]]*3, [eye[2]]*3,
      [right[0], up[0], front[0]], [right[1], up[1], front[1]], [right[2], up[2], front[2]],
      length=len,
      pivot='tail',
      linestyles=['solid', 'solid', 'dashed'],
      color=color)

  drawViewport(initial_viewport, (1,0,0))
  drawViewport(target_viewport, (0,0,1))
  drawViewport(current_viewport, (0,0.3,0))

  if eye_positions is not None and eye_positions.shape[0]>0:
    axes.plot3D(eye_positions[:,0], eye_positions[:,1], eye_positions[:,2], color=(0,0,0))

  #print(axes.elev, axes.azim, axes.dist)
  #axes.elev = 20
  #axes.dist = 7

if __name__ == '__main__':
  initial_viewport = np.array([
    [ 0.27023358, -0.175266,   -0.62149468],
    [ 0.91706007,  0.,          0.39874907],
    [-0.09983879, -0.96814764,  0.22961351]])
  target_viewport = np.array([
     [-0.02768447, -0.48700914, -0.63407859],
     [0.99904822,  0., -0.04361939],
     [0.0265538, -0.79335334, 0.60818202]])
  current_viewport = target_viewport

  fig = plt.figure(figsize=(6,4))
  ax = fig.add_subplot(111, projection='3d')
  renderCamera(initial_viewport, target_viewport, current_viewport, 0.7, None, ax)
  plt.show()