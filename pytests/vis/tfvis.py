import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import json
import io
from PIL import Image

def arrowed_spines(ax=None, arrowLength=10, labels=('X', 'Y'), arrowStyle='<|-'):
  xlabel, ylabel = labels

  for i, spine in enumerate(['left', 'bottom']):
    # Set up the annotation parameters
    t = ax.spines[spine].get_transform()
    xy, xycoords = [1, 0], ('axes fraction', t)
    xytext, textcoords = [arrowLength, 0], ('offset points', t)

    # create arrowprops
    arrowprops = dict(arrowstyle=arrowStyle,
                      facecolor=ax.spines[spine].get_facecolor(),
                      linewidth=ax.spines[spine].get_linewidth(),
                      alpha=ax.spines[spine].get_alpha(),
                      zorder=ax.spines[spine].get_zorder(),
                      linestyle=ax.spines[spine].get_linestyle())

    if spine is 'bottom':
      ha, va = 'left', 'center'
      xarrow = ax.annotate(xlabel, xy, xycoords=xycoords, xytext=xytext,
                           textcoords=textcoords, ha=ha, va='center',
                           arrowprops=arrowprops)
    else:
      ha, va = 'center', 'bottom'
      yarrow = ax.annotate(ylabel, xy[::-1], xycoords=xycoords[::-1],
                           xytext=xytext[::-1], textcoords=textcoords[::-1],
                           ha='center', va=va, arrowprops=arrowprops)
  return xarrow, yarrow

def lerp(x, a, b):
  return a + x * (b-a)

def renderTfLinear(
        tf : np.ndarray, # R*5
        axes : plt.Axes,
        use_min_density = False
) -> Image:
  axes.clear()
  assert len(tf.shape)==2
  assert tf.shape[1] == 5
  opacityValues = tf[:,3]
  opacityDensities = tf[:,4]
  colorValues = tf[:,:3]
  colorDensities = tf[:,4]

  # compute colors at the control points of the opacities
  xmin, xmax, ymin, ymax = min(opacityDensities), max(opacityDensities), min(opacityValues), max(opacityValues)
  #(xmin, xmax, ymin, ymax)
  colorsAtOpacities = np.empty((1, 200, 3), dtype=float)
  for i,d in enumerate(np.linspace(xmin, xmax, num=colorsAtOpacities.shape[1], endpoint=True)):
    found = False
    for j in range(len(colorDensities)-1):
      d1 = colorDensities[j]
      d2 = colorDensities[j+1]
      if d1 <= d <= d2:
        f = (d-d1) / (d2-d1)
        rgb1 = colorValues[j,:]
        rgb2 = colorValues[j+1,:]
        rgb_mixed = lerp(f, rgb1, rgb2)
        colorsAtOpacities[0,i,:] = rgb_mixed
        found=True
        break
    #print("found:", found)
  colorsAtOpacities = np.clip(colorsAtOpacities, 0, 1)
  #print(colorsAtOpacities)
  line, = axes.plot(opacityDensities, opacityValues, 'o-')
  im = axes.imshow(colorsAtOpacities, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                        origin='lower', zorder=line.get_zorder())
  xy = np.column_stack([opacityDensities, opacityValues])
  xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
  clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
  axes.add_patch(clip_path)
  im.set_clip_path(clip_path)

  #axes.set_ylim(0, axes.get_ylim()[1]*1.05)
  #print(ymax)
  axes.set_ylim(0, ymax*1.05)

  # axes to arrows
  axes.spines['right'].set_visible(False)
  axes.spines['top'].set_visible(False)
  arrowed_spines(plt.gca(), labels=('d', 'tf'))
