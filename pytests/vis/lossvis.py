import numpy as np
import matplotlib.pyplot as plt

def renderLoss(loss : np.ndarray, axes : plt.Axes, frame = None):
  assert len(loss.shape)==1
  if frame is None:
    frame = loss.shape[0]-1

  X = np.arange(frame+1)
  Y = loss[:frame+1]
  axes.plot(X, Y)
  axes.set_xlim(0, loss.shape[0]+0.1)
  max_y = max(loss)
  axes.set_ylim(-0.01*max_y, 1.01*max_y)
  axes.set_xlabel("Iteration")
  axes.set_ylabel("Loss")