import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("NonConvexDensity.tsv", skiprows=2, delimiter="\t")
print(data)

fig = plt.figure(figsize=(5,1.5))

color = 'tab:blue'
ax = plt.gca()
ax.set_xlabel("$d_1:$")
ax.set_ylabel("Cost", color=color)
plt.plot(data[:-1,0], data[:-1,3], color=color)
ax.tick_params(axis='y', labelcolor=color)
ax.axvline(-1.0, color=color, linestyle="--", alpha=0.5)
ax.set_xticks([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.3])
ax.set_xticklabels(["-2.0", "-1.5", "-1.0", "-0.5", "0.0", "0.5", "1.0", "1.5", "2.0", "$\\infty$"])

ticklab = ax.xaxis.get_ticklabels()[0]
trans = ticklab.get_transform()
ax.xaxis.set_label_coords(-2.4, 0.01, transform=trans)

ax2 = ax.twinx()
color2 = 'tab:red'
ax2.set_ylabel("Gradient", color=color2)
ax2.plot(data[:-1,0], data[:-1,4], color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.axhline(0.0, color=color2, linestyle="--", alpha=0.5)

ax.plot([2.3], [data[-1,3]], '.', color=color)
ax2.plot([2.3], [data[-1,4]], '.', color=color2)

plt.subplots_adjust(0.11, 0.17, 0.86, 0.98)
#plt.tight_layout()
plt.savefig("NonConvexDensity-Loss.pdf")


fig2 = plt.figure(figsize=(2.5,1))
mean = 0
variance = 0.5
min_x = -2
max_x = +2
X = np.linspace(min_x, max_x, 100, endpoint=True)
Y = np.exp(-(X-mean)**2/(2*variance*variance))
ax = plt.gca()
ax.set_xlabel("$d:$")
ax.set_ylabel("$r,\\tau$")
plt.plot(X, Y, 'k-')
#plt.tight_layout()
plt.subplots_adjust(0.16, 0.23, 0.99, 0.96)

ticklab = ax.xaxis.get_ticklabels()[0]
trans = ticklab.get_transform()
ax.xaxis.set_label_coords(-2.4, 0.01, transform=trans)

plt.savefig("NonConvexDensity-TF.pdf")

#plt.show()
