import matplotlib.pyplot as plt
import numpy as np
def plot_lbn_weights(weights, name, cmap="OrRd",
                     slot_names=("$tau_{1}$", "$tau_{2}$","$MET$", "$jet_1$", "$jet_2$", "$jet_3$",
                                 "$jet_4$"), **fig_kwargs):
    # normalize weight tensor to a sum of 100 per row
    weights = weights / \
        np.sum(weights, axis=0).reshape((1, weights.shape[1])) * 100

    # create the figure
    fig_kwargs.setdefault("figsize", (10, 5))
    fig_kwargs.setdefault("dpi", 120)
    fig = plt.figure(**fig_kwargs)
    ax = fig.add_subplot(1, 2, 1)

    # create and style the plot
    ax.imshow(weights, cmap=cmap, vmin=0, vmax=100)
    ax.set_title("{} weights".format(name), fontdict={"fontsize": 12})

    ax.set_xlabel("LBN particle number")
    ax.set_xticks(list(range(weights.shape[1])))

    ax.set_ylabel("Input particle")
    ax.set_yticks(list(range(weights.shape[0])))
    ax.set_yticklabels(slot_names)

    # write weights into each bin
    for (i, j), val in np.ndenumerate(weights):
        ax.text(j, i, int(round(weights[i, j])),
                fontsize=8, ha="center", va="center", color="k")

    # return figure and axes
    return fig, ax

particle_weights=np.load(open("savemodeldir/particle_weights","rb"))
restframe_weights=np.load(open("savemodeldir/restframe_weights","rb"))


fig,ax=plot_lbn_weights(particle_weights,"particle")
ax.plot()
fig.savefig("savemodeldir/test_weights_particles")

fig,ax=plot_lbn_weights(restframe_weights,"rest frame")
ax.plot()
fig.savefig("savemodeldir/test_weights_rest")
