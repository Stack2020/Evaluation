import matplotlib.pyplot as plt

def plot_three(gt, pred, err, extent, vmin, vmax, err_vmin, err_vmax, out):
    fig = plt.figure(figsize=(6.6, 2.3), dpi=300)

    # 1行6列：主图,cb,主图,cb,主图,cb
    gs = fig.add_gridspec(
        1, 6,
        width_ratios=[1.0, 0.05, 1.0, 0.05, 1.0, 0.05],  # 固定colorbar宽度
        wspace=0.25, left=0.06, right=0.98, bottom=0.18, top=0.90
    )

    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 4])]
    caxes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[0, 5])]

    ims = []
    ims.append(axes[0].imshow(gt,   origin="lower", extent=extent, vmin=vmin, vmax=vmax, aspect="equal"))
    ims.append(axes[1].imshow(pred, origin="lower", extent=extent, vmin=vmin, vmax=vmax, aspect="equal"))
    ims.append(axes[2].imshow(err,  origin="lower", extent=extent, vmin=err_vmin, vmax=err_vmax, aspect="equal"))

    titles = ["GT Density", "PF-ODE Density", "Absolute Error"]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.tick_params(labelsize=7)
        # 可选：强制“轴盒子”正方形（更稳定）
        # ax.set_box_aspect(1)

    for im, cax in zip(ims, caxes):
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=7)

    # 关键：别用 bbox_inches="tight"（tight 会因标签不同导致外框变化）
    fig.savefig(out, dpi=300)
    plt.close(fig)
