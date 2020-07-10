def plot_data(X, scatter_color='grey', var1_name="Malic acid (g/l)", var2_name="Total phenols (g/l)"):
    plt.scatter(X[:,0], X[:,1], c=scatter_color, alpha=0.7)
    plt.xlabel(var1_name)
    plt.ylabel(var2_name)
    plt.savefig('/Users/michel/Documents/pjs/em-article/data/plots/wine-data.png', dpi=300)


def plot_m_step(X, q, var1_name="malic_acid", var2_name="total_phenols"):
    # https://seaborn.pydata.org/generated/seaborn.JointGrid.html
    plot_df = pd.DataFrame({var1_name: X[:, 0], 
                    var2_name: X[:, 1]})

    cmap = sns.cubehelix_palette(as_cmap=True)

    g = sns.JointGrid(x=var1_name, y=var2_name, data=plot_df)

    g = g.plot_joint(sns.scatterplot, 
                    hue=q[:, 0],
                    palette=cmap)
    
    # Plot var 1 using component 1 weights
    _ = g.ax_marg_x.hist(plot_df[var1_name], 
                        color="#123a6b", 
                        alpha=.6,
                        weights=q[:, 0])

    # Plot var 2 using component 1 weights
    _ = g.ax_marg_y.hist(plot_df[var2_name], 
                        color="#0f5e4d", 
                        alpha=.6,
                        orientation="horizontal",
                        weights=q[:, 0])

    #g.fig.suptitle('M-Step')

def plot_e_step(X, mu, sigma, alpha=0.5, scatter_color='grey', contour_color=['#33052d', "#d996b9"]):
    plt.scatter(X[:,0], X[:,1], c=scatter_color, alpha=0.7)

    grid_x, grid_y = np.mgrid[X[:,0].min():X[:,0].max():200j,
                     X[:,1].min():X[:,1].max():200j]
    grid = np.stack([grid_x, grid_y], axis=-1)

    i = 0
    for mu_c, sigma_c in zip(mu, sigma):
        plt.contour(grid_x, 
                    grid_y, 
                    mvn(mu_c, sigma_c).pdf(grid), 
                    colors=contour_color[i], 
                    alpha=alpha)
        
        i += 1
    
    #plt.suptitle('E-Step')