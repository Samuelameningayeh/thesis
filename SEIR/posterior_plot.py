import math
import matplotlib.pyplot as plt
import seaborn as sns

def plot_posterior_kde_grid(fit, 
                            parameters: list, 
                            n_cols: int, 
                            true_values, 
                            save=False):
    """
    Plot KDEs of posterior samples for given parameters from a CmdStanMCMC fit.

    Args:
        fit (CmdStanMCMC): The fitted model.
        parameters (list of str): Names of parameters to plot.
        n_rows (int): Number of subplot rows.
        n_cols (int): Number of subplot columns.
        true_values (dict, optional): Dict mapping param names to true values for reference lines.
    """

    symbol_dict = {
        "beta[1]": r"$\beta_1$", "beta[2]": r"$\beta_2$", "beta[3]": r"$\beta_3$",
        "sigma[1]": r"$\sigma_1$", "sigma[2]": r"$\sigma_2$", "sigma[3]": r"$\sigma_3$",
        "gamma[1]": r"$\gamma_1$", "gamma[2]": r"$\gamma_2$", "gamma[3]": r"$\gamma_3$",
        "R0[1]": r"$R_{0,1}$", "R0[2]": r"$R_{0,2}$", "R0[3]": r"$R_{0,3}$",
        "sigma": r"$\sigma$", "gamma": r"$\gamma$", "R0": r"$R_{0}$",
        "beta": r"$\beta$"
    }
    
    posterior_samples = fit.draws_pd()
    
    # Dynamically calculate the number of rows
    n_params = len(parameters)
    n_rows = math.ceil(n_params / n_cols)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows))
    ax = ax.flatten()  # flatten axes for easy indexing

    for i, param in enumerate(parameters):
        if param in posterior_samples.columns:
            sns.kdeplot(posterior_samples[param], fill=True, ax=ax[i])
            if true_values and param in true_values:
                ax[i].axvline(x=true_values[param], color='r', linestyle='--', label='True value')
                ax[i].legend()
            ax[i].set_ylabel('Density', fontsize=15)
            ax[i].set_xlabel(f'{param}({symbol_dict.get(param)})' if param in symbol_dict else param, fontsize=18)
            ax[i].set_title(f'Posterior of {symbol_dict.get(param)}' if param in symbol_dict else param, fontsize=18)
            ax[i].tick_params(axis='x', labelsize=15)  # Set font size for x-axis ticks
            ax[i].tick_params(axis='y', labelsize=15)

        else:
            ax[i].text(0.5, 0.5, f"{param} not found", ha='center', va='center')
            ax[i].axis('off')

    # Hide unused axes
    for j in range(len(parameters), len(ax)):
        ax[j].axis('off')

    plt.tight_layout()
    if save:
        plt.savefig(f'./images/{save}.pdf', bbox_inches='tight')
    plt.show()
