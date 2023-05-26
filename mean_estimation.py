import networkx as nx
import numpy as np
import random

import matplotlib.pyplot as plt
import argparse
import dataloader
import powerlaw

random.seed(0)
np.random.seed(0)

FONTSIZE = 18

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='karate_club', type=str)
    parser.add_argument('--eps', default=1, type=float)
    parser.add_argument('-T', default=100, type=int)
    parser.add_argument('-l', default=10, type=float, help='Mean of log-normal distribution')

    return parser.parse_args()

def build_network(G):
    A = nx.to_numpy_array(G).astype(np.float64)
    n = A.shape[0]

    d = A.sum(0)

    for i in range(n):
        A[i, i] = 0
        for j in range(i):
            if A[i, j] != 0:
                A[i, j] = 1.0 / max(d[i], d[j])
                A[j, i] = A[i, j]
    
    for i in range(n):
        A[i, i] = 1 - A[i, :].sum()

    return A, n

def mean_estimation(A, n, T, l, eps, signals=None, intermittent=True, protect_network=False):

    # Mean Estimation
    mu = np.zeros((n, T + 1))
    nu = np.zeros((n, T + 1)) 

    if signals is not None:
        if intermittent:
            mu_theta = signals.mean()
        else:
            mu_theta = signals[:, 0].mean()
    else:
        mu_theta = l
    
    print('mu_theta', mu_theta)

    if protect_network:
        I = np.eye(n)
        a_diag = np.diag(A)
        A_ndiag = (1 - I) * A
        a_max = A_ndiag.max(0)

    V = lambda x: np.sum((x - x.mean())**2)

    V_mu = np.zeros(T + 1)
    V_nu = np.zeros(T + 1)

    for t in range(1, T + 1):
        eta = 1 / t
        if intermittent:
            s = signals[:, t - 1]  
            s_exp = np.exp(s)
            xi = s
            if protect_network:
                self_weight = (1 - eta * (2 - a_diag))
                d = np.random.laplace(np.maximum(a_max, 1 / s_exp) / eps)
                mu[:, t] = self_weight * mu[:, t - 1] + eta * xi + (eta * A_ndiag) @ mu[:, t - 1]
                nu[:, t] = self_weight * nu[:, t - 1] + eta * xi + (eta * A_ndiag) @ nu[:, t - 1] + eta * d
            else:
                d = np.random.laplace(1/(eps * s_exp))
                mu[:, t] = (1 - eta) * (A @ mu[:, t - 1]) + eta * xi
                nu[:, t] = (1 - eta) * (A @ nu[:, t - 1]) + eta * (xi + d)
        else:
            if t == 1:
                s = signals[:, 0]
                s_exp = np.exp(s)
                xi = s
                if protect_network:
                    d = np.random.laplace(np.maximum(a_max, 1 / s_exp) / eps)
                else:
                    d = np.random.laplace(scale=1/(s_exp * eps))
                mu[:, t] = xi
                nu[:, t] = xi + d
            else:
                nu[:, t] = A @ nu[:, t - 1]
                mu[:, t] = A @ mu[:, t - 1]
            

        V_mu[t] = V(mu[:, t])
        V_nu[t] = V(nu[:, t])


    return mu, nu

def sample_path_plot(A, n, T, l, eps, signals=None, intermittent=True, name='', protect_network=False):
    mu, nu = mean_estimation(A, n, T, l, eps, signals, intermittent, protect_network=protect_network)
    
    if signals is None:
        mu_theta = l
    else:
        if intermittent:
            mu_theta = signals.mean()
        else:
            mu_theta = signals[0, :].mean()

    error_mu = np.sqrt(np.sum((mu - mu_theta)**2, 0))
    error_nu = np.sqrt(np.sum((mu - mu_theta)**2, 0))

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))

    if intermittent:
        plt.suptitle(f"Online Learning of Expected Values ({get_title(name)}) with {'Network' if protect_network else 'Signal'} DP")
    else:
        plt.suptitle(f"Minimum Variance Unbiased Estimation ({get_title(name)}) with {'Network' if protect_network else 'Signal'} DP")

    for i in range(n):
        ax[0].plot(mu[i, 1:])
        ax[1].plot(nu[i, 1:])

    ax[1].set_xlabel('Round $t$')

    ax[0].set_ylabel('Sample paths $\\mu_{i, t}$')
    ax[1].set_ylabel('Sample paths $\\nu_{i, t}$')
    ax[2].set_ylabel('Mean Squared Error')

    ax[0].set_title('Estimates without DP')
    ax[1].set_title(f'Estimates with DP ($\epsilon = {eps}$)')


    ax[2].plot(error_mu, label='$|| \\bar {\\mu}_t - \\bar {m_{\\theta}} ||_{2}$')
    ax[2].plot(error_nu, label='$|| \\bar {\\nu}_t - \\bar {m_{\\theta}} ||_{2}$')
    ax[2].legend()

    ax[2].set_xscale('log')
    ax[2].set_yscale('log')

    plt.tight_layout()

    plt.savefig(f"figures/sample_paths_{'intermittent' if intermittent else 'initial'}{'_network' if  protect_network else ''}_{name}.pdf")


def mse_plot(A, n, T, l, n_sim=10, signals=None, eps_conv=1e-3, name=''):

    if signals is None:
        mu_theta_mvue = l
        mu_theta_ol = l
    else:
        mu_theta_mvue = signals[0, :].mean()
        mu_theta_ol = signals.mean()

    eps_range = 2**np.arange(-4, 4).astype(np.float64)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_ylabel('MSE (log)')
    ax.set_xlabel('$\\epsilon$ (log)')
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_title(f'MSE vs. Privacy Budget ({get_title(name)})')

    for intermittent in [True, False]:
        for protect_network in [True, False]:
            mse_omni = np.zeros((n_sim, eps_range.shape[0]))
            mse = np.zeros((n_sim, eps_range.shape[0]))
            t_conv = np.zeros((n_sim, eps_range.shape[0]))
            for s in range(n_sim):
                for i, eps in enumerate(eps_range):
                    mu, nu = mean_estimation(A=A, n=n, T=T, l=l, eps=eps, signals=signals, intermittent=intermittent, protect_network=protect_network)
                    
                    if not intermittent:
                        mse_omni[s, i] = np.sqrt(np.sum((nu[:, -1] - mu_theta_mvue)**2))
                    else:
                        mse_omni[s, i] = np.sqrt(np.sum((nu[:, -1] - mu_theta_ol)**2))

                    mse[s, i] = np.sqrt(np.sum((nu[:, -1] - mu[:, -1])**2))

            mse_omni_mean = mse_omni.mean(0)
            mse_mean = mse.mean(0)

            if intermittent and protect_network:
                label = 'OL w/ Network DP'
                color = 'r'
            elif intermittent and not protect_network:
                label = 'OL w/ Signal DP'
                color = 'g'    
            elif not intermittent and protect_network:
                label = 'MVUE w/ Network DP'
                color = 'b'
            else:
                label = 'MVUE w/ Signal DP'
                color = 'k'
    
            ax.plot(eps_range, mse_omni_mean, label=f'{label} (TE)', color=color, linestyle='dashed')
            ax.plot(eps_range, mse_mean, label=f'{label} (CoP)', color=color)
        

    ax.legend()
    
    plt.tight_layout()

    plt.savefig(f'figures/mse_plot_{name}.pdf')

def get_title(name):
    if name == 'germany_consumption':
        return 'German Households Consumption'
    elif name == 'us_power_grid':
        return 'US Power Grid'
    elif name == 'karate_club':
        return 'Karate Club'
    else:
        return name
    
def visualize(G, signals, name):
    
    plt.figure(figsize=(6, 6))

    plt.axis("off")
    plt.title(get_title(name), fontsize=FONTSIZE)

    if name == 'germany_consumption':
        # position is stored as node attribute data for random_geometric_graph
        pos = nx.get_node_attributes(G, "pos")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        pos = nx.spring_layout(G, k=6/np.sqrt(len(G)))

    node_color = np.mean(signals, 1)

    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=np.arange(len(G)),
        node_size=5,
        node_color=node_color,
        cmap=plt.cm.Reds_r,
    )
        
    plt.savefig(f'figures/{name}_network.pdf')

    degrees = np.array([1 + G.degree(u) for u in G.nodes()])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f'Degree Distribution ({get_title(name)})')

    
    fit = powerlaw.Fit(degrees, xmin=1)

    powerlaw.plot_pdf(degrees, color='b', ax=ax, label='Empirical Distribution')
    fit.lognormal.plot_pdf(color='b', linestyle='--', ax=ax, label=f'Log-Normal Fit ($\\mu = {fit.lognormal.mu:.2f}, \\sigma = {fit.lognormal.sigma:.2f}$)')
    plt.xlabel('Degree', fontsize=FONTSIZE)
    plt.ylabel('Frequency', fontsize=FONTSIZE)
    plt.legend(fontsize=0.5*FONTSIZE)

    plt.savefig(f'figures/{name}_degree_distribution.pdf')



if __name__ == '__main__':
    args = get_argparser()

    G, signals = dataloader.load_dataset(args)

    A, n = build_network(G)
    l = args.l
    eps = args.eps

    if signals is None:
        T = args.T
        signals = np.log(np.random.lognormal(mean=l, sigma=1, size=(n, T)))
    else:
        T = signals.shape[1]

    # visualize(G, signals, args.name)

    print(f'n = {n}, m = {len(G.edges())}, T = {T}')

    # sample_path_plot(A, n, T, l, eps=eps, signals=signals, intermittent=True, name=args.name, protect_network=True)
    # sample_path_plot(A, n, T, l, eps=eps, signals=signals, intermittent=True, name=args.name, protect_network=False)
    # sample_path_plot(A, n, T, l, eps=eps, signals=signals, intermittent=False, name=args.name)
    sample_path_plot(A, n, T, l, eps=eps, signals=signals, intermittent=False, protect_network=True, name=args.name)
    # mse_plot(A, n, T, l, signals=signals, name=args.name)



            









