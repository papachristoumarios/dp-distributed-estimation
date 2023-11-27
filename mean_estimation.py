import networkx as nx
import numpy as np
import random

import matplotlib.pyplot as plt
import argparse
import dataloader
import powerlaw

FONTSIZE = 18

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='karate_club', type=str)
    parser.add_argument('--eps', default=1, type=float)
    parser.add_argument('--delta', default=0.01, type=float)
    parser.add_argument('-T', default=100, type=int)
    parser.add_argument('-l', default=10, type=float, help='Mean of log-normal distribution')
    parser.add_argument('--task', default='sample_path_plot', type=str, help='sample_path or mse_plot', choices=['sample_path_plot', 'mse_plot', 'visualize'])
    parser.add_argument('--protect_network', action='store_true', help='Protect network structure')
    parser.add_argument('--intermittent', action='store_true', help='Intermittent signals')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--method', choices=['ours', 'rizk'], default='ours')
    parser.add_argument('--lr_rizk', default=1e-3, type=float)
    parser.add_argument('--distribute_budget', action='store_true', help='Distribute the budget by using heterogeneous eps')
    parser.add_argument('--eps_upper_bound', default=float('inf'), type=float)

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

def smooth_sensitivity_lognormal(s_exp, eps, delta, a_max, protect_network=False, eta=1.0, distribute_budget=False, eps_upper_bound=float('inf')):
    beta = eps / (2 * np.log(2 / delta))
    S_signal =  1 / (beta * np.exp(1) * s_exp)

    n = s_exp.shape[0]

    if protect_network:
        D = np.maximum(a_max, 2 * eta * S_signal)
    else:
        D = 2 * eta * S_signal

    max_total_power = (1 / eps) * D.sum()
    min_total_power = (1 / eps_upper_bound) * D.sum()

    if distribute_budget:
        eps_new = (n * eps) * np.sqrt(D) / (np.sqrt(D).sum())
        eps_new = np.minimum(eps_new, eps_upper_bound)
        scale = D / eps_new
    else:
        scale = D / eps

    total_power = scale.sum()

    return scale, max_total_power, min_total_power, total_power

def mean_estimation_rizk(A, n, T, l, eps, delta, signals=None, protect_network=False, noise='iid', lr=0.4, distribute_budget=False, eps_upper_bound=float('inf')):

    nu = np.zeros((n, T + 1))
    mu = np.zeros((n, T + 1))
    max_total_power = np.zeros(T + 1)
    min_total_power = np.zeros(T + 1)
    total_power = np.zeros(T + 1)
    
    I = np.eye(n)
    a_diag = np.diag(A)
    A_ndiag = (1 - I) * A
    a_max = A_ndiag.max(0)


    s = signals[:, 0]
    s_exp = np.exp(s)
    xi = s

    nu[:, 0] = xi
    mu[:, 0] = xi

    eps_prime = eps / T

    for t in range(1, T + 1):
        scale, max_total_power[t], min_total_power[t], total_power[t] = smooth_sensitivity_lognormal(s_exp, eps_prime, delta, a_max, protect_network=protect_network, eta=lr, distribute_budget=distribute_budget, eps_upper_bound=eps_upper_bound)
        d = np.random.laplace(loc=0, scale=scale)

        grad_nu = nu[:, t - 1] - xi
        grad_mu = mu[:, t - 1] - xi

        nu[:, t] = A @ nu[:, t - 1] - lr * grad_nu + d
        mu[:, t] = A @ mu[:, t - 1] - lr * grad_mu        
       
    return mu, nu, max_total_power, min_total_power, total_power

def mean_estimation(A, n, T, l, eps, delta, signals=None, intermittent=True, protect_network=False, distribute_budget=False, eps_upper_bound=float('inf')):

    # Mean Estimation
    mu = np.zeros((n, T + 1))
    nu = np.zeros((n, T + 1)) 
    max_total_power = np.zeros(T + 1)
    min_total_power = np.zeros(T + 1)
    total_power = np.zeros(T + 1)
    
    if signals is not None:
        if intermittent:
            mu_theta = signals.mean()
        else:
            mu_theta = signals[:, 0].mean()
    else:
        mu_theta = l
    
    print(f'MEAN ESTIMATE: mu_theta = {mu_theta}')

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
            scale, max_total_power[t], min_total_power[t], total_power[t] = smooth_sensitivity_lognormal(s_exp, eps, delta, a_max, protect_network=protect_network, distribute_budget=distribute_budget, eps_upper_bound=eps_upper_bound)
            d = np.random.laplace(loc=0, scale=scale)

            if protect_network:
                self_weight = (1 - eta * (2 - a_diag))
                mu[:, t] = self_weight * mu[:, t - 1] + eta * xi + (eta * A_ndiag) @ mu[:, t - 1]
                nu[:, t] = self_weight * nu[:, t - 1] + eta * xi + (eta * A_ndiag) @ nu[:, t - 1] + eta * d
            else:
                mu[:, t] = (1 - eta) * (A @ mu[:, t - 1]) + eta * xi
                nu[:, t] = (1 - eta) * (A @ nu[:, t - 1]) + eta * (xi + d)
        else:
            if t == 1:
                s = signals[:, 0]
                s_exp = np.exp(s)
                xi = s
                scale, max_total_power[t], min_total_power[t], total_power[t] = smooth_sensitivity_lognormal(s_exp, eps, delta, a_max, protect_network=protect_network, distribute_budget=distribute_budget, eps_upper_bound=eps_upper_bound)
                d = np.random.laplace(loc=0, scale=scale)
                mu[:, t] = xi
                nu[:, t] = xi + d
            else:
                nu[:, t] = A @ nu[:, t - 1]
                mu[:, t] = A @ mu[:, t - 1]
            

        V_mu[t] = V(mu[:, t])
        V_nu[t] = V(nu[:, t])


    return mu, nu, max_total_power, min_total_power, total_power

def sample_path_plot(A, n, T, l, eps, delta, signals=None, intermittent=True, name='', protect_network=False, method='ours', lr_rizk=1e-3, distribute_budget=False, eps_upper_bound=float('inf')):
    if method == 'ours':
        mu, nu, max_total_power, min_total_power, total_power  = mean_estimation(A, n, T, l, eps, delta, signals, intermittent, protect_network=protect_network, distribute_budget=distribute_budget, eps_upper_bound=eps_upper_bound)
    elif method == 'rizk':
        mu, nu, max_total_power, min_total_power, total_power = mean_estimation_rizk(A, n, T, l, eps, delta, signals, protect_network=protect_network, lr=lr_rizk, distribute_budget=distribute_budget, eps_upper_bound=eps_upper_bound)

    if signals is None:
        mu_theta = l
    else:
        if intermittent:
            mu_theta = signals.mean()
        else:
            mu_theta = signals[0, :].mean()

    error_mu = np.sqrt(np.sum((mu - mu_theta)**2, 0))
    error_nu = np.sqrt(np.sum((nu - mu_theta)**2, 0))

    fourth_plot = int(distribute_budget and intermittent)

    fig, ax = plt.subplots(1, 3 + fourth_plot, figsize=(9 + 3 * fourth_plot, 3))

    if intermittent:
        fig.suptitle(f"Online Learning of Expected Values ({get_title(name)}) with {'Network' if protect_network else 'Signal'} DP{' (Rizk et al. 2023)' if method == 'rizk' else ''}")
    else:
        fig.suptitle(f"Minimum Variance Unbiased Estimation ({get_title(name)}) with {'Network' if protect_network else 'Signal'} DP{' (Rizk et al. 2023)' if method == 'rizk' else ''}")

    for i in range(n):
        ax[0].plot(mu[i, 1:])
        ax[1].plot(nu[i, 1:])

    fig.supxlabel('Round $t$')

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


    if distribute_budget and intermittent:
        ax[3].set_ylabel('Privacy Overhead')

        ax[3].plot(max_total_power[1:], label='Upper Bound')
        ax[3].plot(min_total_power[1:], label='Lower Bound')
        ax[3].plot(total_power[1:], label='Optimal')

        ax[3].set_xlim(0, T - 1)

        ax[3].legend(loc='upper left')

        
    fig.tight_layout()
    fig.savefig(f"figures/sample_paths_{'intermittent' if intermittent else 'initial'}{'_network' if  protect_network else ''}_{name}{'_rizk' if method == 'rizk' else ''}{'_heterogeneous' if distribute_budget else ''}.png", bbox_inches='tight')


def mse_plot(A, n, T, l, delta, n_sim=10, signals=None, eps_conv=1e-3, name='', method='ours', lr_rizk=1e-3, distribute_budget=False, eps_upper_bound=float('inf')):

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

    ax.set_title(f"MSE vs. Privacy Budget ({get_title(name)}){' (Rizk et al. 2023)' if method == 'rizk' else ''}")

    for intermittent in [True, False]:
        for protect_network in [True, False]:
            mse_omni = np.zeros((n_sim, eps_range.shape[0]))
            mse = np.zeros((n_sim, eps_range.shape[0]))
            t_conv = np.zeros((n_sim, eps_range.shape[0]))
            for s in range(n_sim):
                for i, eps in enumerate(eps_range):
                    if method == 'ours':
                        mu, nu, _, _, _ = mean_estimation(A=A, n=n, T=T, l=l, eps=eps, delta=delta, signals=signals, intermittent=intermittent, protect_network=protect_network, distribute_budget=distribute_budget, eps_upper_bound=eps_upper_bound)
                    elif method == 'rizk':
                        mu, nu, _, _, _ = mean_estimation_rizk(A=A, n=n, T=T, l=l, eps=eps, delta=delta, signals=signals, protect_network=protect_network, lr=lr_rizk, distribute_budget=distribute_budget, eps_upper_bound=eps_upper_bound)

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

    plt.savefig(f"figures/mse_plot_{name}{'_rizk' if method == 'rizk' else ''}{'_heterogeneous' if distribute_budget else ''}.png", bbox_inches='tight')

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
        
    plt.savefig(f'figures/{name}_network.png')

    degrees = np.array([1 + G.degree(u) for u in G.nodes()])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f'Degree Distribution ({get_title(name)})')

    
    fit = powerlaw.Fit(degrees, xmin=1)

    powerlaw.plot_pdf(degrees, color='b', ax=ax, label='Empirical Distribution')
    fit.lognormal.plot_pdf(color='b', linestyle='--', ax=ax, label=f'Log-Normal Fit ($\\mu = {fit.lognormal.mu:.2f}, \\sigma = {fit.lognormal.sigma:.2f}$)')
    plt.xlabel('Degree', fontsize=FONTSIZE)
    plt.ylabel('Frequency', fontsize=FONTSIZE)
    plt.legend(fontsize=0.5*FONTSIZE)

    plt.savefig(f'figures/{name}_degree_distribution.png')

if __name__ == '__main__':
    args = get_argparser()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f'PARAMETERS: {str(args)}\n')

    G, signals = dataloader.load_dataset(args)

    A, n = build_network(G)
    l = args.l
    eps = args.eps
    delta = args.delta
    task = args.task
    protect_network = args.protect_network
    intermittent = args.intermittent
    method  = args.method
    lr_rizk = args.lr_rizk
    distribute_budget = args.distribute_budget
    eps_upper_bound = args.eps_upper_bound

    if method == 'rizk' and intermittent:
        raise Exception('Rizk et al. (2023) does not support intermittent signals')

    if signals is None:
        T = args.T
        signals = np.log(np.random.lognormal(mean=l, sigma=1, size=(n, T)))
    else:
        T = signals.shape[1]

    print(f'NETWORK STATISTICS: n = {n}, m = {len(G.edges())}, T = {T}\n')

    if task == 'sample_path_plot':
        sample_path_plot(A, n, T, l, eps=eps, delta=delta, signals=signals, intermittent=intermittent, name=args.name, protect_network=protect_network, method=method, lr_rizk=lr_rizk, distribute_budget=distribute_budget, eps_upper_bound=eps_upper_bound)
    elif task == 'mse_plot':
        mse_plot(A, n, T, l, delta=delta, signals=signals, name=args.name, method=method, lr_rizk=lr_rizk, distribute_budget=distribute_budget, eps_upper_bound=eps_upper_bound)
    elif task == 'visualize':
        visualize(G, signamls=signals, name=args.name)
    else:
        raise Exception('Invalid task')

            









