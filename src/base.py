import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from math import ceil

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes,locatable, axes_size
from sklearn.manifold import TSNE
import matplotlib.cm as cm

import os
import json

def get_tsne_data(tsne_dir, perp):
    coords = pd.read_csv(f"{tsne_dir}/tsne_{perp}.csv", low_memory=False)
    df = pd.read_csv(f"{tsne_dir}/all_columns.csv", low_memory=False)
    df = pd.concat([df, coords[['X','Y']]], axis=1)
    return df

def compute_tsne_datasets(input_df, output_dir, columns_exclude, n_components_pca=40, perp_list = [10, 25, 50], n_iter = 250, learning_rate = 'auto'):
    df_dropped = input_df.drop(columns_exclude, axis=1)
    scaled = StandardScaler().fit_transform(df_dropped)
    if n_components_pca:
        pca = PCA(n_components_pca)
        final = pca.fit_transform(scaled)

    else:
        final = scaled
    
    for perp in perp_list:
        tsne_df = perform_tsne(final, perp=perp, n_iter = n_iter, learning_rate = learning_rate)
        tsne_df.to_csv(f"{output_dir}/tsne_{perp}.csv", index = False)

    input_df.to_csv(f"{output_dir}/all_columns.csv", index=False)

def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cbar = fig.colorbar(mappable, cax = cax)
    plt.sca(last_axes)
    return cbar

def read_from_pickle(filename):
    return pickle.load(open(f'./stored_indices/{filename}.pkl', 'rb'))

def get_tsne_plot(output_dir, perp, group_column=None, selected_groups=None, s = 0.05):
    df = get_tsne_data(output_dir, perp[0])
    fig, ax = plt.subplots(figsize = (6,6))

    markerscale = 4 if s >1 else 1/s
    group_column = group_column[0] if group_column else None

    if group_column and selected_groups:
        print("Can't plot both groups and selected indices, use either --groups or --selected, not both!")
        return

    if selected_groups:
        selected_colors = ['red', 'blue', 'yellow', 'pink', 'grey']
        df['selected'] = 'Other'
        indices_dict = {}

        for group in selected_groups:
            indices = read_from_pickle(group)
            list_indices = list(indices)
            indices_dict[group] = list_indices
            df.loc[list_indices, 'selected'] = group

        for i, group in enumerate(selected_groups):
            indices = indices_dict[group]
            ax.scatter(df.iloc[indices]['X'], df.iloc[indices]['Y'], color=selected_colors[i], s=0.05, edgecolor=None, label=group)
        ax.scatter(df.loc[df['selected'] == 'Other']['X'], df.loc[df['selected'] == 'Other']['Y'], color = 'green', s = 0.05, edgecolor = None, label='Other')
        ax.legend(loc = 'lower left', title = group_column, markerscale = markerscale)

    elif group_column:
        group_type = 'continuous' if df[group_column].unique().shape[0] > 20 else 'discrete'
        if group_type == 'discrete':
            groups = df[group_column].unique()
            colormap = cm.get_cmap('tab20', len(groups))
            for i, group in enumerate(groups):
                group_df = df[df[group_column] == group]
                ax.scatter(group_df['X'], group_df['Y'], color=colormap(i), s=0.05, edgecolor=None, label=group)
                ax.legend(loc='lower left', title=group_column, markerscale = markerscale)

        elif group_type == 'continuous':
            colormap = cm.get_cmap('RdBu_r')
            sc = ax.scatter(df['X'], df['Y'], c = df[group_column], cmap = colormap, s=0.05, edgecolor = None)
            colorbar(sc)

    else:
        ax.scatter(df['X'], df['Y'], s=0.05, edgecolor=None)

    plt.suptitle(f't-SNE - Perplexity = {perp[0]}')
    if group_column:
        plt.title(group_column)
    plt.show()


def perform_tsne(input_df, perp = 25, n_iter = 1000, learning_rate = 'auto'):

    if n_iter:
        tsne = TSNE(n_components = 2, init = 'random', perplexity = perp, random_state = 42, n_iter_without_progress = 300, learning_rate = learning_rate, n_iter = 10000)
    else:
        tsne = TSNE(n_components = 2, init = 'random', learning_rate = 'auto', perplexity = perp, random_state = 42)

    X = tsne.fit_transform(input_df)
    print(f"TSNE ran for {tsne.n_iter} iterations")
    print(f"TSNE ran with learning rate {tsne.learning_rate_}")

    df = pd.DataFrame(X, columns = ['X', 'Y'])
    return df

def get_tsne_plots_perplexities(tsne_dir, perplexities = None, group_columns = None, selected_groups = None, s = 0.05):
    dfs = [get_tsne_data(tsne_dir, perp) for perp in perplexities]
    markerscale = 4 if s >1 else 1/s
    
    group_column = group_columns[0] if group_columns else None

    n = len(perplexities)
    n_rows = ceil(n/2)
    n_columns = 2 if n > 1 else 1
    fig, axes = plt.subplots(n_rows, n_columns, figsize = (n_columns*7, n_rows* 7))

    if n == 1 :
        axes = np.array([[axes]])
    elif n == 2:
        axes = axes.reshape(1, -1)

    if group_column and selected_groups:
        print("Can't plot btoh groups and selected indices, use either --groups or --selected, not both!")
        return
    
    if selected_groups:
        selected_colors = ['red', 'blue', 'yellow', 'pink', 'grey']
        indices_dict = {}

        for df in dfs:
            df['selected'] = 'Other'
            for group in selected_groups:
                indices = read_from_pickle(group)
                list_indices = list(indices)
                indices_dict[group] = list_indices
                df.loc[list_indices, 'selected'] = group

        for i, df in enumerate(dfs):
            df = dfs[i]
            ax_idx1 = i // 2
            ax_idx2 = i % 2
            axes[ax_idx1][ax_idx2].scatter(df.loc[df['selected'] == 'Other']['X'], df.loc[df['selected'] == 'Other']['Y'], color = 'green', s = 0.05, edgecolor = None, label = 'Other')
            for j, group in enumerate(selected_groups):
                axes[ax_idx1][ax_idx2].scatter(df.ilpc[indices_dict[group]]['X'], df.iloc[indices_dict[group]]['Y'], color = selected_colors[j], s = s, edgecolor = None, label = group)
                axes[ax_idx1][ax_idx2].legend(loc = 'lower_right', title = 'Legend', markerscale = markerscale)
                axes[ax_idx1][ax_idx2].set_title(f'Perplexity = {perplexities[i]}')
            
            fig.suptitle(f't-SNE at various perplexities', fontsize = 16)
            fig.tight_layout()
            fig.savefig('./fig2.png')

    elif group_column:
        group_type = 'continuous' if dfs[0][group_column].unique().shape[0] > 20 else 'discrete'

        for i, df in enumerate(dfs):
            df = dfs[i]
            groups = df[group_column].unique()
            ax_idx1 = i // 2
            ax_idx2 = i % 2

            if group_type == 'discrete':
                colormap = cm.get_cmap('tab20', len(groups))
                for j, group in enumerate(groups):
                    group_df = df[df[group_column] == group]
                    axes[ax_idx1][ax_idx2].scatter(group_df['X'], group_df['Y'], color=colormap(j), s=s, edgecolor = None, label = group)
                    axes[ax_idx1][ax_idx2].legend(loc = 'lower right', title = group_column, markerscale = markerscale)
                    axes[ax_idx1][ax_idx2].set_title(f'Perplexity = {perplexities[i]}')

            else:
                colormap = cm.get_cmap('RdBu_r')
                sc = axes[ax_idx1][ax_idx2].scatter(df['X'], df['Y'], c = df[group_column], cmap = colormap, s=s, edgecolor = None)
                colorbar(sc)
                axes[ax_idx1][ax_idx2].set_title(f'Perplexity = {perplexities[i]}')
            
            fig.suptitle(f'{group_column} at various perplexities', fontsize = 16)
            fig.tight_layout()
        else:
            for i, df in enumerate(dfs):
                ax_idx1 = i // 2
                ax_idx2 = i % 2
                axes[ax_idx1][ax_idx2].scatter(df['X'], df['Y'], edgecolor = None, s=s)
                axes[ax_idx1][ax_idx2].set_title(f'Perplexity = {perplexities[i]}')
            
            fig.suptitle(f't-SNE at various perplexities', fontsize = 16)
            fig.tight_layout()
            fig.savefig('./fig2.png')

        plt.show()

    def get_tsne_plots_groups(tsne_dir, perplexity = None, group_columns = None, s = 0.05):
        
        df = get_tsne_data(tsne_dir, perplexity)
        markerscale = 4 if s > 1 else 1/s

        n = len(group_columns)
        n_rows = ceil(n/2)
        n_columns = 2 if n > 1 else 1
        fig, axes = plt.subplots(n_rows, n_columns, figsize = (n_columns*7, n_rows*7))


        if n == 1:
            axes = np.array([[axes]])
        elif n == 2:
            axes = axes.reshape(1, -1)

        for i, group_column in enumerate(group_columns):
            group_type = 'continuous' if df[group_column].unique().shape[0] > 20 else 'discrete'
            groups = df[group_column].unique()

            ax_idx1 = i // 2
            ax_idx2 = i % 2

            if group_type == 'discrete':
                colormap = cm.get_cmap('tab20', len(groups))
                for j, group in enumerate(groups):
                    group_df = df[df[group_column] == group]
                    axes[ax_idx1][ax_idx2].scatter(group_df['X'], group_df['Y'], color=colormap(j), s=0.05, edgecolor = None, label = group)
                    axes[ax_idx1][ax_idx2].legend(loc = 'lower right', title=group_column, markerscale = markerscale)
                    axes[ax_idx1][ax_idx2].set_title(f'Grouped by {group_column}')
            
            else:
                colormap = cm.get_cmap('RdBu_r')
                sc = axes[ax_idx1].scatter(df['X'], df['Y'], c = df[group_column], cmap = colormap, s=0.05, edgecolor = None)
                colorbar(sc)
                axes[ax_idx1][ax_idx2].set_title(f'Grouped by {group_column}')

            fig.suptitle(f'Perplexity = {perplexity}', fontsize = 16)
            fig.tight_layout()
            fig.savefig('./fig.png')
            plt.show()

    def is_previously_computed(output_dir, perp):
        path = f"{output_dir}/tsne_{perp}.csv"
        return os.path.isfile(path)
        
    def get_filters_with_key(key):
        filters_path = f'./keys/{key}/filters.json'
        if os.path.exists(filters_path):
            with open(filters_path, 'r') as f:
                filters = json.load(f)
            return pd.DataFrame(filters)
        else:
            return None
            
    def apply_filters(df, filter_df):
        for _, filter_row in filter_df.iterrows():
            column = filter_row['Column']
            filter_type = filter_row['Type']
            values = filter_row['Values']

            if filter_type == 'number':
                lower_bound, upper_bound = map(float, values.split(' to '))
                filtered_df = df[(df[column].astype(float) >= lower_bound) & (df[column].astype(float) <= upper_bound)]
            
            elif filter_type == 'category':
                selected_categories = values.split(', ')
                filtered_df = df[df[column].isin(selected_categories)]

            return filtered_df
    

        



