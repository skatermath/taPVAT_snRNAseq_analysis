import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.stats import ttest_ind

### Color palettes
### 8 color palettes
custom_palette1 = ['steelblue', 'firebrick', 'darkgreen', 'darkviolet', 'darkorange', 'olive', 'powderblue', 'pink']
dark_palette = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2", "#937860", "#da8bc3", "#8c8c8c"]
bright_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
### 28 color palette
large_palette = ["lightseagreen", "mediumseagreen", "darkseagreen",
                   "lightcoral", "tomato", "darkred",
                   "lightsteelblue", "lightblue", "steelblue",
                   "peachpuff", "goldenrod", "darkgoldenrod",
                   "lightgreen", "green", "darkgreen",
                   "lightgrey", "darkgrey", "grey",
                   "lavenderblush", "pink", "hotpink",
                   "lightcyan", "cyan", "darkcyan",
                   "thistle",  "mediumorchid", "purple",
                   "orange" ]

### Pretty way to graph cell-type marker gene scores
def plot_marker_genes_group(adata, group, marker_gene_list, leiden='leiden', layer=None, use_raw=True, 
                            save_fig=True, output_path="../output/taPVAT_UMAP_violin_plot_for_"):
    """
    Generates and optionally saves UMAP and violin plots for gene expression in a given cell type.

    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix.
    cell_name : str
        Name of the cell type for which the marker genes are being analyzed.
    marker_gene_list : list
        List of marker genes to score and plot.
    leiden : str, optional
        Column name in `adata.obs` that contains the Leiden clustering results. Default is 'leiden'.
    layer : str or None, optional
        Key in `adata.layers` to use for the plot. Default is None.
    use_raw : bool, optional
        Whether to use raw attribute of `adata` if present. Default is True.
    save_fig : bool, optional
        If True, saves the generated figure to the specified output path. Default is True.
    output_path : str, optional
        Path where the figure will be saved if `save_fig` is True. Default is "../output/".

    Returns:
    --------
    None
        Displays the generated plots and optionally saves the figure.
    """
    
    # Create figure and gridspec
    fig = plt.figure(figsize = (16,10))
    widths = [1, 1]
    heights = [2, 1]

    gs = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    
    # Score genes
    score_name = f"{group}_score"
    sc.tl.score_genes(adata, marker_gene_list, score_name=score_name)

    # Plot UMAP with Leiden clustering
    sc.pl.umap(adata, ax=ax1, color=leiden, size=50, add_outline=True, legend_loc='on data',
               legend_fontsize=12, legend_fontoutline=2, show=False, layer=layer,
               title='Leiden Clustering UMAP')

    # Plot UMAP with marker gene score
    sc.pl.umap(adata, ax=ax2, color=[score_name], size=100, add_outline=True, show=False,
               layer=layer, title=f'Score for {group} Cell Markers')

    # Plot violin plot for marker gene score
    sc.pl.violin(adata, [score_name], ax=ax3, groupby=leiden, inner='quartiles', stripplot=False,
                 show=False, layer=layer, use_raw=use_raw)

    # Save the figure if save_fig is True
    if save_fig:
        plt.savefig(f"{output_path}{group}_marker_gene_expression.png")
    
    # Show the plot
    plt.show()

    
### This makes it so individual genes can be plotted with the marker_gene_graph function
def plot_marker_genes(adata, marker_gene_list, leiden = 'leiden', layer = None, use_raw = True, save_fig = True,
                      output_path = "../output/taPVAT_UMAP_violin_plot_for_"):
    for i in marker_gene_list:
        print(i)
        group = str(i)
        marker_genes = [i]
        if group in adata.raw.var.index:
            plot_marker_genes_group(adata, group, marker_genes, leiden, layer, use_raw, save_fig, output_path)
        else:
            print(group, 'is not in the dataset')
            
            
###Makes a bar plot of proportional data with error bars
def proportion_plot(adata, group, group1, group2, leiden_col, sample_id_col, error_bars = True, 
                    save_fig = True, output_name = "", output_path = '../output/taPVAT_proportions_labeled_'):
    """
    Generates a bar plot of proportional data with error bars for two groups within an AnnData object.

    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix.
    group : str
        The name of the grouping variable in `adata.obs` to compare proportions between groups.
    group1 : str
        The first group to compare.
    group2 : str
        The second group to compare.
    leiden_col : str
        The column name in `adata.obs` that contains the clustering results (e.g., Leiden clustering).
    sample_id_col : str
        The column name in `adata.obs` that contains sample identifiers.
    error_bars : bool, optional
        If True, includes error bars in the plot. Default is True.
    save_fig : bool, optional
        If True, saves the generated figure to the specified output path. Default is True.
    output_name : str, optional
        Additional string to append to the output file name if `save_fig` is True. Default is an empty string.
    output_path : str, optional
        Path where the figure will be saved if `save_fig` is True. Default is '../output/taPVAT_proportions_labeled_'.

    Returns:
    --------
    proportions_df1 : DataFrame
        DataFrame containing proportions for `group1`.
    proportions_df2 : DataFrame
        DataFrame containing proportions for `group2`.
    """
    
    adata_1 = adata[adata.obs[group] == group1]
    adata_2 = adata[adata.obs[group] == group2]

    # Calculate proportions of each value within the "leiden" column for each subset
    proportions_split1 = adata_1.obs.groupby([sample_id_col, leiden_col])[leiden_col].count() / adata_1.obs.groupby(sample_id_col)[leiden_col].count()
    proportions_split2 = adata_2.obs.groupby([sample_id_col, leiden_col])[leiden_col].count() / adata_2.obs.groupby(sample_id_col)[leiden_col].count()

    # Fill missing values with zeros
    proportions_split1 = proportions_split1.unstack(fill_value=0).stack()
    proportions_split2 = proportions_split2.unstack(fill_value=0).stack()

    # Combine proportions into a single DataFrame for plotting
    proportions_df1 = proportions_split1.reset_index(name='Proportion')
    proportions_df1[group] = group1
    proportions_df2 = proportions_split2.reset_index(name='Proportion')
    proportions_df2[group] = group2

    # Calculate the standard error of the mean for each 'celltype' and 'Tissue' combination
    std_error_values_df1 = proportions_df1.groupby([leiden_col, group])['Proportion'].sem()
    std_error_values_df2 = proportions_df2.groupby([leiden_col, group])['Proportion'].sem()
    std_error_values = pd.concat([std_error_values_df1, std_error_values_df2])
    
    N = len(adata_1)

    props_dict = {leiden_col:[], "Proportion":[], group:[]}
    for i in np.unique(adata_1.obs[leiden_col]):
        i_len = len(adata_1[adata_1.obs[leiden_col] == i])
        props_dict[leiden_col].append(i)
        props_dict["Proportion"].append(i_len/N)
        props_dict[group].append(group1)
    props_df_1 = pd.DataFrame(props_dict)

    N = len(adata_2)

    props_dict = {leiden_col:[], "Proportion":[], group:[]}
    for i in np.unique(adata_2.obs[leiden_col]):
        i_len = len(adata_2[adata_2.obs[leiden_col] == i])
        props_dict[leiden_col].append(i)
        props_dict["Proportion"].append(i_len/N)
        props_dict[group].append(group2)
    props_df_2 = pd.DataFrame(props_dict)


    props_df = pd.concat([props_df_1, props_df_2])
    props_df['std_error'] = list(std_error_values)

    dfp = props_df.pivot(index=leiden_col, columns=group, values='Proportion')
    yerr = props_df.pivot(index=leiden_col, columns=group, values='std_error')
    
    # Define the order of groups for plotting
    group_order = [group1, group2]
    dfp = dfp[group_order]
    yerr = yerr[group_order]

    ### Generate the plot
    plt.style.use("seaborn-white")
    fig, ax = plt.subplots(figsize = (16,6))

    # Define custom error bar properties
    error_kw = {
        'capsize': 5,  # Adjust cap size
        'elinewidth': 1.5,  # Adjust error bar line width
        'ecolor': 'black'  # Set error bar color
    }
    ### Set color palette
    seaborn_palette = sns.color_palette("deep", n_colors=2)

    # Convert Seaborn palette to Matplotlib colormap
    cmap = ListedColormap(seaborn_palette)
    
    if error_bars:
        g = dfp.plot(kind='bar', yerr=yerr, rot=0, ax=ax, error_kw=error_kw, width = 0.7, colormap = cmap)
    else:
        g = dfp.plot(kind='bar', rot=0, ax=ax, width = 0.7, colormap = cmap)
    plt.xticks(rotation = 90, size = 24)
    plt.yticks(size = 20)
    plt.xlabel("", size = 24)
    plt.ylabel("Relative Proportion", size = 24)
    plt.legend(fontsize = 24)
    g.set_yscale("log")
    
    ### Creating automated annotations
    dft1 = proportions_df1.pivot(index=sample_id_col, columns=leiden_col, values='Proportion')
    dft2 = proportions_df2.pivot(index=sample_id_col, columns=leiden_col, values='Proportion')

    p_list = []
    # Perform t-test and collect p-values
    for column in dft1.columns:
        if column in dft2.columns:
            t_stat, p_val = ttest_ind(dft1[column], dft2[column], equal_var=False)
            p_list.append(p_val)
        else:
            p_list.append(np.nan)
            print("group2 does not contain " + column)

    def annotate_graph(g = g, label = "label"):
        if leiden_col == 'celltype_broad':
            fontsize = 24
        else:
            fontsize = 16
        ###g.annotate("p =" + format(p, '.3f') + " ***",
        g.annotate(label,
                   (patch.get_x() + patch.get_width(), g.get_ylim()[0]), 
                   ha='center', va='bottom', 
                   fontsize = fontsize,
                   weight = 'bold',
                   color = 'white',
                   xytext=(0, 0), 
                   textcoords='offset points')
        
    # Annotate bars with p-values
    for p, patch in zip(p_list, g.patches):
        if p < 0.001: 
            annotate_graph(g, "***")
        elif p < 0.01:
            annotate_graph(g, "**")
        elif p < 0.05:
            annotate_graph(g, "*")
        else:
            annotate_graph(g, "")

    if save_fig:
        plt.savefig(f"{output_path}{group}{output_name}.png", bbox_inches = "tight")
    return proportions_df1, proportions_df2