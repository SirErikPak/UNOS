# import plot libries
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import numpy as np
import scipy.stats as stats



def plot_qq(data, features_list):
    """
    Generate Q-Q plots for multiple features in a dataset.

    Parameters:
    data : pandas.DataFrame
        The dataset containing the features to be plotted.
    features_list : list of str
        A list of column names in `data` for which Q-Q plots are to be generated.

    Returns:
    None
        Displays Q-Q plots for the specified features.
    """
    # Calculate the number of rows needed for subplots
    n_rows = (len(features_list) + 1) // 2  # Two plots per row

    # Create subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(16, 4 * n_rows))
    fig.suptitle("Q-Q Plots", fontsize=16)

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    # Generate Q-Q plots
    for ax, feature in zip(axes, features_list):
        stats.probplot(data[feature].dropna(), dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot: {feature}")

    # Remove any unused subplots
    for ax in axes[len(features_list):]:
        fig.delaxes(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate the main title
    plt.show()



def plot_missingness(data_series):
    """
    Visualizes missing data in a pandas Series or DataFrame using a matrix plot.

    Parameters:
    ----------
    data_series : pandas.Series or pandas.DataFrame
        The data to visualize for missing values.

    Returns:
    -------
    None
    """
    msno.matrix(data_series)
    plt.show()



def plot_count(data, cat_list, txt='', fig_size=(10, 4), label_font_size=10, title_font_size=11, annotation_font_size = 11, legend=False):
    # matplotlib figure
    num_cols = len(cat_list)
    fig_height = fig_size[1] * num_cols  # Adjust height based on number of categories
    plt.figure(figsize=(fig_size[0], fig_height))
    
    # iterate each categorical column and create a subplot
    for i, column in enumerate(cat_list, start=1):
        plt.subplot(num_cols, 1, i)  # create a subplot for each category
        ax = sns.countplot(data=data, y=column, hue=column, legend=legend)
        ax.grid(False)  # remove grid
        
        # customize each plot
        plt.title(f"Count Plot for {column} {txt}", fontsize=title_font_size)
        plt.xlabel("Count", fontsize=label_font_size)
        plt.ylabel(column, fontsize=label_font_size)
    
        # display counts on top of each bar
        for p in ax.patches:  # iterate over the bars
            ax.annotate(f'{int(p.get_width()):,}', 
                        (p.get_width() + 100, p.get_y() + p.get_height() / 2), 
                        ha='left', va='center', fontsize=annotation_font_size)
    
    # adjust the vertical space between subplots
    plt.subplots_adjust(hspace=0.4)

    # show the plot
    plt.tight_layout()
    plt.show()



def plot_histogram(data, lst, bins=30, txt='', title_font=15, label_font=10, tick_font=10, KDE=True):
    """
    The function histogramPlot is well-designed for plotting histograms of multiple columns in a single figure. 
    """
    # calculate the number of rows needed (two plots per row)
    num_cols = min(len(lst), 2)  # max 2 columns
    num_rows = int(np.ceil(len(lst) / num_cols))  # calculate number of rows
    
    # set up the matplotlib figure
    plt.figure(figsize=(10 * num_cols, 5 * num_rows))  # adjust the figure size as needed
    
    # Iterate each categorical column and create a subplot
    for i, column in enumerate(lst):
        plt.subplot(num_rows, num_cols, i + 1)  # create a subplot for each numeric
        ax = sns.histplot(data=data, x=column, kde=KDE, bins=bins)
        ax.grid(False)  # remove grid
        
        # customize each plot
        plt.title(f'Histogram Plot for {column} {txt}', fontsize=title_font)
        plt.xlabel(column, fontsize=label_font, fontweight='bold')
        plt.ylabel('Frequency', fontsize=label_font, fontweight='bold')
        plt.xticks(fontsize=tick_font)
        plt.yticks(fontsize=tick_font)
    
    # adjust the space between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Show the plot
    plt.tight_layout()
    plt.show()



def plot_categorical_feature_count(data, feature_cat, txt='', fig_size=(25, 15), tick_font=13, title_font=12):
    """
    Create count plots for multiple categorical features.
    """
    # calculate the number of features and the grid dimensions
    num_features = len(feature_cat)
    cols = 2  # set the number of columns
    rows = (num_features + cols - 1) // cols  # Calculate the required rows

    # set the figure size
    plt.figure(figsize=fig_size)

    # create count plots for each categorical feature
    for i, feature in enumerate(feature_cat, start=1):
        plt.subplot(rows, cols, i)  # Create subplot dynamically based on rows and columns
        sns.countplot(data=data, y=feature, hue=feature, palette='Set2', legend=False)  # Use hue with the feature
        plt.title(f'\n{feature} {txt}', fontsize=title_font)
        plt.xlabel("Frequency", fontsize=tick_font, fontweight='bold')
        plt.ylabel(feature, fontsize=tick_font, fontweight='bold')
        plt.xticks(fontsize=tick_font)
        plt.yticks(fontsize=tick_font)
        
    # adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()



def plot_multiple_box(df, numeric_col, categorical_col, hue_parameter, width=8, height=6):
    """
    Generate a boxplot of with one mumerical value.
    """
    # figure with the desired size
    plt.figure(figsize=(width, height))
    # create the boxplot
    sns.boxplot(data=df, x=numeric_col, y=categorical_col, hue=hue_parameter)

    # customize the legend
    plt.legend(title=f"{hue_parameter}", loc='upper right', fontsize=10, frameon=True)

    # add title
    plt.title(f'Box Plot of {numeric_col} by {categorical_col} and {hue_parameter}\n')

    # show the plot
    plt.show()



def safe_exp(data, max_value=700):
    """
    Compute the exponential safely to avoid overflow.

    Args:
        data: Input numeric data (scalar, list, or numpy array).
        max_value: Threshold to clip input values, preventing overflow.

    Returns:
        Exponential of the input, clipped to avoid overflow.
    """
    # Clip input values to avoid overflow beyond np.exp(700)
    data = np.clip(data, -max_value, max_value)
    return np.exp(data)



def plot_transform_distribution(data, txt='', bins=30, fig_size=(20, 10)):
    """
    Plots histograms of the original data, log-transformed data, 
    square root-transformed data, and exponentially-transformed data 
    in a single row of subplots.
    """
    # Transformations
    log_data = np.log(data + 1)  # Adding 1 to avoid log(0)
    sqrt_data = np.sqrt(data) # Square root transformation
    square_data = np.square(data) # Square transformation
    exp_data = safe_exp(data)  # Exponential transformation

    # Creating subplots
    fig, axes = plt.subplots(1, 5, figsize=fig_size, sharey=True)
    
    # Plotting the histograms
    axes[0].hist(data, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_title(f"Original Data - ({txt})")
    axes[0].set_xlabel(f"{txt}")
    axes[0].set_ylabel("Frequency")
    
    axes[1].hist(log_data, bins=bins, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_title(f"Log Transformed Data - ({txt})")
    axes[1].set_xlabel(f"Log({txt} + 1)")
    
    axes[2].hist(sqrt_data, bins=bins, color='orange', alpha=0.7, edgecolor='black')
    axes[2].set_title(f"Square Root Transformed Data - ({txt})")
    axes[2].set_xlabel(f"Sqrt({txt})")
   
    axes[3].hist(square_data, bins=bins, color='cyan', alpha=0.7, edgecolor='black')
    axes[3].set_title(f"Square Transformed Data - ({txt})")
    axes[3].set_xlabel(f"Square({txt})")

    axes[4].hist(exp_data, bins=bins, color='red', alpha=0.7, edgecolor='black')
    axes[4].set_title(f"Exponential Transformed Data - ({txt})")
    axes[4].set_xlabel(f"Exp({txt})")
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()



def plot_density_transform_distribution(data, txt='', bins=30, fig_size=(15, 5), KDE=True):
    """
    Plots histograms of the original data, log-transformed data, 
    and square root-transformed data in a single row of subplots,
    with optional KDE overlays.
    """
    # Transformations
    log_data = np.log(data + 1)  # Adding 1 to avoid log(0)
    sqrt_data = np.sqrt(data)
    
    # Creating subplots
    fig, axes = plt.subplots(1, 3, figsize=fig_size, sharey=False)
    
    # Plotting the histograms with optional KDE overlays
    for ax, transformed_data, title, xlabel, color in zip(
        axes,
        [data, log_data, sqrt_data],
        ["Original Data", "Log Transformed Data", "Square Root Transformed Data"],
        [f"{txt}", f"Log({txt} + 1)", f"Sqrt({txt})"],
        ['blue', 'green', 'orange']
    ):
        ax.hist(transformed_data, bins=bins, color=color, alpha=0.7, edgecolor='black', density=True)
        
        # Add KDE only if KDE=True
        if KDE:
            sns.kdeplot(transformed_data, ax=ax, color='red', lw=2)
        
        ax.set_title(f"{title} - ({txt})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()



def plot_box(data, columns, orientation='h', fig_size=(15, 5)):
    """
    Plots boxplots for specified columns in a DataFrame.

    Parameters:
    data (DataFrame): The DataFrame containing the data to plot.
    columns (list): A list of column names to plot.
    orientation (str): Orientation of the boxplots - 'h' for horizontal, 'v' for vertical. Default is 'h'.
    figsize (tuple): Size of the figure as (width, height). Default is (15, 5).

    Returns:
    None
    """
    # Set the figure size
    plt.figure(figsize=fig_size)
    
    # Determine the number of rows needed for subplots
    num_columns = len(columns)
    num_rows = (num_columns + 1) // 2
    
    # Iterate through each specified column and create a boxplot
    for i, column in enumerate(columns, 1):
        plt.subplot(num_rows, 2, i)
        if orientation == 'v':
            sns.boxplot(y=data[column])
        else:
            sns.boxplot(x=data[column])
        plt.title(f'Boxplot of {column}')
        plt.tight_layout()
    
    # Display the plots
    plt.show()