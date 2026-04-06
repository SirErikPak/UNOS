import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
from wordcloud import WordCloud


def plot_informative_missingness(data, col, target='TransplantSurvivalDay', unknown_val=None):
    """
    Visualize how missingness in a feature relates to a survival outcome.
    Creates a violin + strip plot comparing Known vs Missing/Unknown groups.
    """

    # Build temporary frame
    temp_df = data[[col, target]].copy()

    # Define missing/unknown mask
    if unknown_val is not None:
        is_unknown = (temp_df[col] == unknown_val) | (temp_df[col].isna())
    else:
        is_unknown = temp_df[col].isna()

    # new column
    temp_df['Status'] = np.where(is_unknown, 'Missing/Unknown', 'Known')

    # Plot
    plt.figure(figsize=(10, 6))

    # assign x to hue and disable the redundant legend
    sns.violinplot(
        data=temp_df, 
        x='Status', 
        y=target, 
        hue='Status',      # Assign hue to the same variable as x
        legend=False,      # Hide the legend since the x-axis already labels the groups
        inner="quart",     # Show quartiles inside the violin
        palette="muted",   # Use a muted color palette for better aesthetics
        cut=0              # Prevents the violin from showing 'impossible' negative survival
    )

    sns.stripplot(
        data=temp_df,
        x='Status',
        y=target,
        color="black",
        alpha=0.3,
        size=4
    )

    plt.title(f"Impact of Missing '{col}' on {target}")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_item_wordcloud(data_series, title="Item Frequencies", bg_color="white"):
    """
    Generates and displays a word cloud from a frequency dictionary.
    
    This is ideal for visualizing the 'top hits' in categorical data like 
    medications or crime types, where the size of the word represents its 
    relative frequency.

    Parameters:
    -----------
    data_series : pd.Series
        A pandas Series containing the categorical data to analyze. Each entry can be a string of items (e.g., "MedA, MedB") or NaN.
    title : str, default "Item Frequencies"
        The title to display at the top of the plot.
    bg_color : str, default "white"
        The background color of the word cloud image.

    Returns:
    --------
    None
        Displays the plot using matplotlib.
    """

    # Clean + split + explode
    all_items = (
        data_series
        .dropna()
        .astype(str)
        .str.split(",")      # split on commas
        .explode()           # one item per row
        .str.strip()         # remove whitespace
    )

    # Remove empty strings (important!)
    all_items = all_items[all_items != ""]

    # Count frequencies
    frequencies = Counter(all_items)
    
    # Initialize the WordCloud object
    # 1400x900 provides high resolution for reports
    wc = WordCloud(
        width=1400, 
        height=900, 
        background_color=bg_color,
        colormap='viridis', # Aesthetic color scheme
        max_words=100
    ).generate_from_frequencies(frequencies)

    # Visualization Setup
    plt.figure(figsize=(16, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.title(f"{title}\n", fontsize=24, fontweight='bold')
    plt.axis("off") # Hide axes for a clean look
    plt.tight_layout(pad=0)
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



def plot_survival_trend(data, feature_col, survival_col, window_size=20):
    # Prepare and sort data to ensure the rolling window moves correctly
    plot_data = data[[feature_col, survival_col]].dropna().sort_values(feature_col).copy()
    
    # Calculate the rolling mean (the 'trend line')
    # min_periods=1 ensures we get values at the edges of the data
    plot_data['smoothed_survival'] = (
        plot_data[survival_col]
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
    )

    # Create the visualization
    plt.figure(figsize=(12, 6))
    
    # Plot raw data points with low alpha to see density
    sns.scatterplot(
        data=plot_data, x=feature_col, y=survival_col, 
        alpha=0.3, color='gray', label='Individual Patients'
    )
    
    # Plot the rolling mean trend line
    plt.plot(
        plot_data[feature_col], plot_data['smoothed_survival'], 
        color='red', linewidth=3, label=f'Rolling Mean (Window={window_size})'
    )

    plt.title(f"Survival Trend vs. {feature_col}")
    plt.xlabel(f"{feature_col} (Clinical Value)")
    plt.ylabel("Survival Duration (Days)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



def plot_transform_distribution(data, txt='', bins=30, fig_size=(20, 6)):
    """
    Plots histograms of original and transformed data to find the best 
    distribution for modeling or imputation.
    """
    # 1. Transformations
    # np.log1p(x) is mathematically equal to ln(1 + x) but more accurate for small x
    log_data = np.log1p(data) 
    sqrt_data = np.sqrt(data)
    square_data = np.square(data)
    
    # Using a clip or a safety check for exp to prevent overflow errors
    # P/F ratios are often >100, which would make exp(x) approach infinity
    exp_data = np.exp(np.clip(data, None, 700)) 

    # 2. Creating subplots (1 row, 5 columns)
    fig, axes = plt.subplots(1, 5, figsize=fig_size, sharey=False)
    
    # Configuration for plotting
    configs = [
        (data, 'Original Data', 'blue', f"{txt}"),
        (log_data, 'Log1p Transformed', 'green', f"log1p({txt})"),
        (sqrt_data, 'Sqrt Transformed', 'orange', f"sqrt({txt})"),
        (square_data, 'Square Transformed', 'cyan', f"square({txt})"),
        (exp_data, 'Exp Transformed', 'red', f"exp({txt})")
    ]
    
    # 3. Iterative plotting
    for i, (d, title, color, xlabel) in enumerate(configs):
        axes[i].hist(d, bins=bins, color=color, alpha=0.7, edgecolor='black')
        axes[i].set_title(title, fontsize=12)
        axes[i].set_xlabel(xlabel)
        if i == 0:
            axes[i].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()