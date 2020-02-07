import pandas as pd
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import numpy as np
import warnings
import itertools

def clean(data):
    data = data.replace('(\',)|(\",)', '|',regex=True)
    data = data.replace('[\[\]\"\']', '',regex=True)
    data = clean_columns(data)
    return(data)

def clean_columns(data):
    data.columns = [c.split("-")[1].strip().replace(" ", "_") for c in data.columns]
    return (data)

def firstNormal(dFrame, col_indeces = []):
    '''
    Function that takes in a dataframe and returns a the same information as a dataframe in
    first normal form (each cell contains only one piece of data). Parameter col_indeces is
    a list of column indeces to include when making pairs.
    If not included, all possible combinations are too large.
    '''
    dFrame = clean(dFrame) #removes the special characters
    ind = dFrame.index

    if col_indeces == []:
        cols = dFrame.columns
    else:
        cols = dFrame.columns[col_indeces]

    returnDF = pd.DataFrame(columns = cols)

    for i in ind:
        row_lst = []

        for c in cols:

            cell_list = [x.strip() for x in dFrame[c].loc[i].split("|")]
            row_lst.append(cell_list)

        NF1_rows = pd.DataFrame.from_records(list(itertools.product(*row_lst)), columns=cols)

        returnDF = returnDF.append(NF1_rows)

    return returnDF.reset_index(drop = True)

def oneHot(dFrame, col):
    '''
    Fumnction that takes in a dataframe and returns the information where all categories
    are one-hot encoded.
    '''
    print('{0} col has {1} unique vals'.format(col, dFrame[col].nunique()))
    onehot = data[col].str.get_dummies("|")
    newframe = pd.concat([dFrame, onehot],axis=1)
    return(newframe)

def aggregateFromTo(dFrame, from_col = 'From', to_col = 'To'):
    '''
    Function that takes in a dataframe of from -> to preferences and returns aggregated counts
    of all from -> to pairs.
    '''

    # First group by and use size for aggregations
    from_to_agg = dFrame.groupby([from_col, to_col]).size().reset_index()
    from_to_agg.columns = ['From', 'To', 'Count']

    # Remove any rows where From and To are the same
    from_to_agg = from_to_agg[from_to_agg['From'] != from_to_agg['To']]

    return from_to_agg

def sankeyFormat(dFrame, col_indeces):
    '''
    Function that takes in a data frame in the format that the original survey data comes in, as well as two
    column indeces and returns a dataframe with data aggregated into format for Sankey diagrams.
    '''
    
    if len(col_indeces) != 2:
        print("Column indeces should only have 2 items")
        return 

    dFrame = firstNormal(dFrame = dFrame, col_indeces = col_indeces)
    agg_dFrame = aggregateFromTo(dFrame, from_col=dFrame.columns[0],
                               to_col=dFrame.columns[1]).sort_values(by = 'Count', ascending = False)

    # Giving unique indeces to each origin and destination
    agg_dFrame['from_id'] = pd.factorize(agg_dFrame.iloc[:,0])[0]
    agg_dFrame['to_id'] = pd.factorize(agg_dFrame.iloc[:,1])[0]
    agg_dFrame['to_id'] = agg_dFrame['to_id'].apply(lambda x: x + 1 + max(agg_dFrame['from_id']))
    agg_dFrame.head()

    # Because the Sankey package takes data in a weird format for labeling, have to do a few more transforms
    agg_dFrame = agg_dFrame.sort_values(by = ["from_id", "to_id"])
    agg_dFrame
    labels = np.append(agg_dFrame.iloc[:,0].unique(), agg_dFrame.iloc[:,1].unique())

    # Attaching labels to the data in Sankey format
    n_blank = agg_dFrame.shape[0] - len(labels)
    labels = np.append(labels, [""] * n_blank)
    agg_dFrame['Label'] = labels
    
    return agg_dFrame

def drawSankey(dFrame, title = ""):
    '''
    Wrapper function that takes data (must be submitted in the format given by sankeyFormat helper function) and 
    draws the Sankey diagram. Use iplot(fig, validate=False) to actually see the plot.
    '''
    # Creating a data frame just with nodes for our Sankey plot
    # Need ID & Label
    from_nodes = dFrame[['from_id', 'From']].drop_duplicates()
    from_nodes.columns = ['Node', "Label"]

    to_nodes = dFrame[['to_id', 'To']].drop_duplicates()
    to_nodes.columns = ['Node', 'Label']
    nodes_df = from_nodes.append(to_nodes).reset_index(drop = True).sort_values(by = "Node")

    # # Creating a data frame with just links for Sankey plot
    links_df = dFrame[['from_id', 'to_id', 'Count']]
    links_df.columns = ['Source', 'Target', 'Value']
    links_df = links_df.sort_values(by = ['Source', 'Target'])
    
    # # Want to color all links

    colors = ['rgba(255, 250, 200, 1)',
     'rgba(0, 130, 200, 0.3)',
     'rgba(170, 255, 195, 0.3)',
     'rgba(60, 180, 75, 0.3)',
     'rgba(245, 130, 48, 0.3)',
     'rgba(230, 190, 255, 0.3)',
     'rgba(0, 128, 128, 0.3)',
     'rgba(255, 215, 180, 0.3)',
     'rgba(0, 0, 0, 0.3)',
     'rgba(230, 25, 75, 0.3)',
     'rgba(70, 240, 240, 0.3)',
     'rgba(128, 128, 0, 0.3)',
     'rgba(0, 0, 128, 0.3)',
     'rgba(145, 30, 180, 0.3)',
     'rgba(210, 245, 60, 0.3)',
     'rgba(255, 225, 25, 0.3)',
     'rgba(128, 128, 128, 0.3)',
     'rgba(170, 110, 40, 0.3)',
     'rgba(250, 190, 190, 0.3)',
     'rgba(128, 0, 0, 0.3)',
     'rgba(240, 50, 230, 0.3)',
     'rgba(255, 255, 255, 0.3)']

    colors_trim = colors[:from_nodes.shape[0]]
    colors_df = from_nodes.copy()
    colors_df['Color'] = colors_trim
    links_df = pd.merge(links_df, colors_df, left_on='Source', right_on='Node')

    # Drawing Sankey
    data_trace = dict(
        type='sankey',
        orientation = "h",
        valueformat = ".0f",

        # Creating node structure
        node = dict(
          pad = 10,
          thickness = 30,
          line = dict(
            color = "black",
            width = 0
          ),
          label =  nodes_df['Label'].dropna(axis=0, how='any'),

        ),

        # Creating link structure
        link = dict(
          source = links_df['Source'].dropna(axis=0, how='any'),
          target = links_df['Target'].dropna(axis=0, how='any'),
          value = links_df['Value'].dropna(axis=0, how='any'),
          color = links_df['Color'].dropna(axis=0, how='any')
      )
    )

    layout =  dict(
        title = title,
        height = 850,
        width = 1000,
        font = dict(
          size = 15
        ),    
    )

    fig = dict(data=[data_trace], layout=layout)

    return fig