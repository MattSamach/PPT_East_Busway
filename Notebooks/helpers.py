import pandas as pd
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import numpy as np
import warnings
import itertools
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import re
from fuzzywuzzy import process
from fuzzywuzzy import fuzz

def clean(data):
    data = data.replace('(\',)|(\",)', '|',regex=True)
    data = data.replace('[\[\]\"\']', '',regex=True)
    data = clean_columns(data)
    return(data)

def clean_columns(data):
    data.columns = [c.split("-")[1].strip().replace(" ", "_") if (len(c.split("-")) > 1)
                    else c.strip().replace(" ", "_") for c in data.columns]
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
    onehot = dFrame[col].str.get_dummies("|")
    newframe = pd.concat([dFrame, onehot], axis=1)
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

    # Ranking source ranks
    source_ranks = links_df.groupby('Source').rank(ascending = False)
    links_df['Rank'] = source_ranks.Value

    # # Want to color all links

    colors = ['rgba(255, 250, 200,',
     'rgba(0, 130, 200,',
     'rgba(170, 255, 195,',
     'rgba(60, 180, 75,',
     'rgba(245, 130, 48,',
     'rgba(230, 190, 255,',
     'rgba(0, 128, 128,',
     'rgba(255, 215, 180,',
     'rgba(0, 0, 0,',
     'rgba(230, 25, 75,',
     'rgba(70, 240, 240,',
     'rgba(128, 128, 0,',
     'rgba(0, 0, 128,',
     'rgba(145, 30, 180,',
     'rgba(210, 245, 60,',
     'rgba(255, 225, 25,',
     'rgba(128, 128, 128,',
     'rgba(170, 110, 40,',
     'rgba(250, 190, 190,',
     'rgba(128, 0, 0,',
     'rgba(240, 50, 230,',
     'rgba(255, 255, 255,']

    # Formating the  colors to only have the lenght of nodes (Plotly Sankey formatting)
    colors_trim = colors[:from_nodes.shape[0]]
    colors_df = from_nodes.copy()
    colors_df['Color'] = colors_trim

    # Merging colors to links df
    links_df = pd.merge(links_df, colors_df, left_on='Source', right_on='Node')

    # Making it so that
    links_df['Color'] = [links_df.Color[i] + '0.9)' if (links_df.Rank[i]==1.0 or links_df.Rank[i]==2.0) else
                     links_df.Color[i] + '0.3)'for i in range(links_df.shape[0])]

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
          size = 13
        ),
    )

    fig = dict(data=[data_trace], layout=layout)

    return fig

def find_unique_loc(dFrame, col):
    #locDict = {}
    locs = []
    for i in range(dFrame.shape[0]):
        cell_list = [x.strip() for x in dFrame[col].loc[i].split("|")]
        locs = locs + cell_list
    return(set(locs))

def countyMatcher(placelist, gisTown, gisType, gisAll):
    '''
    Takes in a location column and returns a matcher to a GIS asset
    Matches on the strict name, then finds municipality type
    '''
    # List for dicts for easy dataframe creation
    dict_list = []

    unusual_match = {'2434 south braddock ave': ('SWISSVALE', 'Swissvale Borough', 'BOROUGH'),
                     'mck': ('MCKEESPORT','McKeesport', 'CITY'),
                     'mon valley': ('','',''),
                     'Wexford': ('PINE', 'Pine Township', 'TOWNSHIP'),
                     'Bethel Park Borough': ('BETHEL PARK', 'Bethel Park Municipality', 'MUNICIPALI'),
                     'Pittsburgh': ('PITTSBURGH', 'Pittsburgh', 'CITY')}
    # extracting municipality type
    p = re.compile("(Township|Borough|Municipality|City)")
    # Iterating over nonpgh places
    for place in placelist:
        # New dict for storing data
        dict_ = {}
        # Find muni type first

        # Replace pittsburgh:
        if re.search('Pittsburgh', place, re.IGNORECASE):
            mName = (unusual_match['Pittsburgh'][0], 100)
            aMatch = (unusual_match['Pittsburgh'][1], 100)
            tMatch = (unusual_match['Pittsburgh'][2], 100)
        # Take out messy matches o
        elif place in unusual_match.keys():
            mName = (unusual_match[place][0], 100)
            aMatch = (unusual_match[place][1], 100)
            tMatch = (unusual_match[place][2], 100)
        # Use our method to find best match, we can set a threshold here
        else:
            type_result = p.search(place, re.IGNORECASE)
            if type_result:
                muni_type = type_result.group(1)
                tMatch = process.extractOne(muni_type, gisType, scorer=fuzz.ratio)
            else:
                tMatch = ('', 100)
            mName = process.extractOne(place, gisTown,  scorer=fuzz.ratio)#, score_cutoff = 60)
            aMatch = process.extractOne(place, gisAll,  scorer=fuzz.ratio)

        dict_.update({"from_ppt" : place})
        dict_.update({"from_LabelGIS" : aMatch[0]})
        dict_.update({"labelScore" : aMatch[1]})
        dict_.update({"from_NameGIS" : mName[0]})
        dict_.update({"nameScore" : mName[1]})
        dict_.update({"from_TypeGIS" : tMatch[0]})
        dict_.update({"typeScore" : tMatch[1]})
        dict_list.append(dict_)

    matches_all = pd.DataFrame(dict_list)
    return(matches_all)

def add_county_cat(dFrame, col, countyDF, prefix):
    '''
    Takes a cleaned dataframe and appends geographic information
        at the county level
    '''
    loc_list = find_unique_loc(dFrame, col)
    fromCounty = countyMatcher(loc_list, countyDF.NAME, countyDF.TYPE, countyDF.LABEL)
    dataLoc = pd.merge(dFrame, fromCounty[['from_ppt','from_LabelGIS']],  how='left', left_on=[col], right_on = ['from_ppt'])
    dataLoc = pd.merge(dataLoc,
                        countyDF[['NAME', 'LABEL', 'TYPE', 'COG', 'FIPS', 'MUNICODE', 'OBJECTID']],
                        left_on=['from_LabelGIS'], right_on = ['LABEL'])
    dataLoc.drop(['from_ppt', 'NAME', 'from_LabelGIS', 'TYPE'], axis=1)
    dataLoc = dataLoc.rename(columns={'COG': '{0}COG'.format(prefix), 'FIPS': '{0}FIPS'.format(prefix),
                                      'LABEL': '{0}LABEL'.format(prefix),
                                      'MUNICODE': '{0}MUNICODE'.format(prefix),
                                      'OBJECTID': '{0}OBJECTID'.format(prefix)})
    print('{0} unique regions'.format(dataLoc['{0}COG'.format(prefix)].nunique()))
    return(dataLoc)

def category_percents(dFrame, cat_name="", val_name="", cat_index="", val_index=""):
    '''
    Takes a pandas data frame, a categorical column (given either by name or by index) and value column
    (also given by name or index.) Then calculates the percentage share of all the different values in the
    value column for each category.
    '''
    if cat_index == "":
        cat = cat_name
    else:
        cat = dFrame.columns[cat_index]

    if val_index == "":
        val = val_name
    else:
        val = dFrame.columns[val_index]

    # Get total count of each category
    cat_count = dFrame.groupby(cat, as_index = False).count().iloc[:, :2]
    cat_count.columns = [cat, "Total"]

    # Get count of each value within each category
    val_count = dFrame.groupby([cat, val], as_index = False).count().iloc[:, :3]
    val_count.columns = [cat, val, "Count"]

    # Merge two dataframes together and get within group percentages
    final_df = pd.merge(cat_count, val_count, on = cat)
    final_df['Share'] = final_df.Count/final_df.Total * 100
    return final_df

def graph_percent_bars(dFrame, cat_name, val_name, share_name):
    '''
    Takes a dataframe, a category or grouping name (cat_name), a val_name (the column which shares are split over),
    and the name of the share (or percent) column. Returns a stacked bar chart comapring shares of the values
    across the different categories.
    '''
    cats = dFrame[cat_name].unique()
    vals = dFrame[val_name].unique()

    # First fill a list of lists with all values
    list_of_lists = []

    # Looping through all value types and category types, getting the percentages
    for i in range(len(vals)):
        shares_list = []
        filtered = dFrame[dFrame[val_name] == vals[i]]

        for j in range(len(cats)):

            # Check that this value type exists within this category. If it doesn't append 0 to shares_list

            if filtered[filtered[cat_name] == cats[j]].shape[0] == 0:
                shares_list.append(0)
            else:
                shares_list.append(list(filtered[filtered[cat_name] == cats[j]][share_name])[0])

        list_of_lists.append(shares_list)

    # Getting to graphing

    ind = range(len(cats))
    width = 0.5

    tracker = [0 for x in range(len(list_of_lists[0]))]
    plt_dict = dict()

    figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')

    for i in range(len(list_of_lists)):
        plt_dict[i] = plt.bar(ind, list_of_lists[i], width, bottom=tracker)
        tracker = [a + b for a, b in zip(tracker, list_of_lists[i])]

    plt.ylabel('Percent')
    plt.xticks(ind, cats, rotation=-80)
    plt.legend(vals, bbox_to_anchor = (1.7,0.7), loc = 'upper right')
