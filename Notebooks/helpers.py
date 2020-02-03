import pandas as pd
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import numpy as np
import warnings
import itertools

def clean(data):
    data = data.replace('(\',)|(\",)', '|',regex=True)
    data = data.replace('[\[\]\"\']', '',regex=True)
    return(data)

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
          value = links_df['Value'].dropna(axis=0, how='any')
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
