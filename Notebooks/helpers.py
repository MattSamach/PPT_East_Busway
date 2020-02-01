def clean(data):
    data = raw_data.replace('(\',)|(\",)', '|',regex=True)
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
    nrows = dFrame.shape[0]

    if col_indeces == []:
        cols = dFrame.columns
    else:
        cols = dFrame.columns[col_indeces]

    returnDF = pd.DataFrame(columns = cols)


    for i in range(nrows):
        row_lst = []

        for c in cols:

            cell_list = [x for x in dFrame[c].loc[i].split("|")]
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
