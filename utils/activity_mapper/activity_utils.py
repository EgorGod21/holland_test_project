import pandas as pd

def load_topics(filepath):
    tems = pd.read_excel(filepath, sheet_name=None)
    combined_tems = pd.concat(tems.values(), ignore_index=True)
    combined_tems.columns = ['activity', 'enlarged_activity']
    return combined_tems

def enlarged_activity(activity, combined_tems):
    if not activity or not isinstance(activity, str):
        return ''

    tmp_df = combined_tems[combined_tems['activity'].str.lower().str.contains(activity.lower())]

    if tmp_df.empty:
        return activity

    index = tmp_df.index[0]
    if tmp_df.loc[index, 'enlarged_activity'] == 0:
        while combined_tems.loc[index, 'enlarged_activity'] != 1:
            index -= 1
        return combined_tems.loc[index, 'activity']
    else:
        return activity
