import ast
from tqdm import tqdm
from .text_cleaning import text_preprocess_natasha

def process_group_data(df):
    for i in tqdm(range(len(df))):
        groups = ast.literal_eval(df.at[i, 'items'])
        res = []
        for group in groups:
            gr = {}
            id = group.get('id')
            name = group.get('name', '')
            activity = group.get('activity', '')
            enlarged_activity = group.get('enlarged_activity', '')
            description = group.get('description', '')
            status = group.get('status', '')

            text = ' '.join([name, description, status, activity]).lower()
            clean_text = text_preprocess_natasha(text)

            gr['id'] = id
            gr['activity'] = enlarged_activity
            gr['description'] = clean_text
            res.append(gr)

        df.at[i, 'items'] = res
    return df

def combine_descriptions(df):
    full_descriptions = []

    for i in tqdm(range(len(df))):
        groups = df.at[i, 'items']
        combined = ' '.join(group['description'] for group in groups if group.get('description'))
        full_descriptions.append(combined)

    df['description'] = full_descriptions
    return df
