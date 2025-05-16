import ast
from tqdm import tqdm
from utils.activity_mapper.activity_utils import enlarged_activity

def consolidation_group_data(df, combined_tems):
    for i in tqdm(range(len(df))):
        groups = ast.literal_eval(df.at[i, 'items'])
        res = []

        for group in groups:
            gr = {
                'id': group.get('id'),
                'name': group.get('name'),
                'activity': group.get('activity'),
                'enlarged_activity': enlarged_activity(group.get('activity'), combined_tems),
                'description': group.get('description'),
                'status': group.get('status')
            }
            res.append(gr)

        df.at[i, 'items'] = res

    return df
