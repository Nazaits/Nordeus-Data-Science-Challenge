import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, TargetEncoder
from sklearn.metrics import mean_absolute_error as mae

dynamic_payment_segment_encoder = OneHotEncoder(sparse_output=False, feature_name_combiner=(lambda x, y: y))
registration_platform_specific_encoder = OneHotEncoder(sparse_output=False, feature_name_combiner=(lambda x, y: y))
registration_country_encoder = TargetEncoder(target_type='continuous')
global_competition_level_imputation = 0

fusion_dynamic_payment_segment_encoder = []
fusion_registration_platform_specific_encoder = []
fusion_global_competition_level_imputation = []
for i in range(14):
    fusion_dynamic_payment_segment_encoder.append(OneHotEncoder(sparse_output=False, feature_name_combiner=(lambda x, y: y)))
    fusion_registration_platform_specific_encoder.append(OneHotEncoder(sparse_output=False, feature_name_combiner=(lambda x, y: y)))
    fusion_global_competition_level_imputation.append(0)

def preprocess_fusion(data):
    data.reset_index(drop=True, inplace=True)

    for i in range(14):
        fusion_dynamic_payment_segment_encoder[i].fit(np.reshape(data[f'dynamic_payment_segment_{i+1}'].values, (-1, 1)))
        encoded = fusion_dynamic_payment_segment_encoder[i].transform(np.reshape(data[f'dynamic_payment_segment_{i+1}'].values, (-1, 1)))
        encoded = pd.DataFrame(encoded, columns=np.apply_along_axis((lambda x: x+'_'+str(i+1)), 0, fusion_dynamic_payment_segment_encoder[i].get_feature_names_out()))
        data = pd.concat([data, encoded], axis=1)

        encoded = fusion_registration_platform_specific_encoder[i].fit_transform(np.reshape(data[f'registration_platform_specific_{i+1}'].values, (-1, 1)))
        encoded = pd.DataFrame(encoded, columns=np.apply_along_axis((lambda x: x+'_'+str(i+1)), 0, fusion_registration_platform_specific_encoder[i].get_feature_names_out()))
        data = pd.concat([data, encoded], axis=1)

        fusion_global_competition_level_imputation[i] = data[f'global_competition_level_{i+1}'].mode()[0]
        data[f'global_competition_level_{i+1}'] = data[f'global_competition_level_{i+1}'].fillna(fusion_global_competition_level_imputation[i])

        data = data.drop([f'dynamic_payment_segment_{i+1}', f'registration_platform_specific_{i+1}'], axis=1)

    return data


def process_fusion(data):
    data.reset_index(drop=True, inplace=True)

    for i in range(14):
        encoded = fusion_dynamic_payment_segment_encoder[i].transform(np.reshape(data[f'dynamic_payment_segment_{i+1}'].values, (-1, 1)))
        encoded = pd.DataFrame(encoded, columns=np.apply_along_axis((lambda x: x+'_'+str(i+1)), 0, fusion_dynamic_payment_segment_encoder[i].get_feature_names_out()))
        data = pd.concat([data, encoded], axis=1)

        encoded = fusion_registration_platform_specific_encoder[i].transform(np.reshape(data[f'registration_platform_specific_{i+1}'].values, (-1, 1)))
        encoded = pd.DataFrame(encoded, columns=np.apply_along_axis((lambda x: x+'_'+str(i+1)), 0, fusion_registration_platform_specific_encoder[i].get_feature_names_out()))
        data = pd.concat([data, encoded], axis=1)

        data[f'global_competition_level_{i+1}'] = data[f'global_competition_level_{i+1}'].fillna(fusion_global_competition_level_imputation[i])

        data = data.drop([f'dynamic_payment_segment_{i+1}', f'registration_platform_specific_{i+1}'], axis=1)

    return data.drop_duplicates(['league_id'])


def preprocess(data, y):
    data.reset_index(drop=True, inplace=True)

    # dynamic_payment_segment onehot encoding
    dynamic_payment_segment_encoder.fit(np.reshape(data['dynamic_payment_segment'].values, (-1, 1)))
    encoded = dynamic_payment_segment_encoder.transform(np.reshape(data['dynamic_payment_segment'].values, (-1, 1)))
    encoded = pd.DataFrame(encoded, columns=dynamic_payment_segment_encoder.get_feature_names_out())
    data = pd.concat([data, encoded], axis=1)

    # registration_platform_specific onehot encoding
    encoded = registration_platform_specific_encoder.fit_transform(np.reshape(data['registration_platform_specific'].values, (-1, 1)))
    encoded = pd.DataFrame(encoded, columns=registration_platform_specific_encoder.get_feature_names_out())
    data = pd.concat([data, encoded], axis=1)

    # registration_country target encoding
    registration_country_encoder.fit(np.reshape(data['registration_country'].values, (-1, 1)), y.values)
    data['registration_country_encoded'] = registration_country_encoder.transform(np.reshape(data['registration_country'].values, (-1, 1)))
    
    # global_competition_level imputation
    global_competition_level_imputation = data['global_competition_level'].mode()[0]
    data['global_competition_level'] = data['global_competition_level'].fillna(global_competition_level_imputation)

    # Rename some columns
    data = data.rename(columns={'2) Minnow': 'Minnow', '4) Whale': 'Whale',  '0) NonPayer': 'NonPayer',  '1) ExPayer': 'ExPayer', '3) Dolphin': 'Dolphin'})

    # Drop encoded features and maybe other useless ones
    data = data.drop(['dynamic_payment_segment', 'registration_platform_specific', 'registration_country'], axis=1)

    return data

def process(data):
    if type(data) is str:
        data = pd.read_csv(data)
    data.reset_index(drop=True, inplace=True)
    
    # dynamic_payment_segment onehot encoding
    encoded = dynamic_payment_segment_encoder.transform(np.reshape(data['dynamic_payment_segment'].values, (-1, 1)))
    encoded = pd.DataFrame(encoded, columns=dynamic_payment_segment_encoder.get_feature_names_out())
    data = pd.concat([data, encoded], axis=1)

    # registration_platform_specific onehot encoding
    encoded = registration_platform_specific_encoder.transform(np.reshape(data['registration_platform_specific'].values, (-1, 1)))
    encoded = pd.DataFrame(encoded, columns=registration_platform_specific_encoder.get_feature_names_out())
    data = pd.concat([data, encoded], axis=1)

    # registration_country target encoding
    data['registration_country_encoded'] = registration_country_encoder.transform(np.reshape(data['registration_country'].values, (-1, 1)))
    
    # global_competition_level encoding
    data['global_competition_level'] = data['global_competition_level'].fillna(global_competition_level_imputation)

    # Rename some columns
    data = data.rename(columns={'2) Minnow': 'Minnow', '4) Whale': 'Whale',  '0) NonPayer': 'NonPayer',  '1) ExPayer': 'ExPayer', '3) Dolphin': 'Dolphin'})

    # Drop encoded features and maybe other useless ones
    data = data.drop(['dynamic_payment_segment', 'registration_platform_specific', 'registration_country'], axis=1)

    return data

def league_split(data, split_size=0.2, random_state=42):
    np.random.seed(random_state)
    test_leagues = np.random.choice(data['league_id'].unique(), int(data['league_id'].unique().shape[0]*split_size))

    data_train = data[~data['league_id'].isin(test_leagues)]
    data_test = data[data['league_id'].isin(test_leagues)]
    
    return data_train, data_test

def league_train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    test_leagues = np.random.choice(X['league_id'].unique(), int(X['league_id'].unique().shape[0]*test_size))

    X_train = X[~X['league_id'].isin(test_leagues)]
    y_train = y[~X['league_id'].isin(test_leagues)]
    X_test = X[X['league_id'].isin(test_leagues)]
    y_test = y[X['league_id'].isin(test_leagues)]
    
    return X_train, X_test, y_train, y_test

def league_fusion(data, kaioken=1):
    concat_data = []
    for league_id in data['league_id'].unique():
        for times in range(kaioken):
            group = data[data['league_id'] == league_id].sample(frac=1)
            concat_dict = {'league_id': league_id}
            league_rank_dict = {}
            for i, (_, row) in enumerate(group.iterrows()):
                for col in group.columns:
                    if col != 'league_id' and col != 'league_rank':
                        concat_dict[f'{col}_{i+1}'] = row[col]
                    elif col == 'league_rank':
                        league_rank_dict[f'{col}_{i+1}'] = row[col]
            concat_data.append(concat_dict|league_rank_dict)

    return pd.DataFrame(concat_data).sample(frac=1)

def league_fusion_cross_val(model, X, Y, cv=4):
    scores = []
    leagues = X['league_id'].unique()
    group_size = leagues.shape[0]//cv
    for group in range(cv):
        train_ind = ~X['league_id'].isin(leagues[group*group_size:group*group_size+group_size])
        test_ind = X['league_id'].isin(leagues[group*group_size:group*group_size+group_size])
        X_train = preprocess_fusion(X[train_ind]).drop('league_id', axis=1)
        Y_train = Y[train_ind]
        X_test = process_fusion(pd.concat([X[test_ind], Y[test_ind]], axis=1))
        Y_test = X_test[Y.columns]
        X_test = X_test.drop(Y.columns, axis=1)
        X_test = X_test.drop(['league_id'], axis=1)

        model.fit(X_train, Y_train)
        scores.append(mae(Y_test, model.predict(X_test)))

    return scores
