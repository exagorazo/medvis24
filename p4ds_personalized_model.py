import pandas as pd
import pickle
import modules.personal_data as per
from pycaret.regression import setup, create_model, tune_model, finalize_model, predict_model

def make_model(userid):
    # training dataset
    old_df = pd.read_csv("p4ds_data.csv")

    old_df1 = old_df.drop(['A01_GLU120_TR', 'GLU_1h', 'GLU_2h'], axis=1)
    old_df2 = old_df.drop(['A01_GLU60_TR', 'GLU_1h', 'GLU_2h'], axis=1)

    # user's personal dataset
    new_df = per.personal_measurement_data()
    
    # generate the new 1h glucose column
    new_df['A01_GLU60_TR'] = (new_df['after1'] - new_df['before'])/new_df['sugar'] * 50
    # generate the new 2h glucose column
    new_df['A01_GLU120_TR'] = (new_df['after2'] - new_df['before'])/new_df['sugar'] * 50

    new_df1 = new_df.drop(['A01_GLU120_TR', 'sugar', 'before', 'after1', 'after2'], axis=1)
    new_df1 = new_df.drop(['A01_GLU120_TR', 'sugar', 'before', 'after1', 'after2'], axis=1)
    
    df1 = pd.concat([old_df1, new_df1])
    df2 = pd.concat([old_df2, new_df2])

    filename1 = f'{userid}_1h.pkl'
    filename2 = f'{userid}_2h.pkl'

    # model1 training
    regression_setup1 = setup(data=df1, target='target', silent=True, session_id=42)
    gbr1 = create_model('gbr')
    tuned_gb_model1 = tune_model(gbr1)
    final_gb_model1 = finalize_model(tuned_gb_model1)

    # save model as pkl file
    with open(filename1, 'wb') as file:
        pickle.dump(gbr1.file)

    # model2 training
    regression_setup2 = setup(data=df2, target='target', silent=True, session_id=42)
    gbr2 = create_model('gbr')
    tuned_gb_model2 = tune_model(gbr2)
    final_gb_model2 = finalize_mode2(tuned_gb_model2)

    # save model as pkl file
    with open(filename2, 'wb') as file:
        pickle.dump(gbr2.file)