from app.singleton import GlobalMLOPS

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

INPUT_COLUMNS = [
    'CreditScore',
    'Geography',
    'Gender',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary'
]


class Transformer:
    
    def __init__(self, file) -> None:
        dataset = pd.read_csv(file)
        self.independent_var = dataset.iloc[:, [dataset.columns.get_loc(i) for i in INPUT_COLUMNS]]
        self.dataset = dataset.drop(columns=['Exited', 'RowNumber'])

    def prepare_for_prediction(self):
        data = self.independent_var
        label_encoder = LabelEncoder()
        data['Gender'] = label_encoder.fit_transform(data['Gender'])
        geography_index = data.columns.get_loc('Geography')   
        column_transformer = ColumnTransformer(
            [('encoder', OneHotEncoder(), [geography_index])],
            remainder='passthrough'
        )

        inputs = np.array(column_transformer.fit_transform(data))
        return GlobalMLOPS().scaler.transform(inputs)
    
    def get_html(self, classified_data):
        df = self.dataset.copy()
        df['PROBABILITY'] = classified_data
        df['STATUS'] = df['PROBABILITY'].apply(lambda x: 'Churn' if x > 0.5 else 'Exit')
        df['PROBABILITY'] = df['PROBABILITY'].apply(lambda x: x if x > 0.5 else 1 - x)

        df_exited = df[df['STATUS'] == 'Exit']
        df_churn = df[df['STATUS'] == 'Churn']

        df_exited.sort_values(by=['PROBABILITY'], ascending=False, inplace=True)
        df_churn.sort_values(by=['PROBABILITY'], ascending=True, inplace=True)

        df = pd.concat([df_exited, df_churn])

        html_table = df.to_html(index=False, justify='left')

        # Replace the 'Status' column values with colored text
        html_table = html_table.replace('Exit', '<span style="color: red; font-weight: bold;">Exit</span>')
        html_table = html_table.replace('Churn', '<span style="color: green; font-weight: bold;">Churn</span>')

        # Add CSS styles to the table
        styled_table = f'<style>table {{border-collapse: collapse;}} th, td {{border: 1px solid black; padding: 8px;}} </style>{html_table}'
        return styled_table