import pandas as pd

def limit_gmv(df: pd.DataFrame):
    """Принимает на вход датафрейм с предсказанием оборота,
        возвращает обработанные датафрейм"""
    df_s = df.copy()
    df_s['gmv'] = ((df_s.gmv/df_s.price).astype(int) * df_s['price']) \
                                .clip(upper = df_s.price*df_s.stock) 
    return df_s
