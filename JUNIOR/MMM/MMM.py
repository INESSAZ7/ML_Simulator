import pandas as pd
import numpy as np


def last_touch_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate last touch attribution"""
    df_events = events.copy()
    df_events = df_events[df_events['is_purchased']==1]
    attribution = pd.pivot_table(df_events, values='gmv', index=['week', 'user_id'],
                       columns=['channel'], aggfunc="sum")
    attribution = attribution.rename_axis(None, axis=1).reset_index().fillna(0).astype(int)
    attribution['total_gmv'] = \
                    attribution.loc[:,['bloggers', 'context_ads', 'mobile_ads', 'social_media']].sum(axis=1).astype(int)
    return attribution[~(attribution['total_gmv']==0)]


def first_touch_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate first touch attribution"""
    df_event = events.copy()
    grouped = df_event.groupby('user_id')

    #Итерируемся по юзерам и заполняем номер касания, gmv
    for _, group in grouped:
        counter = 1
        for idx in group.index:
            df_event.loc[idx, 'touch_num'] = counter
            counter+=1
            if group.loc[idx]['is_purchased'] == 1:
                counter = 1    
        for idx in group.index:
            if df_event.loc[idx, 'touch_num'] == 1:
                channel_local = df_event.loc[idx, 'channel']
            if df_event.loc[idx, 'is_purchased'] == 1:
                df_event.loc[idx, 'channel'] = channel_local     
    #Аналогично last_touch
    df_event = df_event[df_event['is_purchased']==1]
    attribution = pd.pivot_table(df_event, values='gmv', index=['week', 'user_id'],
                       columns=['channel'], aggfunc="sum")
    attribution = attribution.rename_axis(None, axis=1).reset_index().fillna(0).astype(int)
    attribution['total_gmv'] = \
                    attribution.loc[:,['bloggers', 'context_ads', 'mobile_ads', 'social_media']].sum(axis=1).astype(int)

    return attribution[~(attribution['total_gmv']==0)]


def linear_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate linear attribution"""
    def linear(group: pd.DataFrame):
        group = group.copy()
        for idx in group.index:
            for channel in group.loc[idx, 'channel']:
                group.loc[idx, channel] += group.loc[idx, 'gmv']/(len(group.loc[idx, 'channel']))
        return  group  

    df_event = events.copy()
    for channel in df_event['channel'].unique():
        df_event[channel] = 0

    grouped = df_event.groupby('user_id')

    #Итерируемся по юзерам и заполняем номер касания, gmv
    for _, group in grouped:
        counter = 1
        for idx in group.index:
            df_event.loc[idx, 'touch_num'] = counter
            counter+=1
            if group.loc[idx]['is_purchased'] == 1:
                counter = 1 
        channels_local = []
        for idx in group.index:
            channels_local.append(df_event.loc[idx, 'channel'])
            if df_event.loc[idx, 'is_purchased'] == 1:
                df_event.at[idx, 'channel'] = channels_local
                channels_local = []

    df_event = df_event[df_event['is_purchased']==1].groupby('user_id', group_keys=False).apply(lambda x: linear(x)) 
    columns = ['week', 'user_id', 'bloggers', 'social_media', 'context_ads', 'mobile_ads', 'gmv']
    df_event = df_event[columns]
    col_channels = ['bloggers', 'social_media', 'context_ads', 'mobile_ads']
    df_event[col_channels] = df_event[col_channels].apply(lambda x: round(x,2))
    df_event = df_event.rename(columns={"gmv": "total_gmv"})
    df_event['total_gmv'] = df_event['total_gmv'].astype(float)
            
    return df_event.sort_values(by=['week', 'user_id'])


def u_shaped_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate U-Shaped attribution"""
    def shaped(group: pd.DataFrame):
        group = group.copy()
        for idx in group.index:
            group_purchase = group.loc[idx, 'channel']
            if len(group_purchase)==1:
                for channel in group_purchase:
                    group.loc[idx, channel] += group.loc[idx, 'gmv']
            elif len(group_purchase)==2:
                for channel in group_purchase:
                    group.loc[idx, channel] += group.loc[idx, 'gmv']/2        
            else:
                first = group_purchase[0]
                last = group_purchase[-1]
                group.loc[idx, first] += group.loc[idx, 'gmv']*0.4 
                group.loc[idx, last] += group.loc[idx, 'gmv']*0.4
                for channel in group_purchase[1:-1]:
                    group.loc[idx, channel] += group.loc[idx, 'gmv']*0.2/(len(group_purchase)-2)
        return  group 
    
    df_event = events.copy()
    for channel in df_event['channel'].unique():
        df_event[channel] = 0

    grouped = df_event.groupby('user_id')

    #Итерируемся по юзерам и заполняем номер касания, gmv
    for _, group in grouped:
        counter = 1
        for idx in group.index:
            df_event.loc[idx, 'touch_num'] = counter
            counter+=1
            if group.loc[idx]['is_purchased'] == 1:
                counter = 1 
        channels_local = []
        for idx in group.index:
            channels_local.append(df_event.loc[idx, 'channel'])
            if df_event.loc[idx, 'is_purchased'] == 1:
                df_event.at[idx, 'channel'] = channels_local
                channels_local = []

    df_event = df_event[df_event['is_purchased']==1].groupby('user_id', group_keys=False).apply(lambda x: shaped(x)) 
    columns = ['week', 'user_id', 'bloggers', 'social_media', 'context_ads', 'mobile_ads', 'gmv']
    df_event = df_event[columns]
    col_channels = ['bloggers', 'social_media', 'context_ads', 'mobile_ads']
    df_event[col_channels] = df_event[col_channels].apply(lambda x: round(x,2))
    df_event = df_event.rename(columns={"gmv": "total_gmv"})
    df_event['total_gmv'] = df_event['total_gmv'].astype(float)
            
    return df_event.sort_values(by=['week', 'user_id'])

def roi(attribution: pd.DataFrame, ad_costs: pd.DataFrame) -> pd.DataFrame:
    """Calculate ROI"""
    roi = ad_costs.copy()
    for channel in list(roi['channel']):
        cond = roi['channel']==channel
        gmv = round(attribution[channel].sum())
        costs = roi.loc[cond, 'costs']
        roi.loc[cond, 'gmv'] = gmv
        roi.loc[cond, 'roi%'] = round((gmv - costs)/costs *100)
        
    return roi
