import pandas as pd
import numpy as np

def fill_missing_values(df_data):
    ind = 0
    while ind < len(df_data):
        if (df_data.iloc[ind]['bid']==0) or (df_data.iloc[ind]['vol']==0): 
            start_loc=ind
            rep_loc=start_loc+1
            while True:           
                if (df_data.iloc[rep_loc]['bid']>0) and (df_data.iloc[rep_loc]['vol']>0):
                    df_data.at[start_loc,'last']=df_data.at[rep_loc,'last']
                    df_data.at[start_loc,'ask']=df_data.at[rep_loc,'ask']
                    df_data.at[start_loc,'bid']=df_data.at[rep_loc,'bid']
                    df_data.at[start_loc,'vol']=df_data.at[rep_loc,'vol']
                    #print('replaced %d with %d' % (start_loc, rep_loc))
                    break
                else:
                    rep_loc = rep_loc +1

        ind=ind+1 
    return df_data

df_data = pd.read_csv('./data_SPY/SPY_1998_2010_v2.csv')
# divive into 10 files
tot_len = len(df_data)
num_files = 10.0
gap = int(np.ceil(tot_len/num_files))
list1=[]
for i in range(0, tot_len, gap):
    list1.append(i)
list2=[]
for i in range(9):
    list2.append((list1[i], list1[i+1]))
list2.append((list1[-1], tot_len))
num=0
for s in list2:
    df_temp = df_data.iloc[s[0]:s[1]]
    df_temp.to_csv('./data_SPY/df_%d.csv' % num)
    num=num+1

cols=['day', 'min', 'last', 'ask', 'bid', 'vol']
for i in range(len(list2)):
    df_a = pd.read_csv('./data_SPY/df_%d.csv' % i)
    df_a = df_a[cols]  
    df_a_f = fill_missing_values(df_a)[cols]
    df_a_f.to_csv('./data_SPY/filled_df_%d.csv' % i)

list_df=[]
for i in range(len(list2)):
    df_a = pd.read_csv('./data_SPY/filled_df_%d.csv' % i)
    list_df.append(df_a)
df_tot = pd.concat(list_df)
df_tot = df_tot[cols]
df_tot = df_tot.reset_index(drop=True)
df_tot.to_csv('./data_SPY/SPY_1998_2010_v2_filled.csv')

