import pandas as pd
import pickle as pkl
import pyarrow.feather as feather
import numpy as np
from tqdm import tqdm
from functions import *

####################
#    All Stocks    #
####################
with open('chars_q_raw.feather', 'rb') as f:
    chars_q = feather.read_feather(f)

chars_q = chars_q.dropna(subset=['permno'])
chars_q[['permno', 'gvkey']] = chars_q[['permno', 'gvkey']].astype(int)
chars_q['jdate'] = pd.to_datetime(chars_q['jdate'])
chars_q = chars_q.drop_duplicates(['permno', 'jdate'])

with open('chars_a_raw.feather', 'rb') as f:
    chars_a = feather.read_feather(f)

chars_a = chars_a.dropna(subset=['permno'])
chars_a[['permno', 'gvkey']] = chars_a[['permno', 'gvkey']].astype(int)
chars_a['jdate'] = pd.to_datetime(chars_a['jdate'])
chars_a = chars_a.drop_duplicates(['permno', 'jdate'])

# information list
obs_var_list = ['gvkey', 'permno', 'jdate', 'sic', 'ret', 'retx', 'retadj', 'exchcd', 'shrcd']
# characteristics with quarterly and annual frequency at the same time
accounting_var_list = ['datadate', 'acc', 'bm', 'agr', 'alm', 'ato',  'cash', 'cashdebt', 'cfp', 'chcsho', 'chpm',
                       'chtx', 'depr', 'ep', 'gma', 'grltnoa', 'lev', 'lgr', 'ni', 'noa', 'op', 'pctacc', 'pm',
                       'rd_sale', 'rdm', 'rna', 'roa', 'roe', 'rsup', 'sgr', 'sp']
a_var_list = ['a_'+i for i in accounting_var_list]
q_var_list = ['q_'+i for i in accounting_var_list]
# annual frequency only list
a_only_list = ['adm', 'bm_ia', 'herf', 'hire', 'me_ia']
# quarterly frequency only list
q_only_list = ['abr', 'sue', 'cinvest', 'nincr', 'pscore',
               # 'turn', 'dolvol'
               ]
# monthly frequency only list
m_var_list = ['baspread', 'beta', 'ill', 'maxret', 'mom12m', 'mom1m', 'mom36m', 'mom60m', 'mom6m', 're', 'rvar_capm',
              'rvar_ff3', 'rvar_mean', 'seas1a', 'std_dolvol', 'std_turn', 'zerotrade', 'me', 'dy',
              'turn', 'dolvol' # need to rerun the accounting to put them in to char_a
              ]

df_a = chars_a[obs_var_list + accounting_var_list + a_only_list + m_var_list]
df_a.columns = obs_var_list + a_var_list + a_only_list + m_var_list
df_a = df_a.sort_values(obs_var_list)

df_q = chars_q[obs_var_list + accounting_var_list + q_only_list]
df_q.columns = obs_var_list + q_var_list + q_only_list
# drop the same information columns for merging
df_q = df_q.drop(['sic', 'ret', 'retx', 'retadj', 'exchcd', 'shrcd'], axis=1)

df = df_a.merge(df_q, how='left', on=['gvkey', 'jdate', 'permno'])

# first element in accounting_var_list is datadate
for i in tqdm(accounting_var_list[1:]):
    print('processing %s' % i)
    a = 'a_'+i
    q = 'q_'+i
    t1 = 'tmp1_'+i
    t2 = 'tmp2_'+i
    t3 = 'tmp3_'+i
    t4 = 'tmp4_'+i
    t5 = 'tmp5_'+i
    
    # tmp1: if the annual variable is available
    df[t1] = np.where(df[a].isna(), False, True)
    # tmp2: if the quarterly variable is available
    df[t2] = np.where(df[q].isna(), False, True)
    # tmp3: both
    df[t3] = df[t1] & df[t2]
    # tmp4: latest one
    df[t4] = np.where(df['q_datadate'] < df['a_datadate'], df[a], df[q])
    # available one
    df[t5] = np.where(df[t1], df[a], df[q])
    # final
    df[i] = np.where(df[t3], df[t4], df[t5])
    df = df.drop([a, q, t1, t2, t3, t4, t5], axis=1)

# drop the datadate of different frequency
df = df.drop(['a_datadate', 'q_datadate'], axis=1)

# drop optional variables, you can adjust it by your selection
df = df.drop(['ret', 'retx'], axis=1)
df = df.rename(columns={'retadj': 'ret'})  # retadj is return adjusted by dividend
df['ret'] = df.groupby(['permno'])['ret'].shift(-1)  # we shift return in t period to t+1 for prediction
df['date'] = df.groupby(['permno'])['jdate'].shift(-1)  # date is return date, jdate is predictor date
df = df.drop(['jdate'], axis=1)  # now we only keep the date of return
df = df.dropna(subset=['ret']).reset_index(drop=True)
df.replace([-np.inf, np.inf], np.nan, inplace=True)

# save raw data
with open('chars60_raw_no_impute.feather', 'wb') as f:
    feather.write_feather(df, f)

# impute missing values, you can choose different func form functions.py, such as ffi49/ffi10
df_impute = df.copy()
df_impute['sic'] = df_impute['sic'].astype(int)
df_impute['date'] = pd.to_datetime(df_impute['date'])

df_impute['ffi49'] = ffi49(df_impute)
df_impute['ffi49'] = df_impute['ffi49'].fillna(49)  # we treat na in ffi49 as 'other'
df_impute['ffi49'] = df_impute['ffi49'].astype(int)

# there are two ways to impute: industrial median or mean
df_impute = fillna_ind(df_impute, method='median', ffi=49)

df_impute = fillna_all(df_impute, method='median')
df_impute['re'] = df_impute['re'].fillna(0)  # re use IBES database, there are lots of missing data

df_impute['year'] = df_impute['date'].dt.year
df_impute = df_impute[df_impute['year'] >= 1972]
df_impute = df_impute.drop(['year'], axis=1)

with open('chars60_raw_imputed.feather', 'wb') as f:
    feather.write_feather(df_impute, f)

# standardize raw data
df_rank = df.copy()
df_rank['lag_me'] = df_rank['me']
df_rank['bm'] = np.where(df_rank['bm']<0,np.nan,df_rank['bm']) # if bm<0 then bm=nan and rank_bm=0
df_rank = standardize(df_rank)
df_rank['year'] = df_rank['date'].dt.year
df_rank = df_rank[df_rank['year'] >= 1972]
df_rank = df_rank.drop(['year'], axis=1)
df_rank['log_me'] = np.log(df_rank['lag_me'])

with open('chars60_rank_no_impute.feather', 'wb') as f:
    feather.write_feather(df_rank, f)

# standardize imputed data
df_rank = df_impute.copy()
df_rank['lag_me'] = df_rank['me']
df_rank = standardize(df_rank)
df_rank['year'] = df_rank['date'].dt.year
df_rank = df_rank[df_rank['year'] >= 1972]
df_rank = df_rank.drop(['year'], axis=1)
df_rank['log_me'] = np.log(df_rank['lag_me'])

with open('chars60_rank_imputed.feather', 'wb') as f:
    feather.write_feather(df_rank, f)


# ####################
# #      SP1500      #
# ####################
# with open('/home/jianxinma/chars/data/sp1500_impute_benchmark.feather', 'rb') as f:
#     sp1500_index = feather.read_feather(f)

# sp1500_index = sp1500_index[['gvkey', 'date']]

# sp1500_impute = pd.merge(sp1500_index, df_impute, how='left', on=['gvkey', 'date'])

# # for test
# # test = sp1500_rank.groupby(['jdate'])['gvkey'].nunique()

# with open('sp1500_impute_60.feather', 'wb') as f:
#     feather.write_feather(sp1500_impute, f)

# # standardize characteristics
# sp1500_rank = pd.merge(sp1500_index, df_rank, how='left', on=['gvkey', 'date'])

# with open('sp1500_rank_60.feather', 'wb') as f:
#     feather.write_feather(sp1500_rank, f)
