import random
import numpy as np
import pandas as pd
import datetime
import math
import gym
from gym import spaces
import matplotlib.pyplot as plt

START_TIME = 34200 
END_TIME = 57600 
LOOKBACK_LENGTH = 60
DECISION_INTERVAL = 10 
ENTRY_TIME_MIN = START_TIME + 63*60
ENTRY_TIME_MAX = END_TIME - 63*60
MAX_NUM_LONG_SHORT = 5 
STOP_LOSS_PERCENT = 0.1
LOG_BASE = 20 

AMPLIFIER = 1000.0

class MarketEnv(gym.Env): 
    
    def __init__(self, data_path, investment_each=1.0, gain_scale=1.0, loss_scale=1.0, 
                 loan_yearly_rate=0.05, transaction_fee_percent=0.01):
        self.test = False
        self.data_path = data_path
        self.investment_each = investment_each   
        self.long_max_times = MAX_NUM_LONG_SHORT
        self.short_max_times = MAX_NUM_LONG_SHORT
        self.target_stock = 'SPY'
        self.df_data = self.load_data()
        self.gain_scale = gain_scale
        self.loss_scale = loss_scale 
        self.loan_daily_interest = 1.0 * loan_yearly_rate / 365.0
        self.transaction_fee_percent = transaction_fee_percent
        self.actions = [0, 1, 2] 
        self.action_space = spaces.Discrete(len(self.actions))
        self.stop_loss_percent = STOP_LOSS_PERCENT
        self.observation_space = spaces.Box(np.ones(120) * -10, 
                                            np.ones(120)*10)
        self._reset()
        self._seed()

    def load_data(self):
        df_data = pd.read_csv(self.data_path)
        max_close = df_data['last'].max()
        min_close = df_data[df_data['last'] > 0]['last'].min()
        price_scale = 1.0/((max_close - min_close)/2.0)
        df_data['ask'] = df_data['ask'].apply(lambda x: x * price_scale)
        df_data['bid'] = df_data['bid'].apply(lambda x: x * price_scale)
        df_data['last'] = df_data['last'].apply(lambda x: x * price_scale)
        df_data['log_vol'] = df_data['vol'].apply(lambda x: math.log(max(1.1,x), LOG_BASE))
        df_data['day'] = pd.to_datetime(df_data['day'], format='%Y%m%d')
        self.price_scale = price_scale
        mean_last = df_data['last'].mean()
        mean_log_vol = df_data['log_vol'].mean()
        df_data['last'] = df_data['last'].apply(lambda x: x+0.001)
        df_data['log_vol'] = df_data['log_vol'].apply(lambda x: x)
        return df_data
      
    def get_random_entry(self, test_indicator):
        #year = random.choice(range(1998, 2007))
        if test_indicator:
            year = random.choice(range(2008, 2010))
        else:
            year = random.choice(range(1998, 2007))
        month = random.choice(range(1, 13))
        start_date_str = '%04d%02d01'% (year, month)
        if month < 12:
            next_month = month + 1
            next_year = year
        else:
            next_month = 1
            next_year = year + 1
        end_date_str = '%04d%02d01'% (next_year, next_month)
        start_date = datetime.datetime.strptime(start_date_str, "%Y%m%d").date()
        end_date = datetime.datetime.strptime(end_date_str, "%Y%m%d").date()
        df_data = self.df_data
        df_selected_dates = pd.DataFrame(df_data[(df_data['day'] >=start_date) & (df_data['day']< end_date)]
            ['day'].unique())        
        day_ind = random.choice(range(len(df_selected_dates)))
        day = df_selected_dates.iloc[day_ind][0].day        
        rand_min = random.choice(range(ENTRY_TIME_MIN, ENTRY_TIME_MAX, 60))      
        selected_day_str = '%04d%02d%02d'% (year, month, day)
        selected_date = datetime.datetime.strptime(selected_day_str, "%Y%m%d").date()
        
        ind = df_data[(df_data['day']==selected_date) & (df_data['min']==rand_min)].index[0]
        return ind

    def _reset(self):  
        self.current_index = self.get_random_entry(self.test)       
        self.current_position = 0 
        self.long_times = 0
        self.short_times = 0
        self.boughts = []
        self.done = False
        self.reward = 0.
        self.long_stop_loss_price = self.df_data['last'].min()
        self.short_stop_loss_price = self.df_data['last'].max()
        self.trade_day = self.df_data.iloc[self.current_index]['day']
        self.close_price = self.df_data[(self.df_data['day']==self.trade_day) & 
                                        (self.df_data['min']==END_TIME)]['last'].values[0]
        self.state = self.get_state()
        return self.state
    
    def _calculate_gain(self, current_price):
        curent_list_gains = []
        investment_list = []
        if self.current_position == 1:          
            for bought_price, shares in self.boughts:
                close_gain = (current_price - bought_price) * np.abs(shares)
                curent_list_gains.append(close_gain)
                investment_list.append(np.abs(bought_price*np.abs(shares)))
        if self.current_position == 2:          
            for bought_price, shares in self.boughts:
                close_gain = (bought_price - current_price) * np.abs(shares)
                curent_list_gains.append(close_gain) 
                investment_list.append(np.abs(bought_price*np.abs(shares)))

        current_gain = sum(curent_list_gains) 
        if sum(investment_list) > 0:
            gain_percent = current_gain / sum(investment_list) 
        else:
            gain_percent = 0.0
        return AMPLIFIER*gain_percent
    
    def _update_avg_stop_loss(self):
        investment_list = []
        total_shares = 0
        if self.current_position == 1:           
            for bought_price, shares in self.boughts:
                investment_list.append(bought_price*np.abs(shares))
                total_shares = total_shares + np.abs(shares)
            if len(investment_list) > 0:
                if sum(investment_list) > 0:
                    self.long_stop_loss_price = (sum(investment_list) /total_shares)*(
                            1-self.stop_loss_percent)
                else:
                    self.long_stop_loss_price = (sum(investment_list) /total_shares)*(
                            1+self.stop_loss_percent)                    
            else:
                self.long_stop_loss_price = self.df_data['last'].min()
                
        if self.current_position == 2:         
            for bought_price, shares in self.boughts:
                investment_list.append(bought_price*np.abs(shares))
                total_shares = total_shares + np.abs(shares)
            if len(investment_list) > 0:
                if sum(investment_list) > 0:
                    self.short_stop_loss_price = (sum(investment_list) /total_shares)*(
                            1 + self.stop_loss_percent)
                else:
                    self.short_stop_loss_price = (sum(investment_list) /total_shares)*(
                            1 - self.stop_loss_percent)                    
            else:
                self.short_stop_loss_price = self.df_data['last'].max()  
      
    def _step(self, action):
        current_price = self.df_data.iloc[self.current_index]['last']
        if self.df_data.iloc[self.current_index]['min'] >= END_TIME:
            # terminate
            self.done = True
            self.reward = self._calculate_gain(self.close_price)
            self.boughts = []
            return None, self.reward, self.done, {}
        if (self.current_position == 1) and (current_price <= self.long_stop_loss_price):
            self.done = True
            self.reward = self._calculate_gain(self.long_stop_loss_price)
            self.boughts = []
            return None, self.reward, self.done, {}
        if (self.current_position == 2) and (current_price >= self.short_stop_loss_price):
            self.done = True
            self.reward = self._calculate_gain(self.short_stop_loss_price)
            self.boughts = []
            return None, self.reward, self.done, {}
        if action == 1:
            self.long_times +=1
            if self.current_position == 2:
                self.done = True
                self.reward = self._calculate_gain(current_price)
                self.boughts = []
                self.current_index = self.current_index + DECISION_INTERVAL       
                self.state = self.get_state()
                return self.state, self.reward, self.done, {}                
            if self.current_position == 0:
                self.boughts.append((current_price, np.abs(self.investment_each / current_price)))
                self.current_position = 1
                self._update_avg_stop_loss()
                self.reward = 0 
                self.current_index = self.current_index + DECISION_INTERVAL       
                self.state = self.get_state()
                return self.state, self.reward, self.done, {}
            if self.current_position == 1:
                if self.long_times < self.long_max_times:
                    self.boughts.append((current_price, np.abs(self.investment_each / current_price)))
                    self._update_avg_stop_loss()
                    self.reward = 0
                else: 
                    self.reward = 0
                self.current_index = self.current_index + DECISION_INTERVAL       
                self.state = self.get_state()
                return self.state, self.reward, self.done, {}

        if action == 2:
            self.short_times += 1
            if self.current_position == 1:
                self.done = True
                self.reward = self._calculate_gain(current_price)
                self.boughts = []
                self.current_index = self.current_index + DECISION_INTERVAL       
                self.state = self.get_state()
                return self.state, self.reward, self.done, {}
            if self.current_position == 0:
                self.current_position = 2
                self.boughts.append((current_price, (-1)*np.abs(self.investment_each / current_price)))
                self._update_avg_stop_loss()
                self.reward = 0
                self.current_index = self.current_index + DECISION_INTERVAL       
                self.state = self.get_state()
                return self.state, self.reward, self.done, {}
            if self.current_position == 2:
                if self.short_times < self.short_max_times:
                    self.boughts.append((current_price, (-1)*np.abs(self.investment_each / current_price)))
                    self._update_avg_stop_loss()
                    self.reward = 0
                else: 
                    self.reward = 0
                self.current_index = self.current_index + DECISION_INTERVAL       
                self.state = self.get_state()
                return self.state, self.reward, self.done, {}           
        if action == 0:
            self.reward = 0
            self.current_index = self.current_index + DECISION_INTERVAL       
            self.state = self.get_state()
            return self.state, self.reward, self.done, {}                              

    def _render(self, mode='human', close=False):
        if close:
            return
        return self.state

    def _seed(self):
        return int(np.random.randint(1,100))

    def get_state(self):
        list_prices = [s[0] for s in self.boughts]
        list_shares = [s[1] for s in self.boughts]
        avg_price_per_share = sum(list_prices)/len(list_prices) if len(list_prices)>0 else 0
        tot_shares = sum(list_shares)
        account_status = [avg_price_per_share, tot_shares]
        
        previous_prices = self.df_data['last'][self.current_index-LOOKBACK_LENGTH:self.current_index]
        previous_volumes = self.df_data['log_vol'][self.current_index-LOOKBACK_LENGTH:self.current_index]
        state_list = account_status + previous_prices.values.tolist() + previous_volumes.values.tolist()
        list_prices = previous_prices.values.tolist()
        list_vols = previous_volumes.values.tolist()
        length_array = len(list_prices)
        state = np.array((list_prices+list_vols))
        return state
