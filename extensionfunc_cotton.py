#functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta

#All dataframes here should be sorted first according to date, ascending

#1. Trading-day Related and Date Processing
#Defining workday using a long enough period of index data, e.g. 40 years, which covers testing interval
def input_workday(path):
    # use S&P 500 data for US workday, for chinese workday, use CSI 300 index data
    df = pd.read_excel(path)
    df = df.sort_values(by="Date", ascending=True)
    df = df.reset_index(drop=True)
    workday = df.drop(columns='SHSN300 Index')
    return(workday)

def is_workday(workday,date):
    find=workday["Date"].loc[workday["Date"].isin([date])]
    if len(find)==0:
        re=False
    else:
        re=True
    return re

# first and last workday of each month, each year
def find_first_workday(workday,year,month):
    workday['year'] = workday['Date'].apply(lambda x: x.year)
    workday['month'] = workday['Date'].apply(lambda x: x.month)
    grouped = workday['Date'].groupby([workday['year'], workday['month']])
    workday.drop(['year', 'month'], axis=1, inplace=True)
    first_day = grouped.min()
    re=str(first_day.loc[year,month])
    return(re)

def find_last_workday(workday,year,month):
    workday['year'] = workday['Date'].apply(lambda x: x.year)
    workday['month'] = workday['Date'].apply(lambda x: x.month)
    grouped = workday['Date'].groupby([workday['year'], workday['month']])
    workday.drop(['year', 'month'], axis=1, inplace=True)
    last_day = grouped.max()
    re=str(last_day.loc[year,month])
    return(re)

def date_modify(date,days_add): #if reduce days, days_add input negative int
    #This function is used for modifying date according to natural calendar
    #use datetime for date input form
    re=date+timedelta(days=days_add)
    return(re)

def date_modify_workday(workday,date,days_add): #if reduce days, days_add input negative int
    #This function is used for modifying date on the workday list
    #use datetime for date input form
    workday_list = workday.Date.tolist()
    date_ts = pd.Timestamp(date)
    date_index = workday_list.index(date_ts)
    date_index_modified = date_index + days_add
    re=pd.to_datetime(workday.loc[date_index_modified].tolist()[0]).strftime("%Y-%m-%d")
    return(re)

def nearest_workday(workday,date):
    for i in range(7):
        date_plus_i=date_modify(date,i)
        date_minus_i=date_modify(date,-i)
        if is_workday(workday,date_plus_i)==True:
            re=date_plus_i
            break
        elif is_workday(workday,date_minus_i)==True:
            re = date_minus_i
            break
        else:
            continue
    return re

def date_modify_workday_bymonth(workday,date,months_add): #if reduce months, months_add input negative int
    #This function is used for modifying date on the workday list
    #use datetime for date input form
    #Find the nearest workday when months changed
    date_modified = date + relativedelta(months=months_add)
    #If date_input is the last day of a month,
    #then date_modified is automatically set to be the last day of the corresponding month
    if is_workday(workday,date_modified)==True:
        re=date_modified
    else:
        re=nearest_workday(workday,date_modified)
    return re
    #re type: datetime.date

def minus_workdays(workday,start,end):#start and end are workdays
    start_str= start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    find_start= workday[workday["Date"] == start_str].index.tolist()[0]
    find_end = workday[workday["Date"]== end_str].index.tolist()[0]
    minus_workdays=find_end-find_start
    return minus_workdays

#******************************************
#2. Extension Strategy: Select Contracts

#Compute implied rolling yield (annualized)
def implied_rolling_rate_ann(F,i):
    #input F as a list of future prices:
    #F=[F(t-1,n),F(t,n),F(t,n+i)]
    #F[0]=F(t-1,n)
    #F[1]=F(t,n)
    #F[2]=F(t,n+i)
    re=((F[1]-F[2])/F[0]+1)**(12/i)-1
    return re

#Decide whether a trading day is the extension day
def whether_extension_date(workday,date,list_maturity,k):
    #date should be in form of datetime
    if date.month+1 in list_maturity:
        first_work_day_str=find_first_workday(workday, date.year, date.month)
        first_work_day=datetime.datetime.strptime(first_work_day_str, "%Y-%m-%d %H:%M:%S").date()
        #last_work_day=find_last_workday(workday, date.year, month)
        first_extension_day=first_work_day
        #date_modify_workday(workday,last_work_day,-k)
        one_day_before_extension_day_str=date_modify_workday(workday,first_extension_day,-1)
                                                         #last_work_day,-k-1)
        one_day_before_extension_day= datetime.datetime.strptime(one_day_before_extension_day_str, "%Y-%m-%d").date()
        one_day_after_end_day_str=date_modify_workday(workday, one_day_before_extension_day,k+1)
        one_day_after_end_day = datetime.datetime.strptime(one_day_after_end_day_str, "%Y-%m-%d").date()
        #whether last k days
        if date>one_day_before_extension_day:
            if date==first_extension_day:
                whether_extension_day=1
            elif date<one_day_after_end_day:
                whether_extension_day=minus_workdays(workday,one_day_before_extension_day,date)
            else:
                whether_extension_day=0
        else:
            whether_extension_day = 0
    else:
        whether_extension_day=0
    return whether_extension_day
    #the first extension day: return 1; other extension day: return add; else return 0

#Compute index
def index_compute(workday,date,list_maturity,k,conditions_today,list_contract,num_ext,tick_pre,i_list):
    decide=whether_extension_date(workday, date, list_maturity, k)
    position_list = conditions_today[1]
    W=conditions_today[2]
    if decide==0:#No extension
        index=compute_market_value(date,position_list,W)
        W=[1]
        conditions_today = [index, position_list, W,date]
    elif decide==1:#First extension day
        index=compute_market_value(date,position_list,W)
        W=[1-decide/k,decide/k]
        new_tick=find_new_contract(workday,date,list_contract,num_ext,tick_pre,i_list)
        position_list=position_list+[new_tick]
        conditions_today=[index,position_list,W,date]
    else: #other extension day
        index=compute_market_value(date, position_list, W)
        if decide==k:
            W=[1]
            position_list=[position_list[1]]
        else:
            W=[1-decide/k,decide/k]
        conditions_today = [index,position_list,W,date]
    return conditions_today

# Find the best contract to do extension
def find_new_contract(workday,date,list_contract,num_ext,tick_pre,i_list):
    #date is already 1st extension date
    available_list=find_selection_range(date,list_contract,num_ext,tick_pre)
    data_of_available_contracts=combine_data_of_available_contracts(workday,date,available_list)
    best_contract_num=choose_best_contract(data_of_available_contracts, i_list)
    new_tick=available_list[best_contract_num]
    print(new_tick)
    return new_tick

#Choose best contract -return the no. of the best contract

#data_of_available_contracts must has the following form:
#column index:
#        0              1             2            3              4               ...
#       Date     Now holding  Available no.1  Available no.2  Available no.3      ...
#0   decision_day-1   3692.25      3682.00       3670.25       3660.25       ...
#1   decision_day     3694.25      3692.00       3672.25       3681.25       ...

#use this function when today==decision_date
#def choose_best_contract(data_of_available_contracts,decision_date,i_list):
def choose_best_contract(data_of_available_contracts,i_list):
    #i_list is the months between maturity months of available contracts and the now holding contact
    #input i_list as a list, input decision_date as datetime
    F_list=np.zeros(3)
    F_list[0] = data_of_available_contracts.loc[0].tolist()[1]  # F(t-1,n)
    F_list[1] = data_of_available_contracts.loc[1].tolist()[1]  # F(t,n)
    re=np.zeros(len(i_list))
    for j in range(len(i_list)):
        i=i_list[j]
        F_list[2]=data_of_available_contracts.loc[1].tolist()[j+2] #F(t,n+i)
        re[j]=implied_rolling_rate_ann(F_list,i)
    re=np.argmax(re)+1 #no.1,2,3,4,... from available contracts (exclude now holding contract)
    return re #best contract column index in dataframe of future prices: (re+1)

def combine_data_of_available_contracts(workday,extension_date_1st,available_list):
    column_list=["Date"]
    column_list=column_list+available_list
    F = pd.DataFrame(data=None, columns=column_list)
    date_start_str=date_modify_workday(workday,extension_date_1st,-1)
    date_start=datetime.datetime.strptime(date_start_str, "%Y-%m-%d").date()
    d1={"Date": date_start_str}
    for i in range(len(available_list)):
        d1[available_list[i]]=read_price(date_start,available_list[i])
    F=F.append(d1, ignore_index=True)
    extension_date_1st_str=extension_date_1st.strftime("%Y-%m-%d")
    d2={"Date": extension_date_1st_str}
    for i in range(len(available_list)):
        d2[available_list[i]] = read_price(extension_date_1st, available_list[i])
    F = F.append(d2, ignore_index=True)
    F=F.reset_index(drop=True)
    return F
    #F=data_of_available_contracts

def find_selection_range(date,list_contract,num_ext,tick_pre):
    #list_contract=[3,6,9,12]
    #num_ext: how many future contracts to be selected
    month_now=date.month
    year_now=date.year
    """
    for i in range(len(list_contract)):
        if i>0:
            if list_contract[i]>month_now and list_contract[i-1]<month_now:
                maturity_month=list_contract[i]
            elif month_now<list_contract[0]:
                maturity_month = list_contract[0]
            else:
                maturity_month = list_contract[len(list_contract)-1]
    """
    maturity_month=month_now+1
    #month_tick_list=["F","G","H","J","K","M","N","Q","U","V","X","Z"]
    #m_ann=12
    #mod_num=1
    month_tick_list = [ "H","M", "U", "Z"]
    m_ann = 4
    mod_num = 12/m_ann
    available_list=[]
    for i in range(num_ext+1):
        if maturity_month/mod_num+i<m_ann+1:
            contract_new=tick_pre+month_tick_list[int(maturity_month/mod_num)+i-1]+str(year_now-2000)
            available_list.append(contract_new)
        else:
            contract_new = tick_pre + month_tick_list[int(maturity_month/mod_num)+i-m_ann-1] + str(year_now+1-2000)
            available_list.append(contract_new)
    print(available_list)
    return available_list
    #available_list=["ESH21",...]includes now holding one


def compute_market_value(date,position_list,W):
    #W: Weight of the list;
    # position_list is list of ticks;
    # W is [] list;
    # Num_position is number of contracts for one certain forward, i.e. =N
    N=len(W)
    market_value=0
    for i in range(N):
        tick=position_list[i]
        mkt_price=read_price(date,tick)
        weight=W[i]
        market_value=market_value+mkt_price*weight
    return market_value

def read_price(date,tick):
    #date is datetime, tick is string like "ESH1"
    if isinstance(tick,list):
        tick=tick[0]
    #print(tick)
    path=tick+" Index.xlsx"
    data=pd.read_excel(path)
    date_str = date.strftime("%Y-%m-%d")
    index_date = data[data.Date == date_str].index.tolist()
    price = data[tick+" Index"].iloc[index_date].tolist()[0]
    return price


#******************************************
#3. Extension Strategy: Market Timing

def find_maturity_date(tick):
    path=tick+" Index.xlsx"
    df = pd.read_excel(path)
    df = df.sort_values(by="Date", ascending=True)
    df = df.reset_index(drop=True)
    date_max = max(df["Date"])
    date_max = pd.Timestamp.date(date_max)
    return date_max
    #date_max type: datetime.date

def find_maturity_month(tick,list_maturity,list_month_tick,pre):
    # list_maturity = [3, 6, 9, 12]
    # list_month_tick = ["H", "M", "U", "Z"]
    # pre="ES"
    n = len(pre)
    m = len(tick)
    month_tick=tick[n:m - 2]
    index_loc=list_month_tick.index(month_tick)
    month_maturity=list_maturity[index_loc]
    return month_maturity
    # month=3,6,9,12,...

def find_month_tick(tick,list_month_tick,pre):
    # list_maturity = [3, 6, 9, 12]
    # list_month_tick = ["H", "M", "U", "Z"]
    # pre="ES"
    n = len(pre)
    m = len(tick)
    month_tick=tick[n:m - 2]
    return month_tick
    # month_tick="H","M","U",...

def find_next_tick(tick_now,list_month_tick,pre):
    tick_month_now = find_month_tick(tick_now, list_month_tick, pre)
    index_loc = list_month_tick.index(tick_month_now)
    year_add_now=tick_now[len(tick_now)-2:len(tick_now)]
    if index_loc<len(list_month_tick)-1:
        tick_month_next = list_month_tick[index_loc + 1]
        tick_next=pre+tick_month_next+year_add_now
    else:
        tick_month_next = list_month_tick[0]
        tick_next=pre+tick_month_next+str(int(year_add_now)+1)
    return tick_next

def mkt_timing(date,workday,conditions,list_month_tick,pre,pre_days):
    # conditions=[index,holding_tick,main_tick]
    # list_maturity = [3, 6, 9, 12]
    # list_month_tick = ["H", "M", "U", "Z"]
    index_now=conditions[0]
    tick_now=conditions[1]
    tick_main=conditions[2]
    maturity_date_main = find_maturity_date(tick_main)
    next_day=date_modify_workday(workday,maturity_date_main,1)
    next_day=datetime.datetime.strptime(next_day, "%Y-%m-%d").date()
    if date == next_day:
        tick_pre_main = tick_main
        tick_main = find_next_tick(tick_pre_main, list_month_tick, pre)
        if tick_now == tick_pre_main:
            tick_now = tick_main
    if tick_now==tick_main:
        maturity_date_now=find_maturity_date(tick_now)
        tick_next=find_next_tick(tick_now,list_month_tick,pre)
        pre_month=3
        start_monitor_date=date_modify_workday_bymonth(workday,maturity_date_now,-pre_month)
        start_irr_date=date_modify_workday_bymonth(workday,maturity_date_now,-pre_month-1)
        price_now_today = read_price(date, tick_now)
        yesterday = date_modify_workday(workday, date, -1)
        yesterday = datetime.datetime.strptime(yesterday, "%Y-%m-%d").date()
        price_now_yesterday = read_price(yesterday, tick_now)
        price_next_today = read_price(date, tick_next)
        if date > start_monitor_date:
            #compute implied rate of return between tick_now and tick_next
            i=3
            irr_ann_today=(1+(price_now_today-price_next_today)/price_now_yesterday)**(12/i)-1
            k0=1.1
            k1=0.9
            k_date = compute_k_threshold(workday, date, maturity_date_now,k0,k1)
            ave_impl=ave_impl_re(workday,date,tick_now,tick_next,pre_days)
            threshold = ave_impl * k_date
            if irr_ann_today > threshold:
                signal = True
            else:
                #signal = market_decision(workday, date, maturity_date_now)
                # 若一直没有信号触发，则根据市场持仓量展期，展期收益增强指数实际展期时间不晚于市场。
                signal=False
            if signal==True:
                index_today=index_now*price_next_today/price_now_today
                tick_today=tick_next
                conditions = [index_today, tick_today,tick_main]
            else:
                index_today=index_now*price_now_today/price_now_yesterday
                tick_today=tick_now
                conditions=[index_today,tick_today,tick_main]
        else:
            index_today = index_now * price_now_today / price_now_yesterday
            tick_today = tick_now
            conditions = [index_today, tick_today,tick_main]
    else:
        price_now_today = read_price(date, tick_now)
        yesterday = date_modify_workday(workday, date, -1)
        yesterday = datetime.datetime.strptime(yesterday, "%Y-%m-%d").date()
        price_now_yesterday = read_price(yesterday, tick_now)
        index_today = index_now * price_now_today / price_now_yesterday
        tick_today = tick_now
        conditions = [index_today, tick_today, tick_main]
    return conditions

def ave_impl_re0(workday,date,tick_now,tick_next,pre_days):
    summ=0
    for j in range(pre_days):
        yesterday_str = date_modify_workday(workday, date, -1)
        yesterday = datetime.datetime.strptime(yesterday_str, "%Y-%m-%d").date()
        price_now_yesterday = read_price(yesterday, tick_now)
        price_now_today = read_price(date, tick_now)
        price_next_today = read_price(date, tick_next)
        i = 3
        irr_ann_today = (1 + (price_now_today - price_next_today) / price_now_yesterday) ** (12 / i) - 1
        summ=summ+irr_ann_today
        date=date_modify_workday(workday,date,-1)
        date=datetime.datetime.strptime(date, "%Y-%m-%d").date()
    ave=summ/pre_days
    return ave

def ave_impl_re(workday,date,tick_now,tick_next,pre_days):
    path_now=tick_now+" Index.xlsx"
    data_now = pd.read_excel(path_now)
    data_now = data_now.sort_values(by="Date", ascending=True)
    data_now = data_now.reset_index(drop=True)
    path_next = tick_next + " Index.xlsx"
    data_next = pd.read_excel(path_next)
    data_next = data_next.sort_values(by="Date", ascending=True)
    data_next = data_next.reset_index(drop=True)
    date_str=date.strftime("%Y-%m-%d")
    index_date_now = data_now[data_now.Date == date_str].index.tolist()[0]
    index_start_now=index_date_now-pre_days-1
    index_end_now=index_date_now-1
    index_date_next = data_next[data_next.Date == date_str].index.tolist()[0]
    index_start_next = index_date_next - pre_days - 1
    index_end_next = index_date_next - 1
    price_now = data_now[tick_now+" Index"].iloc[index_start_now:index_end_now].tolist()
    price_next = data_next[tick_next + " Index"].iloc[index_start_next:index_end_next].tolist()
    price_now=np.array(price_now)
    price_next = np.array(price_next)
    i=3
    irr=((price_now[1:pre_days]-price_next[1:pre_days])/price_now[0:pre_days-1]+1)**(12/i)-1
    ave=np.mean(irr)
    return ave


def available_two_list(date, list_maturity, pre_tick, list_month_tick,tick_now):
    # list_maturity=[3,6,9,12]
    # list_month_tick=["H","M","U","Z"]
    month_now = date.month
    year_now = date.year
    pre_m=3
    # pre_m: Begin monitor implied rate of return 3 months ahead of now holding contract maturity
    if np.mod(month_now + pre_m,12) in list_maturity:
        if month_now + pre_m<13:
            month_maturity_now = month_now + pre_m
            year_maturity=year_now
        else:
            month_maturity_now=month_now + pre_m-12
            year_maturity=year_now+1
        index_now = list_maturity.index(month_maturity_now)
        tick_month_now = list_month_tick[index_now]
        if index_now < len(list_maturity) - 1:
            index_next = index_now + 1
            year_next = year_maturity
        else:
            index_next = 0
            year_next = year_maturity + 1
        # month_maturity_next = list_maturity[index_next]
        tick_month_next = list_month_tick[index_next]
        # Now we have the current contract's maturity month and the next main contact's maturity month
        tick_now = pre_tick + tick_month_now + str(np.mod(year_maturity, 100))  # "ESH21"
        tick_next = pre_tick + tick_month_next + str(np.mod(year_next, 100))  # "ESM21"
        available_list = [tick_now, tick_next] #["ESH21","ESM21"]
    return available_list


def extension_signal(workday,date,list_maturity,pre_tick,list_month_tick,i,pre_days):
    # list_maturity=[3,6,9,12]
    # list_month_tick=["H","M","U","Z"]
    month_now = date.month
    year_now = date.year
    pre_m = 3
    # pre_m: Begin monitor implied rate of return 3 months ahead of now holding contract maturity
    if np.mod(month_now + pre_m, 12) in list_maturity:
        if month_now + pre_m < 13:
            month_maturity_now = month_now + pre_m
            year_maturity = year_now
        else:
            month_maturity_now = month_now + pre_m - 12
            year_maturity = year_now + 1

        available_list=available_two_list(date, list_maturity, pre_tick, list_month_tick)
        data_of_available_contracts=combine_data_of_available_contracts(workday,date,available_list)

        tick_maturity_now=available_list[0]
        path_now=tick_maturity_now+" Index.xlsx"
        data_now=pd.read_excel(path_now)
        date_maturity_stp = max(data_now["Date"])
        date_maturity=pd.Timestamp.date(date_maturity_stp)
        date_move=date_modify_workday_bymonth(workday,date_maturity,-3)
        move_month=date_move.month
        move_year=date_move.year
        date_start_cnt_str=find_first_workday(workday,move_year,move_month)
        date_start_cnt= datetime.datetime.strptime(date_start_cnt_str, "%Y-%m-%d %H:%M:%S").date()
        N_3m=minus_workdays(workday,date_start_cnt,date_maturity)
        signal=decide_threshold(workday, date, data_of_available_contracts, i, pre_days, list_maturity, pre_tick,
                         list_month_tick, N_3m, date_maturity)
    else:
        signal=False
    print(signal)
    return signal
    #return True: do extension next day; False: don't do extension next day.


def decide_threshold(workday,date,data_of_available_contracts,i,pre_days, list_maturity, pre_tick, list_month_tick, N_3m, maturity_date):
    impl_re=compute_implied_rate(data_of_available_contracts,i)
    threshold=threshold_compute(workday, date, pre_days, list_maturity, pre_tick, list_month_tick, N_3m, maturity_date,i)
    if impl_re>threshold:
        signal=True
    else:
        signal=market_decision(workday,date,maturity_date)
        # 若一直没有信号触发，则根据市场持仓量展期，展期收益增强指数实际展期时间不晚于市场。
    return signal

def compute_implied_rate(data_of_available_contracts,i):
    #i = 3  # for SPX Index, the interval for two contracts are 3 months
    F_list = np.zeros(3)
    F_list[0] = data_of_available_contracts.loc[0].tolist()[1]  # F(t-1,n)
    F_list[1] = data_of_available_contracts.loc[1].tolist()[1]  # F(t,n)
    F_list[2] = data_of_available_contracts.loc[1].tolist()[2]  # F(t,n+i)
    impl_re = implied_rolling_rate_ann(F_list, i)
    return impl_re

def threshold_compute(workday,date,pre_days,list_maturity,pre_tick,list_month_tick,N_3m,maturity_date,i):
    #pre_days=40 #Threshold= (pre_days average of implied rate of returns)*k_t
    available_list=available_two_list(date,list_maturity,pre_tick,list_month_tick)
    date_1st_str=date_modify_workday(workday,date,-pre_days)
    date_1st = datetime.datetime.strptime(date_1st_str, "%Y-%m-%d").date()
    ave_impl_re=ave_implied_rate_predays(workday,date_1st,available_list,pre_days,i)
    k_date=compute_k_threshold(workday,date,N_3m,maturity_date)
    threshold=ave_impl_re*k_date
    print(threshold)
    print(k_date)
    return threshold

def ave_implied_rate_predays(workday,date_1st,available_list,pre_days,i):
    column_list=["Date"]
    column_list=column_list+available_list
    F = pd.DataFrame(data=None, columns=column_list)
    for t in range(pre_days):
        date_str=date_modify_workday(workday,date_1st,-t-1)
        date=datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        d={"Date": date_str}
        data_of_available_contracts = combine_data_of_available_contracts(workday, date, available_list)
        impl_rate=compute_implied_rate(data_of_available_contracts, i)
        d["Implied rate of return"]=impl_rate
        F = F.append(d, ignore_index=True)
    F=F.reset_index(drop=True)
    ave_impl_re=sum(F["Implied rate of return"])/pre_days
    return ave_impl_re
    #average of previous 40 days' implied rate of return

def compute_k_threshold(workday,date,maturity_date,k0,k1):
    #Here define monotonically decreasing linear function to allocate threshold coefficient k
    #On the first day, k0=1.1. last day: k1=0.9;
    N_days=minus_workdays(workday,date,maturity_date)
    pre_month=3
    start_date=date_modify_workday_bymonth(workday,date,-pre_month)
    N_3m=minus_workdays(workday,start_date,maturity_date)
    k=N_days*(k0-k1)/N_3m+k1
    return k

def market_decision(workday,date,maturity_date):
    end_date=date_modify_workday(workday,maturity_date,-2)
    end_date=datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    if date<end_date:
        decision=False
    else:
        decision=True
    return decision
    #decision True/False

def compute_Index_mkt_timing(workday, date, conditions,list_maturity, pre_tick, list_month_tick, i, pre_days):
    signal=extension_signal(workday, date, list_maturity, pre_tick, list_month_tick, i, pre_days)
    index=conditions[0]
    tick=conditions[1]
    yesterday_str=date_modify_workday(workday,date,-1)
    yesterday=datetime.datetime.strptime(yesterday_str, "%Y-%m-%d").date()
    price_pre=read_price(yesterday,tick)
    if signal==True:
        available_list = available_two_list(workday, date, list_maturity, pre_tick, list_month_tick)
        tick_new=available_list[1]
        price_today = read_price(date, tick_new)
        index_today=price_today*index/price_pre
    else:
        price_today = read_price(date, tick)
        index_today=price_today*index/price_pre
        tick_today=tick
    conditions_today=[index_today,tick_today]
    return conditions_today


#**********************
def whether_ext(date,data):
    ext_signal

    return ext_signal


def max_holdings(date,data):
    date_str = date.strftime("%Y-%m-%d")
    index_date = data[data.Date == date_str].index.tolist()
    CF01M_H = data["CF01M_H"].iloc[index_date].tolist()[0]
    CF05M_H = data["CF05M_H"].iloc[index_date].tolist()[0]
    CF09M_H = data["CF09M_H"].iloc[index_date].tolist()[0]
    Holdings=[CF01M_H,CF05M_H,CF09M_H]
    max_index=Holdings.index(max(Holdings))
    return max_index



