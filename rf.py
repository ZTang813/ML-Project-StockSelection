# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 08:27:12 2018

@author: Administrator
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import talib as ta
import calendar
from datetime import timedelta,datetime
import pymssql
import seaborn as sns

conn=pymssql.connect(               #connect
	server='192.168.0.28',
	port=1433,
	user='sa',
	password='abc123',
	database='rawdata'
	)

def get_stock_day(stock,start_date,end_date,freq = '1D'):

    SQL_code = "S_INFO_WINDCODE="+"\'"+stock+"\'"

    SQL_date ="TRADE_DT between "+"\'"+start_date+"\' and \'" + end_date+"\'"

    SQL_price = "select S_DQ_ADJOPEN,S_DQ_ADJHIGH,S_DQ_ADJLOW,S_DQ_ADJCLOSE,S_DQ_VOLUME,TRADE_DT from ASHAREEODPRICES where "+SQL_code+' and '+SQL_date+' and S_DQ_VOLUME>0'

    price = pd.read_sql(SQL_price,conn,index_col='TRADE_DT')
    
    return price

def get_index_day(index,start_date,end_date,freq = '1D'):

    index_code2,index_code1 = index.split('.')

    SQL_code = "scode1="+"\'"+index_code1+"\' and scode2=\'"+index_code2+"\'"

    SQL_date ="ddate between "+"\'"+start_date+"\' and \'" + end_date+"\'"

    SQL_price = "select sclose,sopen,high,low,ddate from indexs_day where "+SQL_code+' and '+SQL_date

    SQL_vol = "select volumn,ddate from indexs_day where "+SQL_code+' and '+SQL_date

    price = pd.read_sql(SQL_price,conn,index_col='ddate')
    
    close = price.sclose.resample(freq).last().dropna()
    
    open = price.sopen.resample(freq).first().dropna()
    
    high = price.high.resample(freq).max().dropna()
    
    low = price.low.resample(freq).min().dropna()
    
    vol = pd.read_sql(SQL_vol,conn,index_col='ddate')

    vol = vol.resample(freq).sum().dropna()
    
    price = pd.concat([close,open,high,low],axis=1)

    data = price.join(vol)
    
    return data

pool_dic={}
u=set()
universe=pd.read_excel('code.xlsx')
pool_dic[universe.index[0].strftime('%Y%m')]=universe.ix[0].tolist()
for i in range(1,len(universe)):
    if universe.index[i].month!=universe.index[i-1].month:
        pool_dic[universe.index[i].strftime('%Y%m')]=universe.ix[i-1].tolist()
        u=u.union(universe.ix[i-1].tolist())
all_stock={}
for stock in u:
    SS=get_stock_day(stock,'2009-01-01','2018-08-01',freq='1D')
    
    MA = pd.DataFrame()
    for i in range(10,90,10):
        local_MA = ta.MA(SS.S_DQ_ADJCLOSE,timeperiod = i)
        local_MA.name = 'MA'+str(i)
        MA = pd.concat([MA,local_MA],axis=1)
    
    MACD1,MACD2,XX = ta.MACD(SS.S_DQ_ADJCLOSE)
    MACD = pd.concat([MACD1,MACD2],axis=1)
    ADX = ta.ADX(SS.S_DQ_ADJHIGH,SS.S_DQ_ADJLOW,SS.S_DQ_ADJCLOSE)
    ADXR = ta.ADXR(SS.S_DQ_ADJHIGH,SS.S_DQ_ADJLOW,SS.S_DQ_ADJCLOSE)
    aroondown,aroonup = ta.AROON(SS.S_DQ_ADJHIGH,SS.S_DQ_ADJLOW)
    ATR = ta.ATR(SS.S_DQ_ADJHIGH,SS.S_DQ_ADJLOW,SS.S_DQ_ADJCLOSE)
    Bupper,Bmiddle,Blower = ta.BBANDS(SS.S_DQ_ADJCLOSE)
    group1 = pd.concat([SS,MA,MACD,ADX,ADXR,aroondown,aroonup,ATR,Bupper,Bmiddle,Blower],axis=1)
    
    BOP = ta.BOP(SS.S_DQ_ADJOPEN,SS.S_DQ_ADJHIGH,SS.S_DQ_ADJLOW,SS.S_DQ_ADJCLOSE)
    CCI = ta.CCI(SS.S_DQ_ADJHIGH,SS.S_DQ_ADJLOW,SS.S_DQ_ADJCLOSE)
    CMO = ta.CMO(SS.S_DQ_ADJCLOSE)
    DEMA = ta.DEMA(SS.S_DQ_ADJCLOSE)
    DX = ta.DX(SS.S_DQ_ADJHIGH,SS.S_DQ_ADJLOW,SS.S_DQ_ADJCLOSE)
    EMA = ta.EMA(SS.S_DQ_ADJCLOSE)
    KAMA = ta.KAMA(SS.S_DQ_ADJCLOSE)
    MFI = ta.MFI(SS.S_DQ_ADJHIGH,SS.S_DQ_ADJLOW,SS.S_DQ_ADJCLOSE,SS.S_DQ_VOLUME)
    MOM = ta.MOM(SS.S_DQ_ADJCLOSE)
    RSI = ta.RSI(SS.S_DQ_ADJCLOSE)
    group2 = pd.concat([BOP,CCI,CMO,DEMA,DX,EMA,KAMA,MFI,MOM,RSI],axis=1)
    
    SAR = ta.SAR(SS.S_DQ_ADJHIGH,SS.S_DQ_ADJLOW)
    TEMA = ta.TEMA(SS.S_DQ_ADJCLOSE)
    TRANGE = ta.TRANGE(SS.S_DQ_ADJHIGH,SS.S_DQ_ADJLOW,SS.S_DQ_ADJCLOSE)
    TRIMA = ta.TRIMA(SS.S_DQ_ADJCLOSE)
    TRIX = ta.TRIX(SS.S_DQ_ADJCLOSE)
    group3 = pd.concat([SAR,TEMA,TRANGE,TRIMA,TRIX],axis=1)
    
    raw_ta = pd.concat([group1,group2,group3],axis=1)
    raw_ta=raw_ta.dropna()
    all_stock[stock]=raw_ta
SS = get_index_day('000001.SH','2009-01-01','2018-08-01',freq='1D')
return_rate=[]
for i in range(len(SS)-22):
    return_rate.append((SS.sclose[i+22]-SS.sclose[i])/SS.sclose[i])
return_rate=return_rate+[np.nan]*22
SS['return_rate']=return_rate

stock_all=all_stock.copy()
#stock_all['000001.SH']=SS
for key, value in stock_all.items():
    return_rate=[]
    for i in range(len(value)-22):
        return_rate.append((value.S_DQ_ADJCLOSE[i+22]-value.S_DQ_ADJCLOSE[i])/value.S_DQ_ADJCLOSE[i])
    return_rate=return_rate+[np.nan]*22
    value['return_rate_stock']=return_rate
    target=SS[['return_rate']].join(value,how='inner')
    label=[]
    for j in range(len(target)):
        if target.return_rate_stock[j]>target.return_rate[j]:
            label.append(1)
        else:
            label.append(0)
    value['label']=label
    del value['return_rate_stock']

for key,value in stock_all.items():
    value.index=pd.to_datetime(value.index)

train=pd.DataFrame()
for key, value in pool_dic.items():
    for j in value:
        train=train.append(stock_all[j][key[4:]+'/1/'+key[:4]:key[4:]+'/'+str(calendar.monthrange(int(key[:4]),int(key[4:]))[1])+'/'+key[:4]])
train=train.sort_index()
p={}
for i in range(len(universe)):
    if universe.index[i].month!=universe.index[i-1].month or i==0:
        df=pd.DataFrame()
        code=[]
        for j in pool_dic[universe.index[i].strftime('%Y%m')]:
            df=df.append(stock_all[j][universe.index[i].strftime("%m/%d/%Y"):universe.index[i].strftime("%m/%d/%Y")])
            if not stock_all[j][universe.index[i].strftime("%m/%d/%Y"):universe.index[i].strftime("%m/%d/%Y")].empty:
                code.append(j)
        df.index=code     
        p[universe.index[i].strftime('%Y%m%d')]=df
        

draw=pd.DataFrame()
hold_all=[]
for k in range(5):
    net_value=1
    pnl=[1]
    correct1=[]
    correct2=[]
    ind=[]
    hold={}
    for day in p.keys():
        n=net_value
        ind.append(day)
        if day!='20100104' and day!='20100201' and net_value>0:
            t=train[(pd.to_datetime(day)-timedelta(days=180)).strftime("%m/%d/%Y"):(pd.to_datetime(day)-timedelta(days=35)).strftime("%m/%d/%Y")]
            x_train=t.iloc[:,:-1]
            y_train=t.iloc[:,-1]
            x_predict=p[day].iloc[:,:-1]
            real_predict=p[day].iloc[:,-1]
            rf = RandomForestClassifier(n_estimators=100,oob_score=True)
            rf.fit(x_train,y_train)
            correct1.append (rf.score(x_predict, real_predict))
            correct2.append (rf.score(x_train, y_train))
            y_predict=rf.predict(x_predict)
            pos={}
            position=[]
            for i in range(len(y_predict)):
                if y_predict[i]==1:
                    pos[x_predict.index[i]]=x_predict.S_DQ_ADJCLOSE[i]
                    position.append(x_predict.index[i])
            length=len(pos)
#            if length<5:
#                pos['000001.SH']=SS.loc[pd.to_datetime(key)].S_DQ_ADJCLOSE
            pop_list=[]
            hold[day]=position
            for key,value in pos.items():
                df=stock_all[key][day[4:6]+'/1/'+day[:4]:day[4:6]+'/'+str(calendar.monthrange(int(day[:4]),int(day[4:6]))[1])+'/'+day[:4]]
                for j in range(len(df)):
                    if (df.S_DQ_ADJCLOSE[j]-value)/value<-0.05:
                        pop_list.append(key)
                        net_value=net_value+(df.S_DQ_ADJCLOSE[j]-value)*net_value/length/value*1.006
                        break
                if not key in pop_list:
                    net_value=net_value+(df.S_DQ_ADJCLOSE[-1]-value)*net_value/length/value*0.994
            pnl.append(net_value)
    hold_all.append(hold)
    draw['NetValue'+str(k)]=pnl
ind.remove('20100104')
draw.index=ind
index=[]
for day in p.keys():
    if day!='20100104' and day!='20100201':
        index.append(SS.ix[SS.index.tolist().index(pd.to_datetime(day))-1].sclose/SS.loc[pd.to_datetime('20100301')].sclose)
index.append(SS.loc[pd.to_datetime('20180629')].sclose/SS.loc[pd.to_datetime('20100301')].sclose)
draw['index']=index
draw.plot()

df_collect=[]
for cnt in range(5):
    df = pd.DataFrame()
    for key,values in hold_all[cnt].items():
        local_df = pd.DataFrame(index=[key],data=np.array([values]))
        df = pd.concat([df,local_df])
    df.to_excel('run'+str(cnt)+'.xlsx')
    df_collect.append(df)
    
def overlap(df1,df2):
    collect = []
    for i in range(len(df1)):
        cnt = 0
        for j in df1.iloc[i].dropna():
            for k in df2.iloc[i].dropna():
                if j ==k:
                    cnt+=1
        collect.append(cnt/max(len(df1.iloc[i].dropna()),len(df2.iloc[i].dropna())))
    ovp_df = pd.DataFrame(index = df.index,data =collect)
    return ovp_df

ovp_collect =[]

for cnt1 in range(5):
    local_ovp_collet=[]
    for cnt2 in range(5):
        x =  overlap(df_collect[cnt1],df_collect[cnt2])
        local_ovp_collet.append(x.mean().values[0])
    ovp_collect.append(local_ovp_collet)

sns.heatmap(ovp_collect)

    
    


    


#hold_all是每月月初持仓
