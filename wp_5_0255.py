from math import inf
from turtle import left
import numpy as np
import pytz
from datetime import datetime
import pandas as pd
import sxsc_tushare as sx
import ast



def initialize(context):
    info("--------------Run initialize----------------", 'info') 
    pd.option_context('display.max_rows', 50)
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', 1000)  # 增加列展示宽度
    pass

# 盘前处理
def before_trading_start(context, data):
    g.deal_hand_times = 235    #交易次数
    g.deal_hand_time = '02:55'  #交易时间
    g.sell_hand_times = 30    #交易次数
    g.hand_times = 0    #交易次数
    g.buyedStocks = {}  #需买入的股票
    g.bDebug=False   #是否调试
    g.arrStocks = []    #可选股票
    g.arrTurnoverStocks = {}    #换手率股票
    g.bFinishedCalc = False   #是否完成计算
    g.nBuyMoney = 250000 / 7    #买入金额
    g.mailBuyStocks = {}

def after_trading_end(context, data):
    info("--------------Run after_trading_end----------------", 'info') 
    info(f"{str(g.deal_hand_time)}发送邮件{str(g.buyedStocks)},{str(g.mailBuyStocks)}", 'info') 
    # msg = f"{str(g.deal_hand_time)}发送邮件{str(g.buyedStocks)},{str(g.mailBuyStocks)}"
    # send_email('54229670@qq.com', ['54229670@qq.com', '290291989@qq.com', '68043432@qq.com'], 'zfethvvdbmcbbjaa', info=msg)

#0.取得主板所有股票
def getStocksByBoard():
    arrStocks = get_Ashares()
    info(f'0、主板股票数量为{len(arrStocks)}', 'debug')
    info(arrStocks, 'debug')
    return arrStocks

#step 1 筛选位于主板的，市值 30-400 亿的股票。只筛选出位于沪深主板的股票，剔除ST 类股票，剔除主板以外科创板、创业板等所有非主板股票。
def getStocksByTotalValue(arrStocks, min = 30, max = 400):
    df = get_fundamentals(arrStocks, 'valuation', fields = ['total_value', 'pe_dynamic', 'turnover_rate', 'pb'])
    df_selected = df[(df['total_value'] >= min * 100000000) & (df['total_value'] <= max * 100000000)]
    info(f'1.1、市值 30-400 亿的股票数量为{len(df_selected)}', 'debug')
    info(df_selected, 'debug')
    retArrStocks = df_selected.index.tolist()
    info(f'1.1、市值 30-400 亿的股票数量为{len(retArrStocks)}', 'info')
    info(retArrStocks, 'info')
    #过滤ST、停牌、退市的股票
    retArrStocks = filter_ST(retArrStocks)
    info(f'1.2、过滤ST、停牌、退市的股票数量为{len(retArrStocks)}', 'info')
    info(retArrStocks, 'info')
    #过滤非主板股票
    retArrStocks = filter_not_mainboard(retArrStocks)
    info(f'1.2、过滤非主板股票数量为{len(retArrStocks)}', 'info')
    info(retArrStocks, 'info')
    return retArrStocks


#2.当日股票涨幅不低于+3%，且上影线长度不大于 K 线实体的 1/5，剔除之前最大涨幅达到涨停的股票。
def getStocksByPxChangeRateAndShadowLine(arrStocks):
    #1、取得今日K线
    klines = get_history(1, '1d', ['open','high','low','close', 'high_limit'], arrStocks, fq=None, include=True)
    klines['涨跌幅'] = ((klines['close'] - klines['open'])/klines['open']).round(2)
    klines['上影线'] = (klines['high'] - klines['close']).round(2)
    klines['实体'] = (klines['close'] - klines['open']).round(2)
    klines['占比'] = (klines['上影线'] / klines['实体']).round(2)
    klines = klines[~np.isinf(klines['涨跌幅'])]
    info(f'2.1、取得今日K线{len(klines)}', 'debug')  
    info(klines, 'debug')
    info(f'2.1、取得今日K线{len(klines)}', 'info')  
    info(klines['code'].tolist(), 'info')
    #2、判断涨幅是否不低于+3%
    df_klines = klines[(klines['close'] - klines['open'])/klines['open'] >= 0.03]
    info(f'2.2、涨幅不低于+3%的股票数量为{len(df_klines)}', 'debug')
    info(df_klines, 'debug')
    info(f'2.2、涨幅不低于+3%的股票数量为{len(df_klines)}', 'info')
    info(df_klines['code'].tolist(), 'info')
    #3、判断上影线长度是否不大于 K 线实体的 1/5
    df_klines = df_klines[df_klines['high'] - df_klines['close'] <= (df_klines['close'] - df_klines['open']) / 5]
    info(f'2.3、上影线长度不大于 K 线实体的 1/5的股票数量为{len(df_klines)}', 'debug')
    info(df_klines, 'debug')
    info(f'2.3、上影线长度不大于 K 线实体的 1/5的股票数量为{len(df_klines)}', 'info')
    info(df_klines['code'].tolist(), 'info')
    # #4、剔除之前最大涨幅达到涨停的股票
    df_klines = df_klines[df_klines['high_limit'] > df_klines['high']]
    info(f'2.4、剔除之前最大涨幅达到涨停的股票数量为{len(df_klines)}', 'debug')
    info(df_klines, 'debug')
    info(f'2.4、剔除之前最大涨幅达到涨停的股票数量为{len(df_klines)}', 'info')
    info(df_klines['code'].tolist(), 'info')
    #6、返回满足条件的股票
    return df_klines['code'].tolist()

#3 从交易日 2:30 分之后开始选股，所选股票当日换手率大于 5%。
def getStocksByTurnoverRate(arrStocks, context, data):
    arrTurnoverStocks = {}
    arrBuyStocks = {}
    if g.bDebug:
        date = context.blotter.current_dt.strftime("%Y-%m-%d")
        info(f'++++++++++handle_data取换手率股票{str(arrStocks)} - {date}++++++++++++', 'info')
        for stock in arrStocks:
            # df = get_fundamentals(stock, 'valuation', date = date, fields = ['total_value', 'pe_dynamic', 'turnover_rate', 'pb'])
            # info(df, 'debug')
            # turnover_rate = df.loc[stock, 'turnover_rate']
            turnover_rate = 5.1
            info(f'股票{stock}换手率{turnover_rate}', 'debug')
            if(turnover_rate >= 5):
                arrTurnoverStocks[stock] = turnover_rate
                arrBuyStocks[stock] = data[stock].close
    else:
        info(f'++++++++++tick_data取换手率股票{str(arrStocks)}++++++++++++', 'info')
        for stock in g.arrStocks:
            turnover_ratio = float(data[stock]['tick']['turnover_ratio'])
            #取得当前买二买三价
            buy_price, buy_price_2 = getBuyPrice(stock)
            last_px = float(data[stock]['tick']['last_px'])
            info(f'股票{stock}换手率{turnover_ratio}，买二买三价{buy_price}，{buy_price_2}, 最新价{last_px}', 'debug')
            if(turnover_ratio >= 0.05):
                arrTurnoverStocks[stock] = turnover_ratio
                arrBuyStocks[stock] = buy_price    
    if g.bFinishedCalc:
        getBuyStocks(arrBuyStocks, context)
        g.bFinishedCalc = False
    
    info(f'3、从交易日 {g.deal_hand_time} 分之后开始选股，所选股票当日换手率大于 5% {g.bDebug} - {len(arrTurnoverStocks)}', 'info')
    info(arrTurnoverStocks, 'info')
    return arrTurnoverStocks

#4、从交易日 2:30 分之后开始选股，当日成交量大于前五日平均成交量的 1.5倍。且当日成交量比前五日每一天的成交量均大 1.5 倍。
def getStocksByVolume(arrStocks):
    #1、取得前5日的平均成交量，包括今天
    df = get_history(6, '1d', 'volume', arrStocks, fq=None, include=True)
    info(f'取得前5日的平均成交量，包括今天{len(df)}', 'debug')  
    info(df, 'debug')
    #2、循环每5行数据，计算平均成交量
    df['mean'] = df['volume'].rolling(window=5).mean()
    #设置volume_5为，前第5日的成交量
    df['volume_5'] = df['volume'].shift(5)
    #设置volume_4为，前第4日的成交量
    df['volume_4'] = df['volume'].shift(4)
    #设置volume_3为，前第3日的成交量
    df['volume_3'] = df['volume'].shift(3)
    #设置volume_2为，前第2日的成交量
    df['volume_2'] = df['volume'].shift(2)
    #设置volume_1为，前第1日的成交量
    df['volume_1'] = df['volume'].shift(1)
    #设置mean_last为上一行的mean值
    df['mean_last'] = df['mean'].shift(1)

    info(f'循环每5行数据，计算平均成交量{len(df)}', 'debug')  
    info(df, 'debug')
    #4 取得当前日期
    cur_date = df.index[-1]
    info(f'取得当前日期{cur_date}', 'info')
    df = df[df.index == cur_date]
    info(f'今日股票{len(df)}', 'info')
    info(df, 'info')
    #5、当日成交量比前五日每一天的成交量均大 1.5 倍。
    df_selected = df[(df['volume'] / g.hand_times > df['mean_last'] / 240 * 1.5) & (df['volume'] > df['volume_5'] * 1.5) & (df['volume'] > df['volume_4'] * 1.5) & (df['volume'] > df['volume_3'] * 1.5) & (df['volume'] > df['volume_2'] * 1.5) & (df['volume'] > df['volume_1'] * 1.5)]
    info(f'当日成交量比前五日每一天的成交量均大 1.5 倍{len(df_selected)}', 'debug')
    info(df_selected, 'debug')
    info(f'当日成交量比前五日每一天的成交量均大 1.5 倍{len(df_selected)}', 'info')
    info(df_selected['code'].tolist(), 'info')    
    return df_selected['code'].tolist()

#new 5、股票当日 2:30 分之后成交量放大，2:30 分之后每分钟平均成交量为 60 分钟级别最高，即平均每分钟成交量这个值，2:30 分之后的数值高于 9:30~10:30、10:30~11:30、1:00~2:00 任何时段的值。
def getStocksByVolume3(arrStocks, klines = 3):
    #1、取得今日小时交易信息
    df = get_history(klines, '60m', 'volume', arrStocks, fq=None, include=True)
    info(f'5.1 取得今日小时交易信息{len(df)}', 'debug')  
    info(df, 'debug')
    last_time = df.index[-1]
    df['lines'] = 60
    df.loc[df.index == last_time, 'lines'] = g.hand_times % 60
    df['ave_volume'] = df['volume'] / df['lines']
    #设置avg_volume_last_1为上一行的ave_volume值
    df['avg_volume_last_1'] = df['ave_volume'].shift(1)
    #设置avg_volume_last_2为上上一行的ave_volume值
    df['avg_volume_last_2'] = df['ave_volume'].shift(2)

    info(df, 'debug')
    #2、取得最后一根K的时间
    info(f'5.2 取得最后一根K的时间{last_time}', 'debug')
    #3、取得每支股票中四条K线中volume中最大的，且时间为last_time的股票
    df_selected = df[(df['ave_volume'] == df.groupby('code')['ave_volume'].transform(max))]
    info(f'5.3 取得每支股票中四条K线中volume中最大的，且时间为last_time的股票{len(df_selected)}', 'debug')
    info(df_selected, 'debug')
    info(f'5.3 取得每支股票中四条K线中volume中最大的，且时间为last_time的股票{len(df_selected)}', 'info')
    info(df_selected['code'].tolist(), 'info')
    #4、筛选出index为last_time的股票，并且ave_volume大于avg_volume_last_1和avg_volume_last_2的1.3倍
    df_selected = df_selected[(df_selected.index == last_time) & (df_selected['ave_volume'] > df_selected['avg_volume_last_1'] *1.3) & (df_selected['ave_volume'] > df_selected['avg_volume_last_2'] *1.3)] 
    info(f'5.4 筛选出index为last_time的股票，并且ave_volume大于avg_volume_last_1和avg_volume_last_2的1.3倍{len(df_selected)}', 'debug')
    info(df_selected, 'debug')
    info(f'5.4 筛选出index为last_time的股票，并且ave_volume大于avg_volume_last_1和avg_volume_last_2的1.3倍{len(df_selected)}', 'info')
    info(df_selected['code'].tolist(), 'info')    
    return df_selected['code'].tolist()

    


#5 股票当日 2:30 分之后成交量放大，2:30 分之后每分钟平均成交量高于全天每分钟成交量。
def getStocksByVolume2(arrStocks, klines = 0):
    #1、取得今日分钟交易信息
    df = get_history(klines, '1m', 'volume', arrStocks, fq=None, include=True)
    info(f'取得今日分钟交易信息{len(df)}', 'debug')  
    info(df, 'debug')
    #2、循环每（所有K线 - 20）行数据，计算平均成交量
    df['mean'] = df.groupby('code')['volume'].transform(lambda x: x.rolling(window=klines-20).mean())
    info(f'循环每（所有K线 - 20）行数据，计算平均成交量{len(df)}', 'debug')  
    info(df, 'debug')
    #3、循环每20行数据，计算平均交易量
    df['mean20'] = df.groupby('code')['volume'].transform(lambda x: x.rolling(window=20).mean())
    info(f'循环每20行数据，计算平均交易量{len(df)}', 'debug')  
    info(df, 'debug')
    # 获取最新时间和20个时间点前的时间
    unique_times = df.index.unique().sort_values()
    if len(unique_times) < 21:
        info('数据不足21个时间点', 'debug')
        return []
    avg_date_before = unique_times[-21]  # 21个时间点前的时间
    avg_date_after = unique_times[-1]  # 最新时间
    info(f'两个对比时间分别是：%s:%s' % (avg_date_before, avg_date_after), 'debug')
    # 获取两个时间点的数据
    df_date_before = df[df.index == avg_date_before]
    df_date_after = df[df.index == avg_date_after]
    info(f'时间点:{avg_date_before}的数据,{len(df_date_before)}', 'debug')
    info(df_date_before, 'debug')  
    info(f'时间点:{avg_date_after}的数据,{len(df_date_after)}', 'debug')
    info(df_date_after, 'debug')

    # 合并两个时间点的数据，只保留同时存在于两个时间点的股票
    merged = df_date_after.merge(df_date_before, on='code', suffixes=('_date_after', '_date_before'))
    info(f'合并两个时间点的数据，只保留同时存在于两个时间点的股票{len(merged)}', 'debug')  
    info(merged, 'debug')
    selected = merged[(merged['mean20_date_after'] > merged['mean_date_before'])]
    info(f'满足条件的股票{len(selected)}', 'debug')  
    info(selected, 'debug')
    info(f'满足条件的股票{len(selected)}', 'info')  
    info(selected['code'].tolist(), 'info')    
    return selected['code'].tolist()  
    
#6、主力资金当日净流入金额为大于 0；且主力资金 5 日净流入的和大于 0。
def getStocksByMainFunds(arrStocks, context, data):
    arr = []
    for stock in arrStocks:
        arr.append(stock[:6])
    sx.set_token("fff6b1bd0c1fd3ad9577dcafe77c54a4185ade23351c13356de7ebaa")
    pro=sx.get_api(env="prd")
    #1、取得当前年月日
    date = context.blotter.current_dt.strftime("%Y%m%d")
    df = pro.moneyflow(trade_date=date)
    info(f'取得当前年月日{date}', 'debug')  
    info(df, 'debug')
    #判断df是否为空，如果为空，则直接返回arrStocks
    if df.empty:
        info(f'判断df是否为空，如果为空，则直接返回arrStocks{len(arrStocks)}', 'debug')  
        info(arrStocks, 'debug')
        return arrStocks
    #2、将df中的ts_code的前6位提取出来，存入到shot_code中
    df['shot_code'] = df['ts_code'].str[:6]
    info(f'将df中的ts_code的前6位提取出来，存入到shot_code中{len(df)}', 'debug')  
    info(df, 'debug')
    #3、从df中筛选出ts_code在arr中的数据
    df_selected = df[df['shot_code'].isin(arr)]
    info(f'从df中筛选出ts_code在arr中的数据{len(df_selected)}', 'debug')  
    info(df_selected, 'debug')
    #4、筛选出df_selected中(大单买入金额 + 特大单买入金额) > (大单卖出金额 + 特大单卖出金额)
    df_selected = df_selected[df_selected['buy_lg_amount'] + df_selected['buy_elg_amount'] > df_selected['sell_lg_amount'] + df_selected['sell_elg_amount']]
    info(f'筛选出df_selected中(大单买入金额 + 特大单买入金额) > (大单卖出金额 + 特大单卖出金额){len(df_selected)}', 'debug')  
    info(df_selected, 'debug')
    #4、返回df_selected中的ts_code
    return df_selected['ts_code'].tolist()

    

#以上四条全部满足，则复合买入条件，当日 2 点 55 分之后挂单买入。
def getBuyStocks(arrBuyStocks, context):
    nBuyCount = len(arrBuyStocks)
    if nBuyCount == 0:
        return
    
    # 获取当前的现金
    cash = context.portfolio.cash
    info(f'当前现金{cash}', 'debug')
    if cash < g.nBuyMoney:
        nMoney = cash
    else:
        nMoney = g.nBuyMoney
    info(f'今日买入金额{nMoney}', 'debug')
    
    nMoney = nMoney / nBuyCount
    for stock, price in arrBuyStocks.items():
        if stock not in g.buyedStocks:
            #计算可买入数量
            nStockBuyAmount = int(nMoney / price / 100) * 100
            order(stock, nStockBuyAmount, price)
            # order_value(stock, nMoney)
            g.buyedStocks[stock] = price
            g.mailBuyStocks[stock] = {
                'price': price,
                'amount': nStockBuyAmount
            }
            info(f"买入股票%s,数量%s, 金额%s元，买入价%s"%(stock,nStockBuyAmount, nMoney, price), 'info')
    
    msg = f"{str(g.deal_hand_time)}发送邮件：{str(g.mailBuyStocks)}"
    send_email('54229670@qq.com', ['54229670@qq.com', '290291989@qq.com', '68043432@qq.com'], 'zfethvvdbmcbbjaa', info=msg)

def info(info, type = 'info'):
    # if g.bDebug:
    #     if type == 'info':
    #         print(info)
    #     elif type == 'debug':
    #         print(info)
    #     elif type == 'error':
    #         print(info)
    # else:
    if type == 'info':
        log.info(info)
    elif type == 'debug':
        log.debug(info)
    elif type == 'error':
        log.error(info)


#过滤ST、停牌、退市的股票
def filter_ST(arrStocks):
    #获取股票的状态ST、停牌、退市
    st_status = get_stock_status(arrStocks, 'ST')
    halt_status = get_stock_status(arrStocks, 'HALT')
    delisting_status = get_stock_status(arrStocks, 'DELISTING')
    #将三种状态的股票剔除当日的股票池
    for stock in arrStocks.copy():
        if st_status[stock] or halt_status[stock] or delisting_status[stock]:
            arrStocks.remove(stock)
    return arrStocks

#剔除沪深非主板股票
def filter_not_mainboard(arrStocks):
    for stock in arrStocks.copy():
        if not ((stock.startswith('60') and stock.endswith('.SS')) or ((stock.startswith('00') and stock.endswith('.SZ')))):
            arrStocks.remove(stock)
    return arrStocks

#tick_data每3秒运行一次，在tick_data内，改成每5分钟运行一次
def tick_data(context,data):
    #取得股票换手率
    if g.hand_times > g.deal_hand_times and hasattr(g, 'arrStocks'):
        g.arrTurnoverStocks = getStocksByTurnoverRate(g.arrStocks, context=context, data=data)


def handle_data(context, data):
    if not hasattr(g, 'bDebug'):
        g.bDebug = False
    if not hasattr(g, 'hand_times'):
        g.hand_times = 0
    if not hasattr(g, 'buyedStocks'):
        g.buyedStocks = {}
    if not hasattr(g, 'mailBuyStocks'):
        g.mailBuyStocks = {}
    if not hasattr(g, 'arrStocks'):
        g.arrStocks = []
    if not hasattr(g, 'toGetTurnover_ratio'):
        g.toGetTurnover_ratio = {}
    g.hand_times = get_today_minute_kline_index()
    if not hasattr(g, 'deal_hand_times'):
        g.deal_hand_times = 225


    info(f"--------------Run handle_data, hand_times:{g.hand_times}----------------", 'info') 
    # #1、开放交易后，先卖掉
    # if g.hand_times == g.sell_hand_times:
    #     for stock in get_position_list(context):
    #         info(f"买出股票%s,卖出价：%s"%(stock,data[stock].price), 'info') 
    #         order_target(stock, 0)

    #2、2：30后，取得可选股票
    if g.hand_times < g.deal_hand_times:
        return
    elif g.hand_times == g.deal_hand_times: 
        arrStocks = []
        #step 0、取得主板所有股票
        arrStocks = getStocksByBoard()
        #step 1、筛选位于主板的，市值 30-400 亿的股票。只筛选出位于沪深主板的股票，剔除ST 类股票，剔除主板以外科创板、创业板等所有非主板股票。
        arrStocks = getStocksByTotalValue(arrStocks)
        #step 2、筛选当日股票涨幅不低于+3%，且上影线长度不大于 K 线实体的 1/5后
        arrStocks = getStocksByPxChangeRateAndShadowLine(arrStocks)
        #4、从交易日 2:30 分之后开始选股，当日成交量大于前五日平均成交量的 1.5倍。且当日成交量比前五日每一天的成交量均大 1.5 倍。
        arrStocks = getStocksByVolume(arrStocks)
        #5 股票当日 2:30 分之后成交量放大，2:30 分之后每分钟平均成交量高于全天每分钟成交量。
        # arrStocks = getStocksByVolume2(arrStocks, g.hand_times)
        #判断arrStocks是否为空，如果为空，返回
        if len(arrStocks) > 0:
            arrStocks = getStocksByVolume3(arrStocks)
        else:
            info(f'arrStocks为空，未执行：getStocksByVolume3，5 股票当日 2:30 分之后成交量放大，2:30 分之后每分钟平均成交量高于全天每分钟成交量。', 'info')
        #6、主力资金当日净流入金额为大于 0；且主力资金 5 日净流入的和大于 0。
        # arrStocks = getStocksByMainFunds(arrStocks, context, data)
        g.arrStocks = arrStocks
        g.bFinishedCalc = True

    elif g.hand_times > g.deal_hand_times:
        info(f'++++++++++取换手率股票{g.bDebug}++++++++++++', 'info')
        if g.bDebug:
            g.arrTurnoverStocks = getStocksByTurnoverRate(g.arrStocks, context=context, data=data)
        # for stock in g.arrTurnoverStocks:
        #     arrStocks.append(stock)

    # getBuyStocks(arrStocks, data)

#获取当前分钟K线索引，用于获取分钟K线数据
def get_today_minute_kline_index(tz=pytz.timezone('Asia/Shanghai')):
    return get_current_kline_count()

# 生成昨日持仓股票列表
def get_position_list(context):
    return [
        position.sid
        for position in context.portfolio.positions.values()
        if position.amount != 0
    ]


def getBuyPrice(stock):
    snapshot_data = get_snapshot(stock)
    # info("--------------Run getBuyPrice----------------" + str(snapshot_data), 'info')
    current_price = snapshot_data[stock]['offer_grp'][5][0]
    current_price_2 = snapshot_data[stock]['offer_grp'][3][0]
    return current_price, current_price_2

def getSellPrice(stock):
    snapshot_data = get_snapshot(stock)
    # info("--------------Run getSellPrice----------------" + str(snapshot_data), 'info')
    current_price = snapshot_data[stock]['bid_grp'][5][0]
    current_price_2 = snapshot_data[stock]['bid_grp'][3][0]
    return current_price, current_price_2
    