"""
卖出时间：上午 9:30~下午 2:30；
9% ＜ 股价涨幅:执行全仓卖出
1%＜股价涨幅≤9%: 
        1、生成持股线，持股线=当日最大涨幅*60%；
        2、若股价涨幅≥持股线，继续持股，如果直至2:30 分，股价依然处于此阶段，执行全仓卖出；
        3、若股价涨幅＜持股线，执行全仓卖出。（若股价回到-2%＜股价涨跌幅≤1%，则按该区间策略执行）
-2%＜股价涨跌幅≤1%: 持仓，如果直至 2:30 分，股价依然处于此阶段，执行全仓卖出。
-3%＜股价跌幅≤-2%：观察 10 分钟，10 分钟后如果股价跌幅依然小于-2%，执行全仓卖出；如果股价跌幅大于-2%，继续持股。
股价跌幅≤-3%:执行全仓卖出。
"""
import ast
import pandas as pd
from datetime import datetime
import pytz

def initialize(context):
    info("--------------Run initialize----------------", 'info') 
    pd.option_context('display.max_rows', 50)
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', 1000)  # 增加列展示宽度
    pass

# 盘前处理
def before_trading_start(context, data):
    g.hand_times = 0    #交易次数
    g.hold_position = []  #持仓股票
    g.watch_stocks = {} #观察股票
    g.get_position = False  #是否获取持仓
    pass

def tick_data(context,data):
    info("--------------Run tick_data----------------" + str(g.hand_times), 'info') 
    if not g.get_position:
        g.get_position = True
        g.hold_position = get_position_list(context)
    info("--------------Run tick_data----------------" + str(g.hold_position), 'info') 
    #循环判断持股股票当前涨跌幅
    for stock in g.hold_position:
        #high_px：最高价；
        #last_px：最新成交价；
        #open_px：今开盘价；
        #preclose_px：昨收价；
        #px_change_rate: 涨跌幅；
        current_price = ast.literal_eval(data[stock]['tick']['bid_grp'][0])[1][0]
        high_px = float(data[stock]['tick']['high_px'])
        last_px = float(data[stock]['tick']['last_px'])
        open_px = float(data[stock]['tick']['open_px'])
        preclose_px = float(data[stock]['tick']['preclose_px'])
        px_change_rate = (current_price - preclose_px) / preclose_px * 100
        sys_px_change_rate = float(data[stock]['tick']['px_change_rate'])
        turnover_ratio = float(data[stock]['tick']['turnover_ratio'])


        info("--------------Run tick_data----------------" + str(stock) + "最高价：" + str(high_px) + ",当前价格：" + str(current_price) + ",最新成交价：" + str(last_px) + ",昨收价：" + str(preclose_px) + ",今开盘价：" + str(open_px) + ",换手率：" + str(turnover_ratio) + ",当前涨跌幅:" + str(px_change_rate) + ", 系统涨跌幅：" + str(sys_px_change_rate), 'info')
        if px_change_rate > 9:
            removeWatchStock(stock)
            info("--------------Run tick_data----------------" + str(stock) + "当前涨跌幅大于9%，执行全仓卖出", 'info')
            removeHoldStock(stock)
            order_target(stock, 0)
        elif px_change_rate > 1 and px_change_rate <= 9:
            removeWatchStock(stock)
            #1、生成持股线，持股线=当日最大涨幅*60%；
            hold_line = preclose_px + (high_px - preclose_px) * 0.6
            info("--------------Run tick_data----------------" + str(stock) + "hold_line:" + str(hold_line), 'info')
            #2、若股价涨幅≥持股线，继续持股，如果直至2:30 分，股价依然处于此阶段，执行全仓卖出；
            if current_price < hold_line:
                info("--------------Run tick_data----------------" + str(stock) + "当前价格小于持股线，执行全仓卖出" + "，当前持股线：" + str(hold_line), 'info')
                removeHoldStock(stock)
                order_target(stock, 0)
            if g.hand_times > 210:
                info("--------------Run tick_data----------------" + str(stock) + "2:30后，仍处于1%＜股价涨幅≤9%阶段，执行全仓卖出", 'info')
                removeHoldStock(stock)
                order_target(stock, 0)
        elif px_change_rate > -2 and px_change_rate <= 1:
            #持仓，如果直至 2:30 分，股价依然处于此阶段，执行全仓卖出。
            removeWatchStock(stock)
            if g.hand_times > 210:
                info("--------------Run tick_data----------------" + str(stock) + "2:30后，仍处于-2%＜股价涨幅≤1%阶段，执行全仓卖出", 'info')
                removeHoldStock(stock)
                order_target(stock, 0)
            pass
        elif px_change_rate > -3 and px_change_rate <= -2:
            #观察 10 分钟，10 分钟后如果股价跌幅依然小于-2%，执行全仓卖出；如果股价跌幅大于-2%，继续持股。
            if stock not in g.watch_stocks:
                g.watch_stocks[stock] = g.hand_times
            elif g.hand_times - g.watch_stocks[stock] > 10:
                info("--------------Run tick_data----------------" + str(stock) + "观察10分钟后，股价跌幅依然小于-2%，执行全仓卖出", 'info')
                removeHoldStock(stock)
                order_target(stock, 0)    
                removeWatchStock(stock)
            else:
                info("--------------Run tick_data----------------" + str(stock) + "观察10分钟后，股价跌幅依然小于-2%，继续持股", 'info')
        
        else:
            removeWatchStock(stock)
            info("--------------Run tick_data----------------" + str(stock) + "当前涨跌幅小于-3%，执行全仓卖出", 'info')
            removeHoldStock(stock)
            order_target(stock, 0)

def removeWatchStock(stock):
    if stock in g.watch_stocks:
        g.watch_stocks.pop(stock)
def removeHoldStock(stock):
    if stock in g.hold_position:
        g.hold_position.remove(stock)

def handle_data(context, data):
    g.hand_times = get_current_kline_count()
    info("--------------Run handle_data----------------" + str(g.hand_times), 'info') 
    if not g.get_position:
        g.get_position = True
        g.hold_position = get_position_list(context)


# 生成昨日持仓股票列表
def get_position_list(context):
    return  ['603075.SS', '001236.SZ', '002036.SZ', '002158.SZ', '002201.SZ', '002709.SZ']
    # return [
    #     position.sid
    #     for position in context.portfolio.positions.values()
    #     if position.amount != 0
    # ]

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