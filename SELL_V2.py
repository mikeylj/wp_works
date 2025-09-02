"""
2.1 策略一：上午 9:30~9:40：
    股价涨跌幅≥2%：生成持股线，持股线=当日最大涨幅*90%；
                若股价涨幅≥持股线，继续持股；
                若股价涨幅＜持股线，执行全仓卖出。
                （若股价回到 股价涨跌幅＜2% 区间，依然按此区间策略，跌破持股线执行卖出）
                举例：股票最高涨到 2%，此时持股线=1.8%，股
                价如果跌到 1.8%，依然按此区间策略，跌破持股线全仓卖出
    -5%≤股价涨跌幅＜2%：继续持股
    股价跌幅＜-5%：执行全仓卖出
2.2 策略二：上午 9:40~下午 2:30
    股价跌幅≤-3%：执行全仓卖出。
    -3%＜股价跌幅≤-2%：观察 10 分钟，10 分钟后，如果股价跌幅依然小于-2%，执行全仓卖出；
                    如果股价跌幅大于-2%，继续持股。
                    （观察期间，一旦涨跌幅大于-2%，观察期结束；如果涨跌幅再次小于-2%，重新计时观察 10分钟）
    -2%＜股价涨跌幅≤1%:持仓，如果直至 2:30 分，股价依然处于此阶段，执行全仓卖出。
    1%＜股价涨幅≤3%:生成持股线，持股线=当日最大涨幅*60%；若股价涨幅≥持股线，继续持股，如果直至2:30 分，股价依然处于此阶段，执行全仓卖出；
                  若股价涨幅＜持股线，执行全仓卖出。（若股价回到 -2%＜股价涨跌幅≤1% 区间，则按该区间策略执行，继续持仓）
                  举例：股票最高涨到 1.5%，此时持股线=0.9%，股价如果跌到 0.9%，则按（区间-2%＜股价涨跌幅≤1%）策略，继续持仓
    3%＜股价涨幅≤6%:生成持股线，持股线=当日最大涨幅*70%；若股价涨幅≥持股线，继续持股，如果直至 2:30 分，股价依然处于此阶段，执行全仓卖出；
                   若股价涨幅＜持股线，执行全仓卖出。（若股价回到 股价涨跌幅≤3% 区间，依然按此区间策略，跌破持股线执行卖出）
                    举例：股票最高涨到 4%，此时持股线=2.8%，股价如果跌到 2.8%，依然按此区间策略，跌破持股线全仓卖出。
    6%＜股价涨幅≤9%:生成持股线，持股线=当日最大涨幅*80%；若股价涨幅≥持股线，继续持股，如果直至2:30 分，股价依然处于此阶段，执行全仓卖出；
                    若股价涨幅＜持股线，执行全仓卖出。（若股价回到 股价涨跌幅≤6% 区间，依然按此区间策略，跌破持股线执行卖出）
                    举例：股票最高涨到 7%，此时持股线=5.6%，股价如果跌到 5.6%，依然按此区间策略，跌破持股线全仓卖出。
    9%＜股价涨幅:全仓卖出
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
    g.get_position = False  #是否获取持仓
    g.hold_position = []  #持仓股票
    g.watch_stocks = {} #观察股票

def getBuyPrice(stock):
    snapshot_data = get_snapshot(g.security)
    current_price = ast.literal_eval(snapshot_data[stock]['bid_grp'][0])[2][0]
    current_price_2 = ast.literal_eval(snapshot_data[stock]['bid_grp'][0])[3][0]
    return current_price, current_price_2

def getSellPrice(stock):
    snapshot_data = get_snapshot(g.security)
    current_price = ast.literal_eval(snapshot_data[stock]['offer_grp'][0])[2][0]
    current_price_2 = ast.literal_eval(snapshot_data[stock]['offer_grp'][0])[3][0]
    return current_price, current_price_2


def handle_data(context, data):
    g.hand_times = get_current_kline_count()
    info("--------------Run handle_data----------------" + str(g.hand_times), 'info') 
    if not g.get_position:
        g.get_position = True
        g.hold_position = get_position_list(context)

def tick_data(context,data):
    info("--------------Run tick_data----------------" + str(g.hand_times), 'info') 
    if not g.get_position:
        g.get_position = True
        g.hold_position = get_position_list(context)
    info("--------------Run tick_data----------------" + str(g.hold_position), 'info') 
    #策略一：上午 9:30~9:40：
    
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
        if g.hand_times <= 10:
            if px_change_rate >= 1.8:
                #生成持股线，持股线=当日最大涨幅*90%；
                hold_line = preclose_px + (high_px - preclose_px) * 0.9
                info("--------------Run tick_data----------------" + str(stock) + "hold_line:" + str(hold_line), 'info')
                if current_price < hold_line:
                    info("--------------Run tick_data----------------" + str(stock) + "涨跌幅大于1.8%,当前价格小于持股线，执行全仓卖出" + "，当前持股线：" + str(hold_line), 'info')
                    removeHoldStock(stock)
                    order_target(stock, 0)
            elif px_change_rate < 1.8 and px_change_rate >= -5:
                info("--------------Run tick_data----------------" + str(stock) + "涨跌幅小于1.8%，大于-5%,继续持股", 'info')
            else:
                info("--------------Run tick_data----------------" + str(stock) + "涨跌幅小于-5%，执行全仓卖出", 'info')
                removeHoldStock(stock)
                order_target(stock, 0)
        else:
            #股价跌幅≤-3%：执行全仓卖出。
            if px_change_rate <= -3:
                info("--------------Run tick_data----------------" + str(stock) + "当前涨跌幅小于-3%，执行全仓卖出", 'info')
                removeWatchStock(stock)
                removeHoldStock(stock)
                order_target(stock, 0)
            # -3%＜股价跌幅≤-2%：观察 10 分钟，10 分钟后，如果股价跌幅依然小于-2%，执行全仓卖出；
            #         如果股价跌幅大于-2%，继续持股。
            #         （观察期间，一旦涨跌幅大于-2%，观察期结束；如果涨跌幅再次小于-2%，重新计时观察 10分钟）
            elif px_change_rate > -3 and px_change_rate <= -2 :
                info("--------------Run tick_data----------------" + str(stock) + "当前涨跌幅小于-2%，大于-3%", 'info')
                if stock not in g.watch_stocks:
                    g.watch_stocks[stock] = g.hand_times
                    info("--------------Run tick_data----------------" + str(stock) + "开始观察10分钟", 'info')
                elif g.hand_times - g.watch_stocks[stock] >= 10:
                    info("--------------Run tick_data----------------" + str(stock) + "观察10分钟后，股价跌幅依然小于-2%，大于-3%，执行全仓卖出", 'info')
                    removeHoldStock(stock)
                    removeWatchStock(stock)
                    order_target(stock, 0)    
                else:
                    info("--------------Run tick_data----------------" + str(stock) + "观察10分钟内，股价跌幅依然小于-2%，大于-3%，继续持股、观察", 'info')
            #-2%＜股价涨跌幅≤1%,持仓，如果直至 2:30 分，股价依然处于此阶段，执行全仓卖出。
            elif px_change_rate > -2 and px_change_rate <= 1 :
                info("--------------Run tick_data----------------" + str(stock) + "当前涨跌幅大于-2%，小于1%", 'info')
                removeWatchStock(stock)
                if g.hand_times > 210:
                    info("--------------Run tick_data----------------" + str(stock) + "直至2:30分，执行全仓卖出", 'info')
                    removeHoldStock(stock)
                    order_target(stock, 0)
            #1%＜股价涨幅≤3%:生成持股线，持股线=当日最大涨幅*60%；若股价涨幅≥持股线，继续持股，如果直至2:30 分，股价依然处于此阶段，执行全仓卖出；
            #若股价涨幅＜持股线，执行全仓卖出。（若股价回到 -2%＜股价涨跌幅≤1% 区间，则按该区间策略执行，继续持仓）
            elif px_change_rate > 1 and px_change_rate <= 3:
                info("--------------Run tick_data----------------" + str(stock) + "当前涨跌幅大于1%，小于3%", 'info')
                removeWatchStock(stock)
                hold_line = preclose_px + (high_px - preclose_px) * 0.6
                info("--------------Run tick_data----------------" + str(stock) + "hold_line:" + str(hold_line), 'info')
                if current_price < hold_line:
                    info("--------------Run tick_data----------------" + str(stock) + "当前价格小于持股线，执行全仓卖出" + "，当前持股线：" + str(hold_line), 'info')
                    removeHoldStock(stock)
                    order_target(stock, 0)
                elif g.hand_times > 210:
                    info("--------------Run tick_data----------------" + str(stock) + "直至2:30分，执行全仓卖出" + "，当前持股线：" + str(hold_line), 'info')
                    removeHoldStock(stock)
                    order_target(stock, 0)
            #3%＜股价涨幅≤6%:生成持股线，持股线=当日最大涨幅*70%；若股价涨幅≥持股线，继续持股，如果直至 2:30 分，股价依然处于此阶段，执行全仓卖出；
            #       若股价涨幅＜持股线，执行全仓卖出。（若股价回到 股价涨跌幅≤3% 区间，依然按此区间策略，跌破持股线执行卖出）
            #        举例：股票最高涨到 4%，此时持股线=2.8%，股价如果跌到 2.8%，依然按此区间策略，跌破持股线全仓卖出。
            elif px_change_rate > 3 and px_change_rate <= 6:
                info("--------------Run tick_data----------------" + str(stock) + "当前涨跌幅大于3%，小于6%", 'info')
                removeWatchStock(stock)
                hold_line = preclose_px + (high_px - preclose_px) * 0.7
                info("--------------Run tick_data----------------" + str(stock) + "hold_line:" + str(hold_line), 'info')
                if current_price < hold_line:
                    info("--------------Run tick_data----------------" + str(stock) + "当前价格小于持股线，执行全仓卖出" + "，当前持股线：" + str(hold_line), 'info')
                    removeHoldStock(stock)
                    order_target(stock, 0)
                elif g.hand_times > 210:
                    info("--------------Run tick_data----------------" + str(stock) + "直至2:30分，执行全仓卖出" + "，当前持股线：" + str(hold_line), 'info')
                    removeHoldStock(stock)
                    order_target(stock, 0)
            
            #6%＜股价涨幅≤9%:生成持股线，持股线=当日最大涨幅*80%；若股价涨幅≥持股线，继续持股，如果直至2:30 分，股价依然处于此阶段，执行全仓卖出；
            #        若股价涨幅＜持股线，执行全仓卖出。（若股价回到 股价涨跌幅≤6% 区间，依然按此区间策略，跌破持股线执行卖出）
            #        举例：股票最高涨到 7%，此时持股线=5.6%，股价如果跌到 5.6%，依然按此区间策略，跌破持股线全仓卖出。
            elif px_change_rate > 6 and px_change_rate <= 9:
                info("--------------Run tick_data----------------" + str(stock) + "当前涨跌幅大于6%，小于9%", 'info')
                removeWatchStock(stock)
                hold_line = preclose_px + (high_px - preclose_px) * 0.8
                info("--------------Run tick_data----------------" + str(stock) + "hold_line:" + str(hold_line), 'info')
                if current_price < hold_line:
                    info("--------------Run tick_data----------------" + str(stock) + "当前价格小于持股线，执行全仓卖出" + "，当前持股线：" + str(hold_line), 'info')
                    removeHoldStock(stock)
                    order_target(stock, 0)
                elif g.hand_times > 210:
                    info("--------------Run tick_data----------------" + str(stock) + "直至2:30分，执行全仓卖出" + "，当前持股线：" + str(hold_line), 'info')
                    removeHoldStock(stock)
                    order_target(stock, 0)
            
            #9%＜股价涨幅:全仓卖出
            elif px_change_rate > 9:
                info("--------------Run tick_data----------------" + str(stock) + "当前涨跌幅大于9%，执行全仓卖出", 'info')
                removeWatchStock(stock)
                removeHoldStock(stock)
                order_target(stock, 0)

def removeWatchStock(stock):
    if stock in g.watch_stocks:
        g.watch_stocks.pop(stock)
def removeHoldStock(stock):
    if stock in g.hold_position:
        g.hold_position.remove(stock)

# 生成昨日持仓股票列表
def get_position_list(context):
    return [
        position.sid
        for position in context.portfolio.positions.values()
        if position.amount != 0
    ]

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