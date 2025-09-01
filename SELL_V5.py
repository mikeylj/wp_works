#如果持有的某支股票累计跌幅超过 5%，则这支股票全仓卖出。
#每支股票卖出时间为当日下午 2：30 分
#每支股票买入后持有 7 个交易日；例如：股票在 D1 那天的下午 2 点 55 分买入，则在 D8 那天的下午 2 点 30 分卖出。
import pickle
from collections import defaultdict
import pandas as pd
from datetime import datetime, timedelta
import ast

def initialize(context):
    info("--------------Run initialize----------------", 'info') 
    pd.option_context('display.max_rows', 50)
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', 1000)  # 增加列展示宽度
    g.my_orders = {}  #记录订单信息
    g.notebook_path = get_research_path()
        
def before_trading_start(context, data):
    g.today_sell_stocks = {}    #记录今天卖出的股票
    g.hand_times = 0    #交易次数
    info("--------------Run before_trading_start----------------" + str(g.hand_times), 'info') 
    #记录今天买进、卖出的股票
    readOrdersFromPkl()
    # saveOrders()
 
def tick_data(context,data):
    # 获取当前持仓
    info("--------------Run tick_data----------------" + str(g.hand_times), 'info') 
    if g.hand_times > 0 and g.hand_times < 237:
        sellStocksByDropRate(data)
    

def handle_data(context, data):
    g.hand_times = get_current_kline_count()
    info("--------------Run handle_data----------------" + str(g.hand_times), 'info') 
    # 
    # saveOrders()
    if g.hand_times == 210:
        info("--------------Run handle_data----------------" + "g.hand_times:" + str(g.hand_times) + "运行在 D8 那天的下午 2 点 30 分卖出", 'info') 
        sellStocksByHoldDays()

def after_trading_end(context, data):
    info("--------------Run after_trading_end----------------", 'info') 
    saveOrders()

"""
#每支股票卖出时间为当日下午 2：30 分
#每支股票买入后持有 7 个交易日；例如：股票在 D1 那天的下午 2 点 55 分买入，则在 D8 那天的下午 2 点 30 分卖出。
1.判断当前日期是否为D8，即len(g.my_orders) >= 7
2.如果为D8，取得g.my_orders中第一个值，遍历first_order，取得持仓的股票代码,并卖出
3.从g.my_orders中删除这支股票的下单信息。
"""
def sellStocksByHoldDays():
    if len(g.my_orders) >= 7:
        #取g.my_orders中第一个值
        first_day = next(iter(g.my_orders))
        first_order = g.my_orders[first_day]
        info("--------------sellStocksByHoldDays----------------" + "first_day:" + str(first_day) + "，D8的下单信息:" + str(first_order), 'debug')
        #遍历first_order，取得持仓的股票代码,并卖出
        for order in first_order:
            if int(order['entrust_bs']) == 1 and int(order['status']) == 8:
                stock = order['symbol']
                amount  = order['filled_amount']
                hold_price = order['price']
                sell_price_2, sell_price_3 = getSellPrice(stock)
                #卖出相应的股票
                info("--------------sellStocksByHoldDays----------------日期：" + first_day + "股票:" + str(stock) + "成本价：" + str(hold_price) + "，卖出价：" + str(sell_price_2) + "，卖出量：" + str(amount), 'info')
                toSellStocks(stock, int(amount), sell_price_2)
        #删除g.my_orders中第一个值
        del g.my_orders[first_day]
    else:
        info("--------------sellStocksByHoldDays----------------len(g.my_orders) < 7", 'info')

def toSellStocks(stock, amount, price):
    order(stock, -int(amount), price)
    if stock in g.today_sell_stocks:
        g.today_sell_stocks[stock] = g.today_sell_stocks[stock] + int(amount)
    else:
        g.today_sell_stocks[stock] = int(amount)

"""
如果持有的某支股票累计跌幅超过 5%，则这支股票全仓卖出。
1.循环从g.my_orders中取得下单信息，与当前价对比，如果跌幅超过5%，则这支股票全仓卖出。
2.卖出后，从g.my_orders中删除这支股票的下单信息。
"""
def sellStocksByDropRate(data):
    #循环遍历买入的订单，判断票累计跌幅超过 5%，则这支股票全仓卖出
    temp_my_orders = {}
    info("--------------sellStocksByDropRate----------------定单库（g.my_orders)：" + str(g.my_orders), 'debug')
    for date, orders in g.my_orders.items():
        info("--------------sellStocksByDropRate----------------日期定单（orders）：" + "date:" + str(date) + "，orders:" + str(orders), 'debug')
        temp_orders = []
        for buy_order in orders.copy():
            info("--------------sellStocksByDropRate----------------买入定单（buy_order）：" + str(buy_order), 'debug')
            #累计跌幅超过 5%，则这支股票全仓卖出
            stock = buy_order['symbol']
            hold_price = buy_order['price']
            current_price = ast.literal_eval(data[stock]['tick']['bid_grp'][0])[1][0]
            #计算跌幅
            drop_rate = (current_price - hold_price) / hold_price
            info("--------------sellStocksByDropRate----------------" + str(stock) + "，成本价：" + str(hold_price) + "，当前价：" + str(current_price) + "，涨跌幅：" + str(drop_rate), 'debug')
            if drop_rate < -0.05:
                sell_price_2, sell_price_3 = getSellPrice(stock)
                info("--------------sellStocksByDropRate----------------日期：" + date + "，股票：" + str(stock) + "，成本价：" + str(hold_price) + "，当前价：" + str(current_price) + "，涨跌幅：" + str(drop_rate) + "，卖出价：" + str(sell_price_2) + "，卖出量：" + str(buy_order['filled_amount']) + '，累计跌幅超过 5%，则这支股票全仓卖出', 'info')    
                toSellStocks(stock, int(buy_order['filled_amount']), sell_price_2) 
            else:
                temp_orders.append(buy_order)
        temp_my_orders[date] = temp_orders
    g.my_orders = temp_my_orders
    
"""
1. 读取文件中的持仓信息:my_orders.pkl，存储在g.my_orders中
2. 从g.my_orders中取得持仓的股票代码，存储在hold_stocks的字典中，下标是stock,值是amount
3. 与当前持仓比较，如果持仓股票不在hold_stocks中或小于hold_stocks中的amount，刚报警，如果大于hold_stocks中的amount，更新hold_stocks[昨天]中的amount
"""
def readOrdersFromPkl():
    try:
        with open(g.notebook_path+'my_orders.pkl','rb') as f:
            g.my_orders = pickle.load(f)
    #定义空的全局字典变量
    except:
        g.my_orders = defaultdict(list)
    info("--------------readOrdersFromPkl----------------定单库（g.my_orders)：" + str(g.my_orders), 'debug') 
    
    hold_stocks = {}
    last_date = ''  #记录g.my_orders中的最后一天
    #循环读取g.my_orders，取得持仓的股票代码
    for date, orders in g.my_orders.items():
        last_date = date
        for buy_order in orders:
            stock = buy_order['symbol']
            amount = int(buy_order['filled_amount'])
            if stock not in hold_stocks:
                hold_stocks[stock] = amount
            else:
                hold_stocks[stock] += amount
    info("--------------readOrdersFromPkl----------------定单库持仓信息（hold_stocks）：" + str(hold_stocks), 'debug') 
    #与get_positions()比较，如果持仓股票不在hold_stocks中，刚报警
    #如果last_date为空，则取昨日的年月日
    if last_date == '':
        last_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    info("--------------readOrdersFromPkl----------------last_date：" + str(last_date), 'debug') 
    positions = get_positions()
    info("--------------readOrdersFromPkl----------------当前持仓（positions）：" + str(positions), 'debug') 
    for stock, position in positions.items():
        info("--------------readOrdersFromPkl----------------持仓股票：" + "stock:" + str(stock) + "，position:" + str(position), 'debug') 
        if position.sid not in hold_stocks:
            info("--------------readOrdersFromPkl----------------" + "stock:" + position.sid + "不在定单库持仓信息（hold_stocks）中", 'debug') 
            #取得g.my_orders中最后一笔记录
            last_order = g.my_orders[last_date]
            #将不在hold_stocks中的股票加到last_order中
            last_order.append({
                'symbol': position.sid,
                'filled_amount': position.amount,
                'entrust_bs': 1,
                'status': 8,
                'price': position.cost_basis
            })
            g.my_orders[last_date] = last_order
        else:
            if int(position.amount) > int(hold_stocks[position.sid]):
                info("--------------readOrdersFromPkl----------------" + "stock:" + stock + "在定单库持仓信息（hold_stocks）中的amount小于当前持仓", 'debug') 
                #取得g.my_orders中最后一笔记录
                last_order = g.my_orders[last_date]
                #将不在hold_stocks中的股票加到last_order中
                last_order.append({
                    'symbol': position.sid,
                    'filled_amount': (int(position.amount) - int(hold_stocks[stock])),
                    'entrust_bs': 1,
                    'status': 8,
                    'price': position.cost_basis
                })
                g.my_orders[last_date] = last_order
            else:
                info("--------------readOrdersFromPkl----------------" + "stock:" + stock + "在hold_stocks中的amount大于当前持仓，报警", 'error') 
    info("--------------readOrdersFromPkl----------------g.my_orders：" + str(g.my_orders), 'info') 
    
"""
#循环将sell_orders的定单，在g.my_orders中减去
1.循环遍历今天卖出订单sell_orders，取得持仓的股票代码字典temp_sell_orders，下标为stock，值为amount
2.与today_sell_stocks比较，如果today_sell_stocks中的stock在temp_sell_orders中， amount相减，如果today_sell_stocks中的大，则报警，如果小，则更新temp_sell_orders
3.更新g.my_orders
"""
def removeSellOrders(sell_orders):
    temp_sell_orders = {}
    for sell_order in sell_orders:
        stock = sell_order['symbol']
        if stock in temp_sell_orders:
            temp_sell_orders[stock] = temp_sell_orders[stock] + int(sell_order['filled_amount'])
        else:
            temp_sell_orders[stock] = int(sell_order['filled_amount'])
    info("--------------removeSellOrders----------------今天所有卖出定单为：" + str(temp_sell_orders), 'debug')
    temp_sell_orders_2 = {}
    for stock, amount in temp_sell_orders.items():
        if stock in g.today_sell_stocks:
            if amount < g.today_sell_stocks[stock]:
                info("--------------removeSellOrders----------------" + "stock:" + stock + "，定单列表总量为：" + str(amount) + "，ptrade自动卖出数量：" + str(g.today_sell_stocks[stock]) + "，在定单库持仓信息（sell_orders）中的amount小于ptrade自动卖出数量（today_sell_stocks中的amount），报警", 'error') 
            else:
                temp_sell_orders_2[stock] = amount - g.today_sell_stocks[stock]     #更新temp_sell_orders_2,减去g.today_sell_stocks[stock]
        else:
            temp_sell_orders_2[stock] = amount
    info("--------------removeSellOrders----------------手工卖出定单为：" + str(temp_sell_orders_2), 'info')
    #将手工卖出定单从g.my_orders中减去
    t_my_orders = {}
    for stock, amount in temp_sell_orders_2.items():
        t_amount = amount
        for date, orders in g.my_orders.items():
            t_orders = []
            for order in orders.copy():
                if order['symbol'] == stock:
                    if int(order['filled_amount']) >= t_amount:
                        order['filled_amount'] = int(order['filled_amount']) - t_amount
                        t_amount = 0
                        t_orders.append(order)
                        info("--------------removeSellOrders----------------"+ date + "，stock:" + stock + "从所有定单个删除" + str(amount) + "，定单减量:" + str(order), 'info') 
                    else:
                        t_amount = t_amount - int(order['filled_amount'])
                        info("--------------removeSellOrders----------------"+ date + "，stock:" + stock + "从所有定单个删除" + str(amount) + "，移除定单:" + str(order), 'info') 
                else:
                    t_orders.append(order)        
            t_my_orders[date] = t_orders        
    g.my_orders = t_my_orders
    info("--------------removeSellOrders----------------g.my_orders：" + str(g.my_orders), 'debug')


def saveOrders():
    all_orders = get_all_orders()
    info("--------------saveOrders----------------今日定单（all_orders）：" + str(all_orders), 'debug') 
    # 获取当前日期作为key
    current_date = datetime.now().strftime('%Y%m%d')
    buy_orders = []
    sell_orders = []
    for buy_order in all_orders.copy():
        info("--------------saveOrders----------------定单（buy_order）：" + str(buy_order), 'debug') 
        #假如status为8，且 entrust_bs 为 1，则添加到buy_orders中
        if int(buy_order['status']) == 8 and int(buy_order['entrust_bs']) == 1:
            buy_orders.append(buy_order)
        #假如status为8，且 entrust_bs 为 2，则添加到sell_orders中
        if int(buy_order['status']) == 8 and int(buy_order['entrust_bs']) == 2:
            sell_orders.append(buy_order)
    #从g.my_orders中减去sell_orders
    removeSellOrders(sell_orders)        
    g.my_orders[current_date] = buy_orders
    info("--------------saveOrders----------------buy_orders：" + str(buy_orders), 'debug') 
    info("--------------saveOrders----------------sell_orders：" + str(sell_orders), 'debug') 
    info("--------------saveOrders----------------g.my_orders：" + str(g.my_orders), 'debug') 
    #将g.my_orders写入文件
    with open(g.notebook_path+'my_orders.pkl','wb') as f:
        pickle.dump(g.my_orders,f,-1)

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
   
    
    
    
    
    # #每支股票卖出时间为当日下午 2：30 分
    # #每支股票买入后持有 7 个交易日；例如：股票在 D1 那天的下午 2 点 55 分买入，则在 D8 那天的下午 2 点 30 分卖出。
    # if g.hand_times == 1:
    #     #读取文件中的持仓信息
    #     with open('hold_positions.json', 'r') as f:
    #         g.my_hold_positions = json.load(f)


    #     #取得当前持仓
    #     positions = get_positions()
    #     info(positions, 'info')
    #     #遍历持仓，取得持仓的股票代码
    #     for stock, position in positions.items():
    #         # 获取当前日期作为key
    #         current_date = datetime.now().strftime('%Y%m%d')
    #         position['buy_date'] = current_date

            

            

