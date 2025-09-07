<?php
require_once './init_stock.php';

// 设置页面编码
header('Content-Type: text/html; charset=utf-8');
?>
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>股票资金流向分析</title>
    <link href="./resources/jquery-ui.min.css" rel="stylesheet">
    <script src="./resources/jquery.min.js"></script>
    <script src="./resources/jquery-ui.min.js"></script>
    <script src="./resources/echarts.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            width: 100%;
            overflow-x: hidden;
        }
        .container {
            width: 100%;
            margin: 0 auto;
            padding: 10px;
            box-sizing: border-box;
        }
        #stockInput {
            width: 200px;
            padding: 8px;
            margin-bottom: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        #stockTags {
            margin-bottom: 20px;
        }
        .tag {
            display: inline-block;
            padding: 4px 8px;
            background: #E3F2FD;
            color: #1976D2;
            border-radius: 3px;
            margin-right: 5px;
            margin-bottom: 5px;
            text-decoration: none;
            font-size: 0.9em;
            cursor: pointer;
        }
        .tag:hover {
            background: #BBDEFB;
        }
        .days-tag {
            font-size: 0.8em;
            color: #666;
            margin-left: 4px;
        }
        .new-tag {
            color: #e91e63;
            font-weight: bold;
        }
        .tab-container {
            margin-bottom: 20px;
        }
        .tab-buttons {
            margin-bottom: 10px;
        }
        .tab-button {
            padding: 8px 16px;
            margin-right: 5px;
            border: none;
            background: #f0f0f0;
            cursor: pointer;
            border-radius: 4px;
        }
        .tab-button.active {
            background: #1976D2;
            color: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .collection-stocks {
            margin: 20px 0;
        }
        .collection-title {
            font-size: 1.1em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .collection-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .collection-item {
            background: white;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
            border: 1px solid #ddd;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .collection-item:hover {
            background: #f0f0f0;
        }
        .collection-time {
            color: #666;
            font-size: 0.9em;
        }
        .recommand-stocks {
            margin: 20px 0;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 5px;
        }
        .recommand-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        #recommandDate {
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .recommand-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .recommand-item {
            background: white;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
            border: 1px solid #ddd;
        }
        .recommand-item:hover {
            background: #f0f0f0;
        }
        .recommand-item.user-recommanded {
            background: #1976D2;
            color: white;
            border-color: #1565C0;
        }
        .stock-info {
            margin: 10px 0;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 4px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .stock-code {
            font-weight: bold;
            font-size: 1.1em;
        }
        .recommand-actions {
            display: flex;
            gap: 10px;
        }
        .recommand-btn {
            padding: 5px 10px;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 0.9em;
            display: flex;
            align-items: center;
        }
        .add-recommand {
            background: #4caf50;
            color: white;
        }
        .remove-recommand {
            background: #f44336;
            color: white;
        }
        .recommand-btn:hover {
            opacity: 0.9;
        }
        .chart-controls {
            margin: 10px 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .chart-controls label {
            margin-right: 15px;
            cursor: pointer;
        }
        .fullscreen-btn {
            padding: 5px 10px;
            background: #1976D2;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-left: auto;
        }
        .fullscreen-btn:hover {
            background: #1565C0;
        }
        #klineChart {
            width: 100%;
            height: calc(100vh - 150px);
            margin-bottom: 20px;
        }
        #moneyflowChart {
            width: 100%;
            height: 400px;
        }
        #macdChart {
            width: 100%;
            height: 150px;
        }
        .fullscreen-chart {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: 9999;
            background-color: white;
        }
        @media (max-width: 768px) {
            #klineChart {
                height: calc(100vh - 300px);
            }
            #moneyflowChart {
                height: 300px;
            }
            #macdChart {
                height: 100px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <input type="text" id="stockInput" placeholder="请输入股票代码" value="<?= htmlspecialchars($_GET['code'] ?? '') ?>">
        <div id="stockTags"></div>
        
        <div class="collection-stocks">
            <div class="collection-title">收藏的股票</div>
            <div id="collectionList" class="collection-list"></div>
        </div>
        
        <div class="recommand-stocks">
            <div class="recommand-title">
                今日推荐股票
                <input type="date" id="recommandDate" value="<?= date('Y-m-d', strtotime('-1 day')) ?>">
                <a href="stock_recommand_profit.php?trade_date=<?= date('Y-m-d', strtotime('-3 day')) ?>">查看推荐股票收益</a>
            </div>
            <div class="recommand-list">
                <!-- 添加Tab容器 -->
                <div class="tab-container">
                    <div class="tab-buttons">
                        <!-- Tab按钮将由JavaScript动态生成 -->
                    </div>
                    <div class="tab-contents">
                        <!-- Tab内容将由JavaScript动态生成 -->
                    </div>
                </div>
            </div>
        </div>
        
        <div class="stock-info" id="stockInfo" style="display: none;">
            <div class="stock-code"></div>
            <div class="recommand-actions">
                <button class="recommand-btn add-recommand" id="addCollection">
                    +可购买股票
                </button>
                <button class="recommand-btn remove-recommand" id="removeCollection">
                    -删除可购买股票
                </button>
            </div>
        </div>
        
        <div class="chart-controls">
            <div>
                <label><input type="checkbox" class="ma-line" data-line="MA5"> MA5</label>
                <label><input type="checkbox" class="ma-line" data-line="MA10"> MA10</label>
                <label><input type="checkbox" class="ma-line" data-line="MA20"> MA20</label>
                <label><input type="checkbox" class="ma-line" data-line="MA30"> MA30</label>
                <label><input type="checkbox" class="show-signals" data-signal="dual-cannon" checked> 显示双响炮信号</label>
                <label><input type="checkbox" class="show-signals" data-signal="yang-three" checked> 显示一阳穿三线</label>
                <label><input type="checkbox" class="show-signals" data-signal="macd" checked> 显示MACD信号</label>
            </div>
            <button id="toggleFullscreen" class="fullscreen-btn">切换全屏</button>
        </div>
        <div id="klineChart"></div>
        <div id="moneyflowChart"></div>
        <div id="macdChart"></div>
    </div>

    <script>
        $(document).ready(function() {
            window.klineData = null;
            const klineChart = echarts.init(document.getElementById('klineChart'));
            const moneyflowChart = echarts.init(document.getElementById('moneyflowChart'));
            const macdChart = echarts.init(document.getElementById('macdChart'));
            let currentKlineData = null;
            let dateToIndex = {}; // 日期与索引的映射
            
            // 为K线图添加点击和框选事件
            function bindKlineChartEvents() {
                console.log('绑定缠论K线图事件...');
                
                // 先移除之前的所有事件
                klineChart.off('datazoom');
                console.log('事件绑定: 之前的datazoom事件已移除');
                
                // 添加鼠标直接事件来捕捉框选动作
                let isSelecting = false;
                let selectStartX = 0;
                let selectEndX = 0;
                
                // 直接使用鼠标事件捕捉框选
                const zr = klineChart.getZr();
                
                // 鼠标按下时记录起始位置 - 简化版
                zr.on('mousedown', function(e) {
                    console.log('鼠标按下位置:', { x: e.offsetX, y: e.offsetY });
                    const inGrid = isInChanKlineGrid(e.offsetX, e.offsetY);
                    console.log('是否在缠论K线图区域内:', inGrid);
                    
                    if (inGrid) {
                        isSelecting = true;
                        selectStartX = e.offsetX;
                        console.log('开始选择缠论K线图, 坐标:', selectStartX);
                    }
                });
                
                // 鼠标松开时完成选择 - 简化版
                zr.on('mouseup', function(e) {
                    if (isSelecting) {
                        selectEndX = e.offsetX;
                        console.log('结束选择缠论K线图, 坐标:', selectEndX);
                        
                        // 如果距离足够大，则认为是有效的选择
                        if (Math.abs(selectEndX - selectStartX) > 10) {
                            handleMouseSelection(selectStartX, selectEndX);
                        }
                        
                        isSelecting = false;
                    }
                });
                
                // 正在移动时的鼠标位置
                zr.on('mousemove', function(e) {
                    // 仅当正在选择时跟踪鼠标移动
                    if (isSelecting) {
                        // 这里可以添加实时反馈效果，如显示选择范围等
                    }
                });
                
                // 保留datazoom事件监听器以便调试
                klineChart.on('datazoom', function(params) {
                    // 只记录日志，不处理
                    console.log('收到datazoom事件:', params);
                });
                
                // 获取缠论K线图的网格信息
                function getGridInfo() {
                    try {
                        const option = klineChart.getOption();
                        return option.grid[1]; // 缠论K线图的网格
                    } catch (e) {
                        console.error('获取网格信息失败:', e);
                        return null;
                    }
                }
                
                // 检查点击位置是否在缠论K线图的网格区域内
                function isInChanKlineGrid(x, y) {
                    const gridInfo = getGridInfo();
                    if (!gridInfo) return false;
                    
                    // 计算网格的位置和大小
                    const chartWidth = zr.getWidth();
                    const chartHeight = zr.getHeight();
                    
                    // 计算实际的像素位置
                    let left, top, width, height;
                    
                    if (typeof gridInfo.left === 'string' && gridInfo.left.includes('%')) {
                        left = chartWidth * parseFloat(gridInfo.left) / 100;
                    } else {
                        left = parseFloat(gridInfo.left) || 0;
                    }
                    
                    if (typeof gridInfo.top === 'string' && gridInfo.top.includes('%')) {
                        top = chartHeight * parseFloat(gridInfo.top) / 100;
                    } else {
                        top = parseFloat(gridInfo.top) || 0;
                    }
                    
                    if (typeof gridInfo.right === 'string' && gridInfo.right.includes('%')) {
                        width = chartWidth - left - chartWidth * parseFloat(gridInfo.right) / 100;
                    } else {
                        width = chartWidth - left - (parseFloat(gridInfo.right) || 0);
                    }
                    
                    if (typeof gridInfo.height === 'string' && gridInfo.height.includes('%')) {
                        height = chartHeight * parseFloat(gridInfo.height) / 100;
                    } else {
                        height = parseFloat(gridInfo.height) || 0;
                    }
                    
                    // 输出调试信息
                    console.log('缠论K线图网格区域:', { left, top, width, height });
                    console.log('点击坐标:', { x, y });
                    
                    return x >= left && x <= (left + width) && 
                           y >= top && y <= (top + height);
                }
                
                // 使用精确的手动计算方法提取框选日期
                function handleMouseSelection(startX, endX) {
                    try {
                        // 获取图表关键信息
                        const gridInfo = getGridInfo();
                        if (!gridInfo) {
                            console.error('缺少图表网格信息');
                            return;
                        }
                        
                        // 获取图表数据
                        const option = klineChart.getOption();
                        if (!option.xAxis || !option.xAxis[1] || !option.xAxis[1].data) {
                            console.error('无法获取坐标轴数据');
                            return;
                        }
                        
                        // 获取X轴数据和图表尺寸
                        const axisData = option.xAxis[1].data;
                        const dataLength = axisData.length;
                        const chartWidth = zr.getWidth();
                        
                        console.log('图表原始数据长度:', dataLength);
                        
                        // 计算图表的有效绘图区域
                        let left, right, width;
                        if (typeof gridInfo.left === 'string' && gridInfo.left.includes('%')) {
                            left = chartWidth * parseFloat(gridInfo.left) / 100;
                        } else {
                            left = parseFloat(gridInfo.left) || 0;
                        }
                        
                        if (typeof gridInfo.right === 'string' && gridInfo.right.includes('%')) {
                            right = chartWidth * parseFloat(gridInfo.right) / 100;
                        } else {
                            right = parseFloat(gridInfo.right) || 0;
                        }
                        
                        width = chartWidth - left - right;
                        
                        // 将鼠标坐标限制在图表的有效区域内
                        const validStartX = Math.max(left, Math.min(chartWidth - right, startX));
                        const validEndX = Math.max(left, Math.min(chartWidth - right, endX));
                        
                        // 计算相对位置(归一化为0-1的比例值)
                        const startRatio = (validStartX - left) / width;
                        const endRatio = (validEndX - left) / width;
                        
                        // 确保起始和结束的顺序正确
                        const minRatio = Math.min(startRatio, endRatio);
                        const maxRatio = Math.max(startRatio, endRatio);
                        
                        // 尝试直接根据像素位置找到对应的数据点
                        // 我们采用一种更直接的方法，直接计算鼠标坐标对应的数据点
                        // 这避免了日期排序方向的问题
                        
                        // 获取当前缩放状态下的数据范围
                        let dataStartIndex = 0;
                        let dataEndIndex = dataLength - 1;
                        
                        // 检查是否有数据缩放组件并获取当前数据窗口范围
                        if (option.dataZoom && option.dataZoom[0]) {
                            const zoomInfo = option.dataZoom[0];
                            if (zoomInfo.startValue !== undefined && zoomInfo.endValue !== undefined) {
                                dataStartIndex = zoomInfo.startValue;
                                dataEndIndex = zoomInfo.endValue;
                            } else if (zoomInfo.start !== undefined && zoomInfo.end !== undefined) {
                                // 百分比形式的缩放范围
                                dataStartIndex = Math.floor(dataLength * zoomInfo.start / 100);
                                dataEndIndex = Math.floor(dataLength * zoomInfo.end / 100);
                            }
                        }
                        
                        // 计算当前可见数据的长度
                        const visibleDataLength = dataEndIndex - dataStartIndex + 1;
                        
                        // 将有效宽度平均分配给可见的数据点
                        const pointWidth = width / visibleDataLength;
                        
                        // 右侧起始点和结束点的相对位置
                        const startPos = validStartX - left;
                        const endPos = validEndX - left;
                        
                        // 计算最接近的数据点索引，基于当前可见数据范围
                        let startIndex = dataStartIndex + Math.floor(startPos / pointWidth);
                        let endIndex = dataStartIndex + Math.floor(endPos / pointWidth);
                        
                        console.log('像素位置计算:', {
                            pointWidth,
                            startPos,
                            endPos,
                            dataRange: { start: dataStartIndex, end: dataEndIndex, visible: visibleDataLength },
                            resultIndices: { startIndex, endIndex }
                        });
                        
                        // 确保起始点和结束点正确
                        if (startIndex > endIndex) {
                            [startIndex, endIndex] = [endIndex, startIndex];
                        }
                        
                        // 确保索引在有效范围内
                        startIndex = Math.max(0, Math.min(dataLength - 1, startIndex));
                        endIndex = Math.max(0, Math.min(dataLength - 1, endIndex));
                        
                        // 获取对应的日期
                        console.log(currentKlineData.klineData)
                        console.log('框选计算详情:', {
                            startIndex,
                            endIndex
                        });
                        const startDate = currentKlineData.klineData[startIndex];
                        const endDate = currentKlineData.klineData[endIndex];
                        
                        // const startDate = axisData[startIndex];
                        // const endDate = axisData[endIndex];
                        
                        console.log('框选计算详情:', {
                            鼠标坐标: { startX, endX },
                            有效坐标: { validStartX, validEndX },
                            数据范围: { startIndex, endIndex, total: dataLength, visible: visibleDataLength },
                            可见数据范围: { 开始: dataStartIndex, 结束: dataEndIndex },
                            日期范围: { 开始: startDate, 结束: endDate },
                            坐标和日期项: {
                                起始像素: validStartX, 
                                起始日期: startDate,
                                结束像素: validEndX, 
                                结束日期: endDate
                            }
                        });
                        
                        // 显示框选结果
                        if (startDate && endDate) {
                            // alert('框选时间范围:\n开始日期: ' + startDate + '\n结束日期: ' + endDate);
                        } else {
                            // alert('框选日期获取失败，请重试');
                        }
                    } catch (e) {
                        console.error('框选处理错误:', e);
                        alert('框选处理出错: ' + e.message);
                    }
                }
                
                
                // 移除重复的鼠标移动事件
                
                // 保留datazoom事件监听器以便调试
                klineChart.on('datazoom', function(params) {
                    // 只记录日志，不处理
                    console.log('收到datazoom事件:', params);
                });
                
                // 添加获取正确的日期的API方法
                // 加强版获取日期函数 - 直接使用原始K线数据获取日期
                klineChart.getDateFromPosition = function(x) {
                    try {
                        if (!currentKlineData || !currentKlineData.klineData || !currentKlineData.klineData.length) {
                            console.error('获取日期时缺少K线数据');
                            return null;
                        }
                        
                        // 使用原始K线数据中的日期
                        const rawDates = currentKlineData.klineData.map(item => item[0]);
                        const totalDates = rawDates.length;
                        
                        const gridInfo = getGridInfo();
                        if (!gridInfo) return null;
                        
                        const chartWidth = zr.getWidth();
                        const left = typeof gridInfo.left === 'string' ? 
                            chartWidth * parseFloat(gridInfo.left) / 100 : parseFloat(gridInfo.left);
                        const right = typeof gridInfo.right === 'string' ? 
                            chartWidth * parseFloat(gridInfo.right) / 100 : parseFloat(gridInfo.right);
                        
                        const effectiveWidth = chartWidth - left - right;
                        const relativeX = x - left;
                        const ratio = Math.max(0, Math.min(1, relativeX / effectiveWidth));
                        const index = Math.floor(ratio * (totalDates - 1));
                        
                        return {
                            date: rawDates[index],
                            index: index,
                            ratio: ratio
                        };
                    } catch (e) {
                        console.error('日期计算错误:', e);
                        return null;
                    }
                };
                
                console.log('缠论K线图事件已成功绑定！');
            }
            
            // 控制MA线和信号显示/隐藏
            $('.chart-controls input[type="checkbox"]').change(function() {
                if (currentKlineData) {
                    updateKlineChart(currentKlineData);
                }
            });
            
            // 全屏切换功能
            $('#toggleFullscreen').click(function() {
                const klineChartElement = document.getElementById('klineChart');
                klineChartElement.classList.toggle('fullscreen-chart');
                
                if (klineChartElement.classList.contains('fullscreen-chart')) {
                    $(this).text('退出全屏');
                    // 重新调整图表大小
                    klineChart.resize();
                } else {
                    $(this).text('切换全屏');
                    // 重新调整图表大小
                    klineChart.resize();
                }
            });
            
            // 股票代码自动完成
            $("#stockInput").autocomplete({
                source: function(request, response) {
                    $.ajax({
                        url: 'get_stocks.php',
                        dataType: "json",
                        data: {
                            term: request.term
                        },
                        success: function(data) {
                            response(data);
                        }
                    });
                },
                minLength: 2,
                select: function(event, ui) {
                    loadAllData(ui.item.value);
                }
            });

            let currentStockCode = '';
            let currentStockName = '';

            // 绑定收藏按钮事件
            $('#addCollection').click(function() {
                // updateCollection(currentStockCode, 'add');
                updateRecommand(currentStockCode, 'add');
            });
            
            $('#removeCollection').click(function() {
                // updateCollection(currentStockCode, 'remove');
                updateRecommand(currentStockCode, 'remove');
            });

            function updateCollection(code, action) {
                if (!code) return;
                
                $.ajax({
                    url: 'update_collection.php',
                    method: 'POST',
                    data: {
                        code: code,
                        action: action
                    },
                    success: function(response) {
                        if (response.success) {
                            loadCollectionStocks();
                            loadRecommandStocks();
                        }
                    }
                });
            }

            function loadCollectionStocks() {
                $.ajax({
                    url: 'get_collections.php',
                    dataType: 'json',
                    success: function(data) {
                        if (data.stocks) {
                            const stocksHtml = data.stocks.map(stock => 
                                `<div class="collection-item" data-code="${stock.code}">
                                    <div class="collection-time">${stock.collect_date}</div>
                                    <div class="stock-name">${stock.code} - ${stock.stock_name}</div>
                                </div>`
                            ).join('');
                            $('#collectionList').html(stocksHtml);
                        }
                    }
                });
            }

            function updateRecommand(code, action) {
                if (!code) return;
                
                $.ajax({
                    url: 'update_recommand.php',
                    method: 'POST',
                    data: {
                        code: code,
                        action: action
                    },
                    success: function(response) {
                        if (response.success) {
                            loadRecommandStocks();
                            loadCollectionStocks();
                        }
                    }
                });
            }

            function loadAllData(stockCode) {
                currentStockCode = stockCode;
                loadKlineData(stockCode);
                // loadMoneyflowData(stockCode);
                // MACD数据现在已经在K线数据中加载 - 移除对loadMacdData的调用
                // loadMacdData(stockCode);
                loadStockTags(stockCode);
                updateStockInfo(stockCode);
                
                // 确保绑定K线图事件
                setTimeout(bindKlineChartEvents, 1000);
            }
            // 更新股票信息
            function updateStockInfo(stockCode) {
                $('.stock-code').text('股票编号：' + stockCode + ' - 数据加载中...');
                $('#stockInfo').show();
            }
            // 获取K线数据
            function loadKlineData(stockCode) {
                if (!stockCode) return;
                
                // 清空K线图，防止旧数据永久性残留
                if (klineChart) {
                    klineChart.clear();
                }
                
                $.ajax({
                    url: 'get_kline_data.php',
                    dataType: 'json',
                    data: { code: stockCode },
                    success: function(data) {
                        updateKlineChart(data);
                        updateKlineTitle(stockCode);
                    },
                    error: function(xhr, status, error) {
                        console.error('加载K线数据出错:', error);
                    }
                });
            }
            // 获取资金流向
            function loadMoneyflowData(stockCode) {
                $.ajax({
                    url: 'get_moneyflow.php',
                    dataType: 'json',
                    data: { code: stockCode },
                    success: function(data) {
                        updateMoneyflowChart(data);
                    }
                });
            }
            // 获取MACD数据 - 现在MACD数据已集成到get_kline_data.php中返回
            function loadMacdData(stockCode) {
                // 空函数保留接口兼容性
                console.log('MACD数据现在已通过K线数据加载');
            }
            // 获取股票标签
            function loadStockTags(stockCode) {
                $.ajax({
                    url: 'get_stock_tags.php',
                    dataType: 'json',
                    data: { code: stockCode },
                    success: function(data) {
                        if (data.tags) {
                            const tagsHtml = data.tags.map(tag => 
                                `<a href="tag_details.php?tag=${encodeURIComponent(tag)}" class="tag">${tag}</a>`
                            ).join('');
                            $('#stockTags').html(tagsHtml);
                        }
                        if (data.stockName) {
                            // $('#stockName').text(data.stockName);
                            $('.stock-code').text('股票编号：' + stockCode + ' - ' + data.stockName);
                            $('#stockInfo').show();
                        }
                    }
                });
            }

            function loadRecommandStocks() {
                const selectedDate = $('#recommandDate').val();
                $.ajax({
                    url: 'get_recommands.php',
                    dataType: 'json',
                    data: { date: selectedDate },
                    success: function(response) {
                        if (response.error) {
                            console.error(response.error);
                            return;
                        }

                        // 清空现有的Tab按钮和内容
                        $('.tab-buttons').empty();
                        $('.tab-contents').empty();
                        
                        // 创建Tab按钮
                        Object.keys(response.sources).forEach((source, index) => {
                            const buttonClass = index === 0 ? 'tab-button active' : 'tab-button';
                            $('.tab-buttons').append(`
                                <button class="${buttonClass}" data-source="${source}">
                                    ${response.sources[source]}
                                </button>
                            `);
                            
                            // 创建对应的内容区域
                            const contentClass = index === 0 ? 'tab-content active' : 'tab-content';
                            const stocksHtml = response.stocks[source]
                                .map(stock => {
                                    const daysTag = stock.days === 'new' 
                                        ? '<span class="days-tag new-tag">(NEW)</span>'
                                        : stock.days ? `<span class="days-tag">(${stock.days}天)</span>` : '';
                                    
                                    return `<a class="tag" data-code="${stock.code}">
                                        ${stock.code} - ${stock.stock_name}${daysTag}
                                    </a>`;
                                })
                                .join('');
                            
                            $('.tab-contents').append(`
                                <div class="${contentClass}" data-source="${source}">
                                    ${stocksHtml}
                                </div>
                            `);
                        });
                        
                        // 绑定Tab切换事件
                        $('.tab-button').click(function() {
                            const source = $(this).data('source');
                            $('.tab-button').removeClass('active');
                            $(this).addClass('active');
                            $('.tab-content').removeClass('active');
                            $(`.tab-content[data-source="${source}"]`).addClass('active');
                        });
                        
                        // 绑定股票点击事件
                        $('.tag').click(function() {
                            const code = $(this).data('code');
                            $('#stockInput').val(code);
                            loadAllData(code);
                        });
                    }
                });
            }

            // 初始加载推荐股票
            loadRecommandStocks();

            // 监听日期选择器变化
            $('#recommandDate').change(function() {
                loadRecommandStocks();
            });

            function updateKlineTitle(stockCode) {
                if (klineChart && klineChart.getOption()) {
                    const option = klineChart.getOption();
                    if (option.title && option.title[0]) {
                        option.title[0].text = `股票${stockCode}K线图`;
                        option.title[0].subtext = '';
                        klineChart.setOption({
                            title: option.title
                        });
                    }
                }
            }

            function updateKlineChart(data) {
                currentKlineData = data;
                const showMA5 = $('.ma-line[data-line="MA5"]').prop('checked');
                const showMA10 = $('.ma-line[data-line="MA10"]').prop('checked');
                const showMA20 = $('.ma-line[data-line="MA20"]').prop('checked');
                const showMA30 = $('.ma-line[data-line="MA30"]').prop('checked');
                const showDualCannon = $('.show-signals[data-signal="dual-cannon"]').prop('checked');
                const showYangThree = $('.show-signals[data-signal="yang-three"]').prop('checked');
                const showMacd = $('.show-signals[data-signal="macd"]').prop('checked');
                
                // 创建日期到索引的映射，用于后续快速查找
                dateToIndex = {}; // 更新全局变量
                data.klineData.forEach((item, index) => {
                    dateToIndex[item[0]] = index;
                });
                
                // 判断日期排序方向
                if (data.klineData.length > 1) {
                    const firstDate = data.klineData[0][0];
                    const lastDate = data.klineData[data.klineData.length - 1][0];
                    window.isDateReversed = new Date(firstDate) > new Date(lastDate);
                    console.log('日期排序检测:', {
                        第一项: firstDate,
                        最后一项: lastDate,
                        是否倒序: window.isDateReversed
                    });
                }
                
                // 每次更新图表后重新绑定事件
                setTimeout(function() {
                    bindKlineChartEvents();
                    console.log('缠论K线图事件重新绑定完成');
                }, 1000);
                
                const option = {
                    tooltip: {
                        trigger: 'axis',
                        axisPointer: {
                            type: 'cross'
                        }
                    },
                    legend: {
                        data: ['K线', 'MA5', 'MA10', 'MA20', 'MA30', '成交量', '双响炮', '一阳穿三线', 'MACD金叉', 'MACD死叉', '缠论笔']
                    },
                    toolbox: {
                        feature: {
                            dataZoom: {
                                title: {
                                    zoom: '区域缩放',
                                    back: '还原'
                                },
                                xAxisIndex: [0, 1, 2, 3],
                                yAxisIndex: [0, 1, 2, 3]
                            },
                            restore: {
                                title: '重置'
                            }
                        }
                    },
                    grid: [{
                        left: '10%',
                        right: '8%',
                        height: '30%'  // 原始K线图区域，高度调整为35%
                    }, {
                        left: '10%',
                        right: '8%',
                        top: '40%',    // 缠论K线图区域，从40%位置开始
                        height: '30%'   // 高度调整为20%
                    }, {
                        left: '10%',
                        right: '8%',
                        top: '70%',    // 成交量区域，从63%位置开始
                        height: '10%'
                    }, {
                        left: '10%',
                        right: '8%',
                        top: '85%',    // MACD区域，从80%位置开始
                        height: '10%'
                    }],
                    xAxis: [{
                        type: 'category',
                        data: data.klineData.map(item => item[0]),
                        scale: true,
                        boundaryGap: false,
                        axisLine: {onZero: false},
                        splitLine: {show: false},
                        min: 'dataMin',
                        max: 'dataMax'
                    }, {
                        type: 'category',
                        gridIndex: 1,
                        data: data.klineData.map(item => item[0]),
                        scale: true,
                        boundaryGap: false,
                        axisLine: {onZero: false},
                        axisTick: {show: false},
                        splitLine: {show: false},
                        axisLabel: {show: false},
                        min: 'dataMin',
                        max: 'dataMax'
                    }, {
                        type: 'category',
                        gridIndex: 2,
                        data: data.klineData.map(item => item[0]),
                        scale: true,
                        boundaryGap: false,
                        axisLine: {onZero: false},
                        axisTick: {show: false},
                        splitLine: {show: false},
                        axisLabel: {show: true},
                        min: 'dataMin',
                        max: 'dataMax'
                    }, {
                        type: 'category',
                        gridIndex: 3,
                        data: data.klineData.map(item => item[0]),
                        scale: true,
                        boundaryGap: false,
                        axisLine: {onZero: false},
                        axisTick: {show: false},
                        splitLine: {show: false},
                        axisLabel: {show: true},
                        min: 'dataMin',
                        max: 'dataMax'
                    }],
                    yAxis: [{
                        scale: true,
                        splitArea: {
                            show: true
                        },
                        name: '原始K线价格',
                        nameLocation: 'middle',
                        nameGap: 30,
                        nameTextStyle: {
                            fontSize: 12
                        }
                    }, {
                        scale: true,
                        gridIndex: 1,
                        splitArea: {
                            show: true
                        },
                        name: '缠论K线价格',
                        nameLocation: 'middle',
                        nameGap: 30,
                        nameTextStyle: {
                            fontSize: 12
                        }
                    }, {
                        scale: true,
                        gridIndex: 2,
                        splitNumber: 2,
                        axisLabel: {show: true},
                        axisLine: {show: true},
                        axisTick: {show: true},
                        splitLine: {show: true}
                    }, {
                        scale: true,
                        gridIndex: 3,
                        splitNumber: 2,
                        axisLabel: {show: true},
                        axisLine: {show: true},
                        axisTick: {show: true},
                        splitLine: {show: true}
                    }],
                    dataZoom: [{
                        // 原始K线图的区域缩放 - 可以框选和移动
                        type: 'inside',
                        xAxisIndex: [0, 2, 3],  // 原始K线、成交量和MACD共用
                        start: 50,
                        end: 100,
                        zoomOnMouseWheel: true,
                        moveOnMouseMove: true,
                        moveOnMouseWheel: true,
                        filterMode: 'filter'
                    }, {
                        show: true,
                        xAxisIndex: [0, 1, 2, 3],
                        type: 'slider',
                        top: '97%',
                        start: 50,
                        end: 100
                    }],
                    series: [{
                        name: 'K线',
                        type: 'candlestick',
                        data: data.klineData.map(item => [item[1], item[2], item[3], item[4]]),
                        itemStyle: {
                            color: '#ec0000',
                            color0: '#00da3c',
                            borderColor: '#ec0000',
                            borderColor0: '#00da3c'
                        }
                    }, {
                        name: '成交量',
                        type: 'bar',
                        xAxisIndex: 2,
                        yAxisIndex: 2,
                        data: data.klineData.map(item => item[5])
                    }]
                };

                // 有条件添加MA5线
                if (showMA5) {
                    option.series.push({
                        name: 'MA5',
                        type: 'line',
                        data: data.ma5,
                        smooth: true,
                        lineStyle: {
                            opacity: 0.5
                        }
                    });
                }
                if (showMA10){
                    option.series.push({
                        name: 'MA10',
                        type: 'line',
                        data: data.ma10,
                        smooth: true,
                        lineStyle: {
                            opacity: 0.5
                        }
                    });
                }
                if (showMA20){
                    option.series.push({
                        name: 'MA20',
                        type: 'line',
                        data: data.ma20,
                        smooth: true,
                        lineStyle: {
                            opacity: 0.5
                        }
                    });
                }
                if (showMA30){
                    option.series.push({
                        name: 'MA30',
                        type: 'line',
                        data: data.ma30,
                        smooth: true,
                        lineStyle: {
                            opacity: 0.5
                        }
                    });
                }

                // 添加中枢区域标记到原始K线图
                if (data.centralZones && data.centralZones.length > 0) {
                    // 处理每个中枢
                    data.centralZones.forEach(zone => {
                        // 中枢线段
                        const segments = zone.segments;
                        if (!segments || segments.length < 2) return;
                        
                        // 中枢起始线段（进入线）
                        const inLine = segments[segments.length - 1];
                        // 中枢结束线段（离开线）
                        const outLine = segments[0];
                        
                        // 中枢上下沿水平线
                        const zgValue = parseFloat(zone.zg);
                        const zdValue = parseFloat(zone.zd);
                        const min_in_diffValue = parseFloat(zone.min_in_diff);
                        const min_out_diffValue = parseFloat(zone.min_out_diff);
                        console.log('中枢进入段和离开段MACD值：',  {min_in_diffValue, min_out_diffValue});
                        // 确定中枢起点和终点的日期和坐标
                        if (inLine && inLine.start && inLine.end) {
                            const inStartDate = inLine.start.date;
                            const inEndDate = inLine.end.date;
                            const inStartValue = parseFloat(inLine.start.value);
                            const inEndValue = parseFloat(inLine.end.value);
                            
                            if (dateToIndex[inStartDate] !== undefined && dateToIndex[inEndDate] !== undefined) {
                                // 添加中枢进入线（加粗橙色）
                                option.series.push({
                                    name: '原始中枢进入线',
                                    type: 'line',
                                    // 在原始K线图显示，不设置xAxisIndex和yAxisIndex
                                    data: [
                                        [inStartDate, inStartValue],
                                        [inEndDate, inEndValue]
                                    ],
                                    lineStyle: {
                                        color: '#FF8C00',  // 深橙色
                                        width: 3,
                                        type: 'solid'
                                    },
                                    symbol: 'circle',
                                    symbolSize: 6,
                                    z: 13
                                });
                            }
                        }
                        
                        if (outLine && outLine.start && outLine.end) {
                            const outStartDate = outLine.start.date;
                            const outEndDate = outLine.end.date;
                            const outStartValue = parseFloat(outLine.start.value);
                            const outEndValue = parseFloat(outLine.end.value);
                            
                            if (dateToIndex[outStartDate] !== undefined && dateToIndex[outEndDate] !== undefined) {
                                // 添加中枢离开线（加粗橙色）
                                option.series.push({
                                    name: '原始中枢离开线',
                                    type: 'line',
                                    // 在原始K线图显示，不设置xAxisIndex和yAxisIndex
                                    data: [
                                        [outStartDate, outStartValue],
                                        [outEndDate, outEndValue]
                                    ],
                                    lineStyle: {
                                        color: '#FF8C00',  // 深橙色
                                        width: 3,
                                        type: 'solid'
                                    },
                                    symbol: 'circle',
                                    symbolSize: 6,
                                    z: 13
                                });
                            }
                        }
                        
                        // 获取中枢区域的开始和结束日期
                        let centralStartDate, centralEndDate;
                        
                        if (inLine && inLine.end && outLine && outLine.start) {
                            centralStartDate = inLine.end.date;
                            centralEndDate = outLine.start.date;
                            
                            if (dateToIndex[centralStartDate] !== undefined && 
                                dateToIndex[centralEndDate] !== undefined) {
                                
                                // 添加中枢区域上沿（橙色水平线）
                                option.series.push({
                                    name: '原始中枢上沿',
                                    type: 'line',
                                    // 在原始K线图显示，不设置xAxisIndex和yAxisIndex
                                    data: [
                                        [centralStartDate, zgValue],
                                        [centralEndDate, zgValue]
                                    ],
                                    lineStyle: {
                                        color: '#FF8C00',  // 深橙色
                                        width: 2,
                                        type: 'dashed'
                                    },
                                    z: 13
                                });
                                
                                // 添加中枢区域下沿（橙色水平线）
                                option.series.push({
                                    name: '原始中枢下沿',
                                    type: 'line',
                                    // 在原始K线图显示，不设置xAxisIndex和yAxisIndex
                                    data: [
                                        [centralStartDate, zdValue],
                                        [centralEndDate, zdValue]
                                    ],
                                    lineStyle: {
                                        color: '#FF8C00',  // 深橙色
                                        width: 2,
                                        type: 'dashed'
                                    },
                                    z: 13
                                });
                                
                                // 添加中枢区域填充色
                                option.series.push({
                                    name: '原始中枢区域',
                                    type: 'line',
                                    // 在原始K线图显示，不设置xAxisIndex和yAxisIndex
                                    showSymbol: false,
                                    data: [
                                        // 形成一个矩形路径
                                        [centralStartDate, zgValue],
                                        [centralEndDate, zgValue],
                                        [centralEndDate, zdValue],
                                        [centralStartDate, zdValue],
                                        [centralStartDate, zgValue]
                                    ],
                                    areaStyle: {
                                        color: 'rgba(255, 140, 0, 0.1)'  // 淡橙色填充
                                    },
                                    lineStyle: {
                                        width: 0  // 不显示边框线
                                    },
                                    z: 12
                                });
                            }
                        }
                    });
                    
                    // 更新图例
                    if (!option.legend.data.includes('原始中枢区域')) {
                        option.legend.data.push('原始中枢区域');
                    }
                }
                
                // 添加缠论笔到原始K线图
                if (data.segments && data.segments.length > 0) {
                    // 为每个笔创建一个单独的折线系列
                    data.segments.forEach((segment, index) => {
                        const startDate = segment.start.date;
                        const endDate = segment.end.date;
                        const startValue = parseFloat(segment.start.value);
                        const endValue = parseFloat(segment.end.value);
                        
                        // 确保起点和终点的日期存在于数据中
                        if (dateToIndex[startDate] !== undefined && dateToIndex[endDate] !== undefined) {
                            // 创建一个单独的折线系列，只包含两个点（起点和终点）
                            option.series.push({
                                name: '原始K线缠论笔',
                                type: 'line',
                                // 不指定xAxisIndex和yAxisIndex，默认使用0，在原始K线图中显示
                                data: [
                                    [startDate, startValue],
                                    [endDate, endValue]
                                ],
                                showSymbol: false,  // 不显示点标记
                                lineStyle: {
                                    color: '#0074D9',  // 深橙色，与缠论笔区分
                                    width: 2
                                },
                                z: 12
                            });
                        }
                    });
                    
                    // 在图例中只显示一个原始K线缠论笔条目
                    if (!option.legend.data.includes('原始K线缠论笔')) {
                        option.legend.data.push('原始K线缠论笔');
                    }
                }
                
                // 添加顶底分型标记到原始K线图
                if (data.patterns && data.patterns.length > 0) {
                    // 创建顶分型和底分型的数据点
                    const originalTopPoints = [];
                    const originalBottomPoints = [];
                    
                    data.patterns.forEach(pattern => {
                        const patternDate = pattern.date;
                        const patternIndex = dateToIndex[patternDate];
                        
                        if (patternIndex !== undefined) {
                            // 根据分型类型获取相应的值
                            let displayValue = parseFloat(pattern.value);
                            let label = null;
                            
                            // 顶分型显示high，底分型显示low
                            if (pattern.type === 'top' && pattern.high) {
                                label = pattern.high;
                            } else if (pattern.type === 'bottom' && pattern.low) {
                                label = pattern.low;
                            }
                            
                            const pointData = {
                                value: [patternDate, displayValue],
                                itemStyle: {},
                                label: {
                                    show: label !== null,
                                    position: pattern.type === 'top' ? 'top' : 'bottom',
                                    formatter: label !== null ? label : '',
                                    fontSize: 10,
                                    color: pattern.type === 'top' ? '#FF4136' : '#2ECC40',
                                    backgroundColor: 'rgba(255, 255, 255, 0.7)',
                                    padding: [2, 4],
                                    borderRadius: 3
                                }
                            };
                            
                            if (pattern.type === 'top') {
                                pointData.itemStyle.color = '#FF4136'; // 红色
                                originalTopPoints.push(pointData);
                            } else if (pattern.type === 'bottom') {
                                pointData.itemStyle.color = '#2ECC40'; // 绿色
                                originalBottomPoints.push(pointData);
                            }
                        }
                    });
                    
                    // 添加顶分型散点到原始K线图
                    if (originalTopPoints.length > 0) {
                        option.series.push({
                            name: '原始顶分型',
                            type: 'scatter',
                            symbol: 'triangle',
                            symbolSize: 1,
                            symbolRotate: 180, // 向下的三角形
                            data: originalTopPoints,
                            z: 11
                        });
                    }
                    
                    // 添加底分型散点到原始K线图
                    if (originalBottomPoints.length > 0) {
                        option.series.push({
                            name: '原始底分型',
                            type: 'scatter',
                            symbol: 'triangle',
                            symbolSize: 1,
                            data: originalBottomPoints,
                            z: 11
                        });
                    }
                    
                    // 更新图例
                    if (!option.legend.data.includes('原始顶分型')) {
                        option.legend.data.push('原始顶分型');
                    }
                    if (!option.legend.data.includes('原始底分型')) {
                        option.legend.data.push('原始底分型');
                    }
                }

                // 添加双响炮信号
                if (showDualCannon) {
                    const dualCannonMarkers = [];
                    data.klineData.forEach((item, index) => {
                        if (item[7] === 1) {  // dual_cannon 字段为1
                            const price = item[2];  // 收盘价
                            dualCannonMarkers.push({
                                name: '双响炮',
                                coord: [item[0], price],
                                symbol: 'path://M2,2 L10,10 L18,2 L10,18 L2,2 z',  // 自定义箭头形状
                                symbolSize: 20,
                                symbolRotate: 0,
                                itemStyle: {
                                    color: '#ffa500'  // 橙色
                                },
                                label: {
                                    show: true,
                                    position: 'top',
                                    formatter: '双响炮',
                                    fontSize: 12,
                                    color: '#ffa500'
                                }
                            });
                        }
                    });

                    if (dualCannonMarkers.length > 0) {
                        option.series.push({
                            name: '双响炮信号',
                            type: 'scatter',
                            data: [],
                            markPoint: {
                                data: dualCannonMarkers
                            }
                        });
                    }
                }

                // 添加一阳穿三线信号
                if (showYangThree) {
                    const yangThreeMarkers = [];
                    data.klineData.forEach((item, index) => {
                        if (item[8] === 1) {  // yang_three_mas 字段为1
                            const price = item[2];  // 收盘价
                            yangThreeMarkers.push({
                                name: '一阳穿三线',
                                coord: [item[0], price * 1.05],  // 在K线上方显示，距离为收盘价的5%
                                symbol: 'path://M2,2 L10,2 L10,10 L18,10 L10,18 L2,10 L10,10 z',  // 自定义箭头形状
                                symbolSize: 20,
                                symbolRotate: -90,  // 向上的箭头
                                itemStyle: {
                                    color: '#ff4444'  // 红色
                                },
                                label: {
                                    show: true,
                                    position: 'top',
                                    formatter: '一阳穿三线',
                                    fontSize: 12,
                                    color: '#ff4444',
                                    offset: [0, 5]  // 标签位置微调
                                }
                            });
                        }
                    });

                    if (yangThreeMarkers.length > 0) {
                        option.series.push({
                            name: '一阳穿三线信号',
                            type: 'scatter',
                            data: [],
                            markPoint: {
                                data: yangThreeMarkers
                            }
                        });
                    }
                }

                // 添加MACD信号
                if (showMacd) {
                    const macdMarkers = [];
                    data.klineData.forEach((item, index) => {
                        if (item[9] !== 0) {  // macd_signal不为0
                            const price = item[2];  // 收盘价
                            const isGoldenCross = item[9] === 1;  // 金叉
                            
                            macdMarkers.push({
                                name: isGoldenCross ? 'MACD金叉' : 'MACD死叉',
                                coord: [item[0], price * (isGoldenCross ? 1.07 : 0.93)],  // 金叉在上方7%，死叉在下方7%
                                symbol: isGoldenCross 
                                    ? 'path://M2,18 L10,2 L18,18 L10,18 L2,18 z'  // 向上三角形
                                    : 'path://M2,2 L10,18 L18,2 L10,2 L2,2 z',  // 向下三角形
                                symbolSize: 15,
                                itemStyle: {
                                    color: isGoldenCross ? '#FF6B6B' : '#4CAF50'  // 金叉红色，死叉绿色
                                },
                                label: {
                                    show: true,
                                    position: isGoldenCross ? 'top' : 'bottom',
                                    formatter: isGoldenCross ? 'MACD金叉' : 'MACD死叉',
                                    fontSize: 12,
                                    color: isGoldenCross ? '#FF6B6B' : '#4CAF50',
                                    offset: [0, isGoldenCross ? -5 : 5]
                                }
                            });
                        }
                    });

                    if (macdMarkers.length > 0) {
                        option.series.push({
                            name: 'MACD信号',
                            type: 'scatter',
                            data: [],
                            markPoint: {
                                data: macdMarkers
                            }
                        });
                    }
                }
                //添加处理包含关系后的K线图 (基于缠论)
                if (data.processed_klines && data.processed_klines.length > 0) {
                    // 创建一个日期到索引的映射
                    const dateToIndex = {};
                    data.klineData.forEach((item, index) => {
                        dateToIndex[item[0]] = index;
                    });
                    
                    // 准备缠论K线数据
                    const chanKlineData = [];
                    for (let i = 0; i < data.klineData.length; i++) {
                        chanKlineData.push([null, null, null, null]);
                    }
                    
                    data.processed_klines.forEach(item => {
                        if (item.merged === false) { // 只显示未合并的K线
                            const index = dateToIndex[item.trade_date];
                            if (index !== undefined) {
                                // 将缠论K线的高低点作为开收盘价
                                const high = parseFloat(item.c_high);
                                const low = parseFloat(item.c_low);
                                
                                // 对于蜡烛图，我们将最高价也设为开盘价，最低价也设为收盘价
                                // 这样可以确保蜡烛图是实心的，同时只反映高低点
                                chanKlineData[index] = [
                                    high,  // 开盘价 = 最高价
                                    low,   // 收盘价 = 最低价
                                    low,   // 最低价
                                    high   // 最高价
                                ];
                            }
                        }
                    });
                    
                    // 添加缠论K线系列
                    option.series.push({
                        name: '缠论K线',
                        type: 'candlestick',
                        xAxisIndex: 1,
                        yAxisIndex: 1,
                        data: chanKlineData,
                        itemStyle: {
                            color: '#f0a329',      // 上涨颜色（橙色）
                            color0: '#54d9bf',     // 下跌颜色（青色）
                            borderColor: '#f0a329',
                            borderColor0: '#54d9bf',
                            borderWidth: 0         // 设置为0以移除K线上下两端的线条
                        },
                        z: 10
                    });

                    // 添加顶底分型标记
                    if (data.patterns && data.patterns.length > 0) {
                        // 由于markPoint容易引起渲染错误，我们改用单独的散点图来表示分型
                        const topPoints = [];
                        const bottomPoints = [];
                        
                        data.patterns.forEach(pattern => {
                            const patternDate = pattern.date;
                            const patternIndex = dateToIndex[patternDate];
                            
                            if (patternIndex !== undefined) {
                                // 根据分型类型获取相应的值
                                let displayValue = parseFloat(pattern.value);
                                let label = null;
                                
                                // 顶分型显示high，底分型显示low
                                if (pattern.type === 'top' && pattern.high) {
                                    label = pattern.high;
                                } else if (pattern.type === 'bottom' && pattern.low) {
                                    label = pattern.low;
                                }
                                
                                const pointData = {
                                    value: [patternDate, displayValue],
                                    itemStyle: {},
                                    label: {
                                        show: label !== null,
                                        position: pattern.type === 'top' ? 'top' : 'bottom',
                                        formatter: label !== null ? label : '',
                                        fontSize: 10,
                                        color: pattern.type === 'top' ? '#FF4136' : '#2ECC40',
                                        backgroundColor: 'rgba(255, 255, 255, 0.7)',
                                        padding: [2, 4],
                                        borderRadius: 3
                                    }
                                };
                                
                                if (pattern.type === 'top') {
                                    pointData.itemStyle.color = '#FF4136'; // 红色
                                    topPoints.push(pointData);
                                } else if (pattern.type === 'bottom') {
                                    pointData.itemStyle.color = '#2ECC40'; // 绿色
                                    bottomPoints.push(pointData);
                                }
                            }
                        });
                        
                        // 添加顶分型散点
                        if (topPoints.length > 0) {
                            option.series.push({
                                name: '顶分型',
                                type: 'scatter',
                                xAxisIndex: 1,
                                yAxisIndex: 1,
                                symbol: 'triangle',
                                symbolSize: 1,
                                symbolRotate: 180, // 向下的三角形
                                data: topPoints,
                                z: 11
                            });
                        }
                        
                        // 添加底分型散点
                        if (bottomPoints.length > 0) {
                            option.series.push({
                                name: '底分型',
                                type: 'scatter',
                                xAxisIndex: 1,
                                yAxisIndex: 1,
                                symbol: 'triangle',
                                symbolSize: 1,
                                data: bottomPoints,
                                z: 11
                            });
                        }
                        
                        // 更新图例添加顶底分型
                        if (!option.legend.data.includes('顶分型')) {
                            option.legend.data.push('顶分型');
                        }
                        if (!option.legend.data.includes('底分型')) {
                            option.legend.data.push('底分型');
                        }
                    }
                    
                    // 添加缠论笔
                    if (data.segments && data.segments.length > 0) {
                        // 为每个笔创建一个单独的折线系列，避免使用markLine
                        data.segments.forEach((segment, index) => {
                            const startDate = segment.start.date;
                            const endDate = segment.end.date;
                            const startValue = parseFloat(segment.start.value);
                            const endValue = parseFloat(segment.end.value);
                            
                            // 确保起点和终点的日期存在于数据中
                            if (dateToIndex[startDate] !== undefined && dateToIndex[endDate] !== undefined) {
                                // 创建一个单独的折线系列，只包含两个点（起点和终点）
                                option.series.push({
                                    name: '缠论笔',
                                    type: 'line',
                                    xAxisIndex: 1,
                                    yAxisIndex: 1,
                                    data: [
                                        [startDate, startValue],
                                        [endDate, endValue]
                                    ],
                                    showSymbol: false,  // 不显示点标记
                                    lineStyle: {
                                        color: '#0074D9',  // 蓝色
                                        width: 2
                                    },
                                    z: 12
                                });
                            }
                        });
                        
                        // 在图例中只显示一个缠论笔条目
                        if (!option.legend.data.includes('缠论笔')) {
                            option.legend.data.push('缠论笔');
                        }
                    }
                }
                
                // 添加中枢区域标记
                if (data.centralZones && data.centralZones.length > 0) {
                    // 处理每个中枢
                    data.centralZones.forEach(zone => {
                        // 获取日期索引
                        const startIndex = zone.start_index;
                        const endIndex = zone.end_index;
                        
                        // 找到对应的分型作为中枢的起点和终点
                        let startDate, startValue, endDate, endValue;
                        
                        // 获取中枢线段
                        const segments = zone.segments;
                        if (!segments || segments.length < 2) return;
                        
                        // 中枢起始线段（进入线）
                        const inLine = segments[segments.length - 1];
                        // 中枢结束线段（离开线）
                        const outLine = segments[0];
                        
                        // 中枢上下沿水平线
                        const zgValue = parseFloat(zone.zg);
                        const zdValue = parseFloat(zone.zd);
                        
                        // 确定中枢起点和终点的日期和坐标
                        if (inLine && inLine.start && inLine.end) {
                            const inStartDate = inLine.start.date;
                            const inEndDate = inLine.end.date;
                            const inStartValue = parseFloat(inLine.start.value);
                            const inEndValue = parseFloat(inLine.end.value);
                            
                            if (dateToIndex[inStartDate] !== undefined && dateToIndex[inEndDate] !== undefined) {
                                // 添加中枢进入线（加粗黄色）
                                option.series.push({
                                    name: '中枢进入线',
                                    type: 'line',
                                    xAxisIndex: 1,
                                    yAxisIndex: 1,
                                    data: [
                                        [inStartDate, inStartValue],
                                        [inEndDate, inEndValue]
                                    ],
                                    lineStyle: {
                                        color: '#FFD700',  // 黄色
                                        width: 3,
                                        type: 'solid'
                                    },
                                    symbol: 'circle',
                                    symbolSize: 6,
                                    z: 13
                                });
                            }
                        }
                        
                        if (outLine && outLine.start && outLine.end) {
                            const outStartDate = outLine.start.date;
                            const outEndDate = outLine.end.date;
                            const outStartValue = parseFloat(outLine.start.value);
                            const outEndValue = parseFloat(outLine.end.value);
                            
                            if (dateToIndex[outStartDate] !== undefined && dateToIndex[outEndDate] !== undefined) {
                                // 添加中枢离开线（加粗黄色）
                                option.series.push({
                                    name: '中枢离开线',
                                    type: 'line',
                                    xAxisIndex: 1,
                                    yAxisIndex: 1,
                                    data: [
                                        [outStartDate, outStartValue],
                                        [outEndDate, outEndValue]
                                    ],
                                    lineStyle: {
                                        color: '#FFD700',  // 黄色
                                        width: 3,
                                        type: 'solid'
                                    },
                                    symbol: 'circle',
                                    symbolSize: 6,
                                    z: 13
                                });
                            }
                        }
                        
                        // 获取中枢区域的开始和结束日期
                        let centralStartDate, centralEndDate;
                        
                        if (inLine && inLine.end && outLine && outLine.start) {
                            centralStartDate = inLine.end.date;
                            centralEndDate = outLine.start.date;
                            
                            if (dateToIndex[centralStartDate] !== undefined && 
                                dateToIndex[centralEndDate] !== undefined) {
                                
                                // 添加中枢区域上沿（红色水平线）
                                option.series.push({
                                    name: '中枢上沿',
                                    type: 'line',
                                    xAxisIndex: 1,
                                    yAxisIndex: 1,
                                    data: [
                                        [centralStartDate, zgValue],
                                        [centralEndDate, zgValue]
                                    ],
                                    lineStyle: {
                                        color: '#FF0000',  // 红色
                                        width: 2,
                                        type: 'dashed'
                                    },
                                    z: 13
                                });
                                
                                // 添加中枢区域下沿（红色水平线）
                                option.series.push({
                                    name: '中枢下沿',
                                    type: 'line',
                                    xAxisIndex: 1,
                                    yAxisIndex: 1,
                                    data: [
                                        [centralStartDate, zdValue],
                                        [centralEndDate, zdValue]
                                    ],
                                    lineStyle: {
                                        color: '#FF0000',  // 红色
                                        width: 2,
                                        type: 'dashed'
                                    },
                                    z: 13
                                });
                                
                                // 添加中枢区域填充色
                                option.series.push({
                                    name: '中枢区域',
                                    type: 'line',
                                    xAxisIndex: 1,
                                    yAxisIndex: 1,
                                    showSymbol: false,
                                    data: [
                                        // 形成一个矩形路径
                                        [centralStartDate, zgValue],
                                        [centralEndDate, zgValue],
                                        [centralEndDate, zdValue],
                                        [centralStartDate, zdValue],
                                        [centralStartDate, zgValue]
                                    ],
                                    areaStyle: {
                                        color: 'rgba(255, 0, 0, 0.1)'  // 淡红色填充
                                    },
                                    lineStyle: {
                                        width: 0  // 不显示边框线
                                    },
                                    z: 12
                                });
                            }
                        }
                    });
                    
                    // 更新图例
                    if (!option.legend.data.includes('中枢进入线')) {
                        option.legend.data.push('中枢进入线');
                    }
                    if (!option.legend.data.includes('中枢离开线')) {
                        option.legend.data.push('中枢离开线');
                    }
                    if (!option.legend.data.includes('中枢区域')) {
                        option.legend.data.push('中枢区域');
                    }
                }else
                {
                    // 完全清除之前的中枢相关series
                    const centralZoneSeriesNames = ['中枢进入线', '中枢离开线', '中枢上沿', '中枢下沿', '中枢区域'];
                    
                    // 首先从图例中移除
                    centralZoneSeriesNames.forEach(item => {
                        const index = option.legend.data.indexOf(item);
                        if (index > -1) {
                            option.legend.data.splice(index, 1);
                        }
                    });
                    
                    // 从series数组中过滤掉中枢相关的系列
                    option.series = option.series.filter(series => {
                        return !centralZoneSeriesNames.includes(series.name);
                    });
                }
                // 添加MACD指标
                if (data.macd_diff && data.macd_dea && data.macd_histogram) {
                    // 添加MACD DIFF线
                    option.series.push({
                        name: 'DIFF',
                        type: 'line',
                        xAxisIndex: 3,
                        yAxisIndex: 3,
                        data: data.macd_diff,
                        symbol: 'none',
                        lineStyle: {
                            width: 1.5,
                            color: '#FF6B6B'
                        }
                    });
                    
                    // 添加DEA线
                    option.series.push({
                        name: 'DEA',
                        type: 'line',
                        xAxisIndex: 3,
                        yAxisIndex: 3,
                        data: data.macd_dea,
                        symbol: 'none',
                        lineStyle: {
                            width: 1.5,
                            color: '#4CAF50'
                        }
                    });
                    
                    // 添加MACD柱状图
                    option.series.push({
                        name: 'MACD',
                        type: 'bar',
                        xAxisIndex: 3,
                        yAxisIndex: 3,
                        data: data.macd_histogram,
                        itemStyle: {
                            color: function(params) {
                                return params.data >= 0 ? '#FF6B6B' : '#4CAF50';
                            }
                        }
                    });
                    
                    // 更新图例
                    option.legend.data.push('DIFF', 'DEA', 'MACD');
                }

                klineChart.setOption(option);
            }

            function updateMoneyflowChart(data) {
                const option = {
                    tooltip: {
                        trigger: 'axis',
                        axisPointer: {
                            type: 'cross'
                        }
                    },
                    legend: {
                        data: ['大单买入', '大单卖出', '净流入']
                    },
                    grid: {
                        left: '3%',
                        right: '4%',
                        bottom: '15%',
                        containLabel: true
                    },
                    xAxis: {
                        type: 'category',
                        data: data.dates,
                        boundaryGap: false
                    },
                    yAxis: {
                        type: 'value',
                        splitLine: {
                            show: true
                        }
                    },
                    dataZoom: [{
                        type: 'inside',
                        start: 50,
                        end: 100
                    }, {
                        show: true,
                        type: 'slider',
                        bottom: '0',
                        start: 50,
                        end: 100
                    }],
                    series: [
                        {
                            name: '大单买入',
                            type: 'line',
                            data: data.buyFlow.map(v => v > 0 ? v : 0),
                            itemStyle: { 
                                color: '#ff4444'
                            },
                            areaStyle: {
                                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                                    offset: 0,
                                    color: 'rgba(255, 68, 68, 0.3)'
                                }, {
                                    offset: 1,
                                    color: 'rgba(255, 68, 68, 0.1)'
                                }])
                            },
                            smooth: true
                        },
                        {
                            name: '大单卖出',
                            type: 'line',
                            data: data.sellFlow.map(v => v > 0 ? v : 0),  // 将负值转为正值
                            itemStyle: { 
                                color: '#00aa00'
                            },
                            areaStyle: {
                                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                                    offset: 0,
                                    color: 'rgba(0, 170, 0, 0.3)'
                                }, {
                                    offset: 1,
                                    color: 'rgba(0, 170, 0, 0.1)'
                                }])
                            },
                            smooth: true
                        },
                        {
                            name: '净流入',
                            type: 'line',
                            data: data.netFlow,
                            itemStyle: { 
                                color: '#666666'
                            },
                            lineStyle: {
                                width: 2
                            },
                            smooth: true
                        }
                    ]
                };
                
                moneyflowChart.setOption(option);
            }

            function updateMacdChart(data) {
                const option = {
                    tooltip: {
                        trigger: 'axis',
                        axisPointer: {
                            type: 'cross'
                        }
                    },
                    legend: {
                        data: ['DIFF', 'DEA', 'MACD']
                    },
                    grid: {
                        left: '3%',
                        right: '4%',
                        bottom: '15%',
                        containLabel: true
                    },
                    xAxis: {
                        type: 'category',
                        data: data.dates,
                        boundaryGap: false
                    },
                    yAxis: {
                        type: 'value',
                        splitLine: {
                            show: true
                        }
                    },
                    dataZoom: [{
                        type: 'inside',
                        start: 50,
                        end: 100
                    }, {
                        show: true,
                        type: 'slider',
                        bottom: '0',
                        start: 50,
                        end: 100
                    }],
                    series: [
                        {
                            name: 'DIFF',
                            type: 'line',
                            data: data.diff,
                            itemStyle: { 
                                color: '#FF6B6B'
                            },
                            smooth: true
                        },
                        {
                            name: 'DEA',
                            type: 'line',
                            data: data.dea,
                            itemStyle: { 
                                color: '#4CAF50'
                            },
                            smooth: true
                        },
                        {
                            name: 'MACD',
                            type: 'bar',
                            data: data.histogram,
                            itemStyle: {
                                color: function(params) {
                                    return params.data >= 0 ? '#FF6B6B' : '#4CAF50';
                                }
                            }
                        }
                    ]
                };
                
                macdChart.setOption(option);
            }

            // 响应窗口大小变化
            window.addEventListener('resize', function() {
                klineChart.resize();
                moneyflowChart.resize();
                macdChart.resize();
            });
            
            // 在按下ESC键时退出全屏模式
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape' && document.getElementById('klineChart').classList.contains('fullscreen-chart')) {
                    document.getElementById('klineChart').classList.remove('fullscreen-chart');
                    $('#toggleFullscreen').text('切换全屏');
                    klineChart.resize();
                }
            });

            // 初始加载推荐股票和收藏股票
            loadRecommandStocks();
            loadCollectionStocks();

            // 点击推荐股票加载数据
            $(document).on('click', '.recommand-item, .collection-item', function() {
                const code = $(this).data('code');
                $('#stockInput').val(code);
                loadAllData(code);
            });

            // 如果URL中有code参数，自动加载数据
            const urlParams = new URLSearchParams(window.location.search);
            const stockCode = urlParams.get('code');
            if (stockCode) {
                loadAllData(stockCode);
            }
        });
    </script>
</body>
</html>
