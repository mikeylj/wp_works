<?php
require_once './init_stock.php';
require_once './ChanTheory.php';

header('Content-Type: application/json');

$stockCode = isset($_GET['code']) ? $_GET['code'] : '';

if (empty($stockCode)) {
    echo json_encode(['error' => 'Stock code is required']);
    exit;
}

try {
    $sql = "SELECT 
                trade_date,
                open,
                high,
                low,
                close,
                close as  price,
                vol as volume,
                0 as dual_cannon,
                0 as yang_three_mas,
                0 as macd_signal
            FROM n_tradeinfo 
            WHERE ts_code like :code
            ORDER BY trade_date";

    $stmt = $pdo->prepare($sql);
    $stmt->execute(['code' => substr($stockCode, 0, 6) . '%']);
    
    $data = [];
    while ($row = $stmt->fetch(PDO::FETCH_ASSOC)) {
        $data[] = $row;
    }
    
    // Calculate MA values
    $ma5 = [];
    $ma10 = [];
    $ma20 = [];
    $ma30 = [];
    
    // Calculate MACD values
    $ema12 = [];
    $ema26 = [];
    $diff = [];
    $dea = [];
    $macd = [];
    
    for ($i = 0; $i < count($data); $i++) {
        // MA5
        if ($i >= 4) {
            $sum = 0;
            for ($j = 0; $j < 5; $j++) {
                $sum += floatval($data[$i - $j]['close']);
            }
            $ma5[] = round($sum / 5, 2);
        } else {
            $ma5[] = '-';
        }
        
        // MA10
        if ($i >= 9) {
            $sum = 0;
            for ($j = 0; $j < 10; $j++) {
                $sum += floatval($data[$i - $j]['close']);
            }
            $ma10[] = round($sum / 10, 2);
        } else {
            $ma10[] = '-';
        }
        
        // MA20
        if ($i >= 19) {
            $sum = 0;
            for ($j = 0; $j < 20; $j++) {
                $sum += floatval($data[$i - $j]['close']);
            }
            $ma20[] = round($sum / 20, 2);
        } else {
            $ma20[] = '-';
        }
        
        // MA30
        if ($i >= 29) {
            $sum = 0;
            for ($j = 0; $j < 30; $j++) {
                $sum += floatval($data[$i - $j]['close']);
            }
            $ma30[] = round($sum / 30, 2);
        } else {
            $ma30[] = '-';
        }
    }
    
    // Calculate MACD indicators
    for ($i = 0; $i < count($data); $i++) {
        $close = floatval($data[$i]['close']);
        
        // EMA12
        if ($i == 0) {
            $ema12[] = $close;
        } else {
            $ema12[] = round(($close * 2 / 13) + ($ema12[$i-1] * 11 / 13), 4);
        }
        
        // EMA26
        if ($i == 0) {
            $ema26[] = $close;
        } else {
            $ema26[] = round(($close * 2 / 27) + ($ema26[$i-1] * 25 / 27), 4);
        }
        
        // DIFF (MACD Line): EMA12 - EMA26
        $diff[] = round($ema12[$i] - $ema26[$i], 4);
        
        // DEA (Signal Line): 9-day EMA of DIFF
        if ($i == 0) {
            $dea[] = $diff[0];
        } else {
            $dea[] = round(($diff[$i] * 2 / 10) + ($dea[$i-1] * 8 / 10), 4);
        }
        
        // MACD (Histogram): (DIFF - DEA) * 2
        $macd[] = round(($diff[$i] - $dea[$i]) * 2, 4);
    }

    $chan = new ChanTheory($data);
    $chan->processKlinesInclusion();
    $chan_Process_Klines =  $chan->arr_ProcessIncludeKlines;
    $chan_ProcessDelMerged_Klines =  $chan->arr_ProcessIncludeDelMergedKlines;
    // var_dump($chan_Process_Klines, $chan_ProcessDelMerged_Klines);
    // $processed_klines = processInclusion($data);
    $processed_klines = $chan_Process_Klines ;

    // var_dump($chan_Process_Klines);exit;
    foreach($chan_Process_Klines as $key => $c_kline){
        if($c_kline['merged'] !== $processed_klines[$key]['merged'] ){
            // var_dump($processed_klines[$key]);
            // var_dump($c_kline);
            $chan-> errorLOG("KEY:{$key}\t DATE:{$c_kline['trade_date']}\t low:{$c_kline['low']} \t high:{$c_kline['high']}\t c_low:{$c_kline['c_low']}\t c_high:{$c_kline['c_high']}\t Inclusion:{$c_kline['inclusion']}\t Trend:{$c_kline['trend']}\t Merged:{$c_kline['merged']}\t\t \n" );
            $chan-> errorLOG( "KEY:{$key}\t Date:{$processed_klines[$key]['trade_date']}\t low:{$processed_klines[$key]['low']}\t high:{$processed_klines[$key]['high']}\t c_low :{$processed_klines[$key]['c_low']}\t c_high:{$processed_klines[$key]['c_high']}\t Inclusion:{$processed_klines[$key]['inclusion']}\t Trend:{$processed_klines[$key]['trend']}\t Merged:{$processed_klines[$key]['merged']}----------------\n\n");
        }
        else{
            $chan-> errorLOG( "KEY:{$key}\t DATE:{$c_kline['trade_date']}\t low:{$c_kline['low']} \t high:{$c_kline['high']}\t c_low:{$c_kline['c_low']}\t c_high:{$c_kline['c_high']}\t Inclusion:{$c_kline['inclusion']}\t Trend:{$c_kline['trend']}\t Merged:{$c_kline['merged']}\t\t \n" );
            $chan-> errorLOG( "KEY:{$key}\t Date:{$processed_klines[$key]['trade_date']}\t low:{$processed_klines[$key]['low']}\t high:{$processed_klines[$key]['high']}\t c_low:{$processed_klines[$key]['c_low']}\t c_high:{$processed_klines[$key]['c_high']}\t Inclusion:{$processed_klines[$key]['inclusion']}\t Trend:{$processed_klines[$key]['trend']}\t Merged:{$processed_klines[$key]['merged']}\n\n");

        }
    }
        
    // var_dump($processed_klines);exit;
    // $patterns_t = $findNewPatterns($processed_klines, $data);
    $chan-> findNewPatterns();
    $patterns_t = $chan->arr_Patterns;
    //处理
    $patterns = [];
    // var_dump($patterns_t);exit;
    foreach($patterns_t as $key => $curr){
        if(!empty($curr['type'])){
            $patterns[] = [
                'index' => $key,
                'type' => $curr['type'],
                'value' => ($curr['type'] == 'top' ? $curr['c_high'] : $curr['c_low']),
                'date' => $curr['trade_date'],
                'high' => $curr['c_high'],
                'low' => $curr['c_low'],
                'open' => $curr['open'],
                'close' => $curr['close']
            ];
    
        }
    }
    // $patterns = findPatterns($processed_klines, $data);
    // var_dump($patterns);


    $segments = findSegments($patterns, $processed_klines);
    $centralZones = findCentralZone($segments, $diff);

    echo json_encode([
        'klineData' => array_map(function($row) {
            return [
                $row['trade_date'],
                floatval($row['open']),
                floatval($row['close']),
                floatval($row['low']),
                floatval($row['high']),
                floatval($row['volume']),
                floatval($row['price']),
                floatval($row['dual_cannon']),
                floatval($row['yang_three_mas']),
                floatval($row['macd_signal'])
            ];
        }, $data),
        'ma5' => $ma5,
        'ma10' => $ma10,
        'ma20' => $ma20,
        'ma30' => $ma30,
        'macd_diff' => $diff,
        'macd_dea' => $dea,
        'macd_histogram' => $macd,
        'processed_klines' => $processed_klines,
        'patterns' => $patterns,
        'segments' => $segments,
        'centralZones' => $centralZones
    ]);
    
} catch (PDOException $e) {
    echo json_encode(['error' => 'Database error']);
}

/**
 * 处理K线包含关系 (基于缠论)
 * 
 * @param array $data 原始K线数据
 * @return array 处理后的K线数据
 */
function processInclusion($data) {
    $result = [];
    $i = 0;
    
    while ($i < count($data)) {
        $current = $data[$i];
        $current['c_index'] = $i;
        $current['c_high'] = $current['high'];
        $current['c_low'] = $current['low'];
        $current['merged'] = false;
        $current['inclusion'] = '';
        $current['trend'] = '';

        if ($i <= 1) {  // 前两根K线保持不变
            $result[] = $current;
            $i++;
            continue;
        }
        // 获取最近两个未被合并的K线
        // $nonMerged = array_filter($result, function($row) {
        //     return $row['merged'] === false;
        // });
        // $nonMerged = array_values($nonMerged);
        // $previous = $nonMerged[count($nonMerged) - 1];
        // $prevPrev = $nonMerged[count($nonMerged) - 2];
        //获取前两张K线
        $previous = $result[count($result) - 1];
        $prevPrev = $result[count($result) - 2];
        if ($prevPrev['high'] < $previous['low'] || $prevPrev['low'] > $previous['high']) // 如果跳空，直接保留
        {
            $result[] = $current;
            $i++;
            continue;
        }

        // 判断趋势
        $trend = (floatval($previous['high']) > floatval($prevPrev['high'])) ? 'up' : 'down';
        
        // 判断是否包含
        $hasInclusion = (
            (floatval($previous['c_high']) >= floatval($current['high']) && floatval($previous['c_low']) <= floatval($current['low'])) ||
            (floatval($previous['c_high']) <= floatval($current['high']) && floatval($previous['c_low']) >= floatval($current['low']))
        );

        $current['trend'] = $trend;
        $current['inclusion'] = $hasInclusion;

        if ($hasInclusion) {
            // 根据趋势方向合并K线
            if ($trend == 'up') {
                $newK = $current;
                $newK['c_high'] = max(floatval($previous['c_high']), floatval($current['high']));
                $newK['c_low'] = max(floatval($previous['c_low']), floatval($current['low']));

                // // 上涨趋势，取高点
                // $newK = [
                //     'trade_date' => $current['trade_date'],
                //     'open' => floatval($current['open']),
                //     'high' => max(floatval($previous['high']), floatval($current['high'])),
                //     'low' => max(floatval($previous['low']), floatval($current['low'])),
                //     'close' => floatval($current['close']),
                //     'volume' => floatval($current['volume']),
                //     'merged' => false
                // ];
            } else {
                // // 下跌趋势，取低点
                $newK = $current;
                $newK['c_high'] = min(floatval($previous['c_high']), floatval($current['high']));
                $newK['c_low'] = min(floatval($previous['c_low']), floatval($current['low']));
                // $newK = [
                //     'trade_date' => $current['trade_date'],
                //     'high' => min(floatval($previous['high']), floatval($current['high'])),
                //     'low' => min(floatval($previous['low']), floatval($current['low'])),
                //     'open' => floatval($current['open']),
                //     'close' => floatval($current['close']),
                //     'volume' => floatval($current['volume']),
                //     'merged' => false
                // ];
            }

            // if ($current['trade_date'] =='2025-01-14') {
            //     print_r($previous);
            //     print_r($prevPrev);
            //     print_r($newK);
            //     exit;
            // }
            $nonMerged = array_filter($result, function($row) {
                return $row['merged'] === false;
            });
            $nonMergedKeys = array_keys($nonMerged);
            $delKey = $nonMergedKeys[count($nonMergedKeys) - 1];
            $result[$delKey]['merged'] = true;
            $result[] = $newK;
        } else {
            $result[] = $current;
        }
        $i++;
    }
    return $result; 
    // // 提取未被合并的K线，并格式化为与klineData相同的格式
    // $processedKlines = [];
    // foreach ($result as $row) {
    //     if ($row['merged'] === false) {
    //         $processedKlines[] = [
    //             $row['trade_date'],
    //             isset($row['open']) ? floatval($row['open']) : 0,
    //             isset($row['close']) ? floatval($row['close']) : 0,
    //             isset($row['low']) ? floatval($row['low']) : 0,
    //             isset($row['high']) ? floatval($row['high']) : 0,
    //             isset($row['volume']) ? floatval($row['volume']) : 0,
    //         ];
    //     }
    // }
    
    // return $processedKlines;
}
//第一步：取得合并后的K线
function getProcessedKLines($processedKlines) {
    // 获取未被合并的K线
    $validKlines = [];
    foreach ($processedKlines as $index => $kline) {
        if ($kline['merged'] === false) {
                $validKlines[] = [
                    'index' => $index,
                    'data' => $kline
                ];
            }
        }
    return $validKlines;
}
//第二步：取得所有顶底分开
function getPatternStep1($validKlines){
    $patterns = [];
    // 识别顶底分型
    for ($i = 1; $i < count($validKlines) - 1; $i++) {
        $curr = $validKlines[$i];
        $prev = $validKlines[$i - 1];
        $next = $validKlines[$i + 1];
        
        // 顶分型：当前K线的高点高于前一个和后一个K线的高点
        if (floatval($curr['data']['c_high']) > floatval($prev['data']['c_high']) && 
            floatval($curr['data']['c_high']) > floatval($next['data']['c_high'])) {
            $patterns[] = [
                'index' => $curr['index'],
                'type' => 'top',
                'value' => $curr['data']['c_high'],
                'date' => $curr['data']['trade_date'],
                'high' => $curr['data']['c_high'],
                'low' => $curr['data']['c_low'],
                'open' => $curr['data']['open'],
                'close' => $curr['data']['close']
            ];
        }
        // 底分型：当前K线的低点低于前一个和后一个K线的低点
        else if (floatval($curr['data']['c_low']) < floatval($prev['data']['c_low']) && 
                 floatval($curr['data']['c_low']) < floatval($next['data']['c_low'])) {
            $patterns[] = [
                'index' => $curr['index'],
                'type' => 'bottom',
                'value' => $curr['data']['c_low'],
                'date' => $curr['data']['trade_date'],
                'high' => $curr['data']['c_high'],
                'low' => $curr['data']['c_low'],
                'open' => $curr['data']['open'],
                'close' => $curr['data']['close']
            ];
        }
    }
    return $patterns;
}
//第三步：删除相邻顶底分开
function delCloseSpacePatterns($patterns, $validKlines, $ori_Kdata){
    $deletePatterns = [];
    if (count($patterns) > 0) {
        $comparePattern = $patterns[0];
        
        for ($i = 1; $i < count($patterns); $i++) {
            $currPattern = $patterns[$i];
            // if($currPattern['date'] == '2024-10-08') var_dump("11111111111");

            // if($currPattern['date'] == '2024-10-08'){
            //     var_dump($currPattern, $comparePattern);exit;
            // }
            // 不同类型的分型（一个顶一个底）
            if ($currPattern['type'] != $comparePattern['type']) {
                // if($currPattern['date'] == '2024-10-08') var_dump("2222");

                // 计算两个分型之间的有效K线数量
                //取得compare、curr的index
                $comparePattern_index = array_search($comparePattern['index'], array_column($validKlines, 'index'));
                $currPattern_index = array_search($currPattern['index'], array_column($validKlines, 'index'));
                $kSize = $currPattern_index - $comparePattern_index;
                // if($currPattern['date'] == '2024-10-08') var_dump("{$kSize}");

                // if($currPattern['date'] == '2024-10-21'){
                //     var_dump($currPattern, $comparePattern);exit;

                // }

                // 过滤条件：两个分型之间的K线数量检查
                if ($kSize <= 3) {
                    $compareK_index = $comparePattern['index'];
                    $currK_index = $currPattern['index'];
                    $ori_kSize = $currK_index - $compareK_index;
                    $isdel = true;
                    for($ori_index = 0; $ori_index <= $ori_kSize - 1; $ori_index++){
                        $currK = $ori_Kdata[$currK_index - $ori_index];
                        $prevK = $ori_Kdata[$currK_index - $ori_index - 1];

                        // if($currK['trade_date'] == '2024-10-21'){
                        //     var_dump($currK, $prevK);exit;

                        // }
                        if (floatval($prevK['high']) < floatval($currK['low']) || 
                        floatval($prevK['low']) > floatval($currK['high'])) {
                            $isdel =false;
                            break;
                        }
                    }
                    if($isdel){
                        $deletePatterns[] = $currPattern;
                    }
                    else{
                        $comparePattern = $currPattern;
                    }


                    // // 检查是否有跳空
                    // $currIndex = array_search($currPattern['index'], array_column($validKlines, 'index'));
                    // $prevIndex = $currIndex - 1;
                    // $prevPrevIndex = $currIndex - 2;
                    // if( $kSize == 1 && $prevIndex >=0){
                    //     $currK = $validKlines[$currIndex]['data'];
                    //     $prevK = $validKlines[$prevIndex]['data'];
                        
                    //     if (floatval($prevK['c_high']) < floatval($currK['c_low']) || 
                    //         floatval($prevK['c_low']) > floatval($currK['c_high'])) {
                    //         $comparePattern = $currPattern;
                    //     } else {
                    //         $deletePatterns[] = $currPattern;
                    //     }
                    // }

                    // if (($kSize == 2 || $kSize == 3) &&  $prevIndex >= 0 && $prevPrevIndex >= 0) {
                    //     $currK = $validKlines[$currIndex]['data'];
                    //     $prevK = $validKlines[$prevIndex]['data'];
                    //     $prevPrevK = $validKlines[$prevPrevIndex]['data'];
                    //     if (floatval($prevK['c_high']) < floatval($currK['c_low']) || 
                    //         floatval($prevK['c_low']) > floatval($currK['c_high'])) {
                    //         $comparePattern = $currPattern;
                    //     } 
                    //     elseif (floatval($prevK['c_high']) < floatval($prevPrevK['c_low']) || 
                    //         floatval($prevK['c_low']) > floatval($prevPrevK['c_high'])) {
                    //         $comparePattern = $currPattern;
                    //     } else {
                    //         $deletePatterns[] = $currPattern;
                    //     }
                    // }
                } else {
                    $comparePattern = $currPattern;
                }
            } else {
                // 相同类型的分型比较
                if ($currPattern['type'] == 'top') {
                    if (floatval($currPattern['value']) > floatval($comparePattern['value'])) {
                        $deletePatterns[] = $comparePattern;
                        $comparePattern = $currPattern;
                    } else {
                        $deletePatterns[] = $currPattern;
                    }
                } else { // 底分型
                    if (floatval($currPattern['value']) < floatval($comparePattern['value'])) {
                        $deletePatterns[] = $comparePattern;
                        $comparePattern = $currPattern;
                    } else {
                        $deletePatterns[] = $currPattern;
                    }
                }
            }
        }
    }
    return $deletePatterns;
}

function getChangeVertexPatterns($patterns, $deletePatterns){
    // var_dump($patterns);
    for($i = 0; $i < count($patterns) - 1; $i++){
        $currPattern = $patterns[$i];
        $nextPattern = $patterns[$i + 1];
        $toSubDeletePatterns = [];
        $toAddDeletePatterns = [];
        // if($currPattern['date'] == '2024-05-24'){
        //     var_dump($currPattern);
        //     var_dump('-------------------');
        //     var_dump($nextPattern);
        //     // var_dump($deletePattern);
            
        // }
        foreach($deletePatterns as $deletePattern){
            // if($deletePattern['date'] == '2024-05-29'){
            //     var_dump('***********************');
            //     var_dump($currPattern);
            //     var_dump($nextPattern);
            //     var_dump($deletePattern);
            //     var_dump('***********************');

            // }

            if($currPattern["index"] < $deletePattern['index'] && $nextPattern['index'] > $deletePattern['index']){
                if($currPattern['type'] == 'top' && $deletePattern['type'] == 'top'){
                        if($deletePattern['high'] > $currPattern['high']){

                            // if($currPattern['date'] == '2024-05-06'){
                            //     var_dump($currPattern);
                            //     var_dump($nextPattern);
                                
                            //     var_dump($deletePattern);
                            // }    
                            $patterns[$i] = $deletePattern;
                            $currPattern = $deletePattern;
                        }
                    }
                    if($currPattern['type'] == 'bottom' && $deletePattern['type'] == 'bottom'){
                        if($deletePattern['low'] < $currPattern['low']){
                            // if($currPattern['date'] == '2024-05-06'){
                            //     var_dump($currPattern);
                            //     var_dump($nextPattern);
                                
                            //     var_dump($deletePattern);
                            // }    

                            $patterns[$i] = $deletePattern;
                            $currPattern = $deletePattern;
                        }
                    }
                    
                if($nextPattern['type'] == 'top' && $nextPattern['type'] == $deletePattern['type']){
                    
                    if($deletePattern['high'] > $nextPattern['high']){
                        $patterns[$i+1] = $deletePattern;
                        $nextPattern = $deletePattern;
                    }

                }
                if($nextPattern['type'] == 'bottom' && $nextPattern['type'] == $deletePattern['type']){
                    if($deletePattern['low'] < $nextPattern['low']){
                        $patterns[$i+1] = $deletePattern;
                        $nextPattern = $deletePattern;
                    }
                }

            }
        }
    }

    // for($i = 0; $i < count($toSubDeletePatterns); $i++){
    //     
    // }

    return $patterns;
}

/**
 * 识别顶底分型
 * @param array $processed_Klines 处理包含关系后的K线数据
 * @param array $ori_Kdata 处理包含关系前的K线数据
 * @return array 处理后的顶底分型数据
 */
function findNewPatterns($processed_Klines, $ori_Kdata)
{
    //循环得到第一个顶分型或底分型
    $patterns = [];
    $patterns_index = [];
    for($key = 0; $key < count($processed_Klines); $key++){
        $kline = $processed_Klines[$key];
        // if($kline['trade_date'] == '2025-04-07'){
        //     var_dump($kline);
        // }
        $kline['type'] = '';
        //第一条和最后一条K线，不可能是顶底分型
        if ($key < 1 || $key == count($processed_Klines) - 1){
            $patterns[] = $kline;
            continue;
        }
        if($kline['merged'] == true){
            $patterns[] = $kline;
            continue;
        }
        [$retDelPrevPattern_keys, $retBPattern] = checkTopPattern($key, $processed_Klines, $patterns_index, $patterns);
       
        // print_r([$kline['trade_date'], $key,  'top', $retDelPrevPattern_keys, $retBPattern]);

        if($retDelPrevPattern_keys != -1){
            $prevPattern_index = $patterns_index[$retDelPrevPattern_keys];
            unset($patterns_index[$retDelPrevPattern_keys]);
            $patterns[$prevPattern_index]['type'] = '';
        }
        //是顶分型
        if($retBPattern){
            $kline['type'] = 'top';
            $patterns_index[] = $key;    
            $patterns[] = $kline;
            continue;
        }
        [$retDelPrevPattern_keys, $retBPattern] = checkBottomPattern($key, $processed_Klines, $patterns_index, $patterns);
        if($retDelPrevPattern_keys != -1){
            $prevPattern_index = $patterns_index[$retDelPrevPattern_keys];
            unset($patterns_index[$retDelPrevPattern_keys]);
            $patterns[$prevPattern_index]['type'] = '';
        }
        //是底分型
        if($retBPattern){
            $kline['type'] = 'bottom';
            $patterns_index[] = $key;    
            $patterns[] = $kline;
            continue;
        }
        $patterns[] = $kline;
    }
    return $patterns;
}

function getDelMergedKlines( $processed_Klines){
    $klines = [];
    foreach($processed_Klines as $key => $val){
        if($val['merged'] == false){
            $klines[$key] = $val;
        }
    }
    return $klines;
}
function getPreviousKey(array $array, $currentKey) {
    $keys = array_keys($array);
    $index = array_search($currentKey, $keys, true); // 严格模式匹配键名
    
    if ($index === false || $index === 0) {
        return null; // 键不存在或已是第一个元素
    }
    
    return $keys[$index - 1]; // 返回前一个键
}

function getNextKey(array $array, $currentKey) {
    $keys = array_keys($array);
    $index = array_search($currentKey, $keys, true); // 严格模式匹配键名
    
    if ($index === false || $index === 0) {
        return null; // 键不存在或已是第一个元素
    }
    
    return $keys[$index + 1]; // 返回前一个键
}


/***
 * 
 * 
 * return : retDelPrevPattern_key 需删除index中的下标
 *              -1:不删除
 *              N:删除下标为N的分型
 *          retBPattern  返回是否为顶分型
 *              false:不是
 *              true:是
 */
function checkTopPattern($key, $processed_Klines, $patterns_index, $patterns ){
    $delMergedKlines = getDelMergedKlines($processed_Klines);
    
    $kline = $processed_Klines[$key];

    //循环得到第一个顶峰型
    $preKline = $processed_Klines[getPreviousKey($delMergedKlines, $key)];
    $nextKline = $processed_Klines[getNextKey($delMergedKlines, $key)];
    $prevPattern_key = getLastPatternIndex($patterns_index);    //分型的下标
    $retDelPrevPattern_keys = -1;
    $retBPattern = false;

    // 顶分型：当前K线的高点高于前一个和后一个K线的高点
    if (floatval($kline['c_high']) > floatval($preKline['c_high']) && 
        floatval($kline['c_high']) > floatval($nextKline['c_high'])) {
        $error_msg = "TOP_START:\t" . $kline['trade_date'] . "\t" . $kline['c_high'] . "\n";
        errorLOG($error_msg);
        if($prevPattern_key == -1){
            $retBPattern = true;
            $error_msg = "TOP_ADD:\t" . $kline['trade_date'] . "\t" . $kline['c_high'] . "\t第一个顶分型\n\n";
            errorLOG($error_msg);    
            return [$retDelPrevPattern_keys, $retBPattern];
        }
        
        $prevPattern_index = $patterns_index[$prevPattern_key];
        //取得上一个顶分型的kline
        $prevPattern_Kline = $patterns[$prevPattern_index];

        //上一个是顶分型，则对比，如果大小
        if($prevPattern_Kline['type'] == 'top'){
            //前一个比当前的大，当前这个不是顶分型
            if(floatval($prevPattern_Kline['c_high']) <= floatval($kline['c_high'])){
                $error_msg = "TOP_DELETE:\t" . $kline['trade_date'] . "\t" . $kline['c_high'] . "\t" . 'Delete:'. $prevPattern_Kline['trade_date'] . "\t" . $prevPattern_Kline['c_high'] . "\t Reason:较前一个顶分型大 \n";
                errorLOG($error_msg);
                $retDelPrevPattern_keys = $prevPattern_key;
                //前一个是底分型，当前的NextK线如果 < 底分型，则不构成顶分型
                $prevPrevPattern_key = getLastPatternIndex($patterns_index, 2);    //取上上个分型的下标
                if($prevPrevPattern_key != -1)
                {
                    $prevPrevPattern_index = $patterns_index[$prevPrevPattern_key];
                    //取得上一个顶分型的kline
                    $prevPrevPattern_Kline = $patterns[$prevPrevPattern_index];
                    if(floatval($nextKline['c_low']) < floatval($prevPrevPattern_Kline['c_low']) && $prevPrevPattern_Kline['type'] == "bottom"){
                        $error_msg = "TOP_CANCEL:\t" . $kline['trade_date'] . "\t" . $kline['c_high'] .  "\tFrom:" . $prevPrevPattern_Kline['trade_date'] . "\t" . $prevPrevPattern_Kline['c_low'] . "\t NEXT:" . "\t{$nextKline['trade_date']}\t{$nextKline['c_low']}\t" ."\tREASON:下一条K线小于上一个底分型 - TOP\n\n";
                        errorLOG($error_msg);
                    }
                    else{
                        $error_msg = "TOP_ADD:\t" . $kline['trade_date'] . "\t" . $kline['c_high'] . "\tREASON:大于前一个顶分型K线\n\n";
                        errorLOG($error_msg);
                        $retBPattern = true;
                    }
                }
                else{
                    $error_msg = "TOP_ADD:\t" . $kline['trade_date'] . "\t" . $kline['c_high'] . "\tREASON:大于前一个顶分型K线\n\n";
                    errorLOG($error_msg);
                    $retBPattern = true;
                }
            }
            else{

                $error_msg = "TOP_CANCEL:\t" . $kline['trade_date'] . "\t" . $kline['c_high'] . "\tREASON:未大过前一个顶分型\n\n";
                errorLOG($error_msg);
            }
        }
        else{
            //前一个是底分型，当前的NextK线如果 < 底分型，则不构成顶分型
            if(floatval($nextKline['c_low']) < floatval($prevPattern_Kline['c_low'])){
                $error_msg = "TOP_CANCEL:\t" . $kline['trade_date'] . "\t" . $kline['c_high'] .  "\tFrom:" . $prevPattern_Kline['trade_date'] . "\t" . $prevPattern_Kline['c_low'] . "\t NEXT:" . "\t{$nextKline['trade_date']}\t{$nextKline['c_low']}\t" ."\tREASON:下一条K线小于上一个底分型 - BOTTOM\n\n";
                errorLOG($error_msg);
            }
            else{
                //上一个是底分型，需要判断是否跳空，是否中单是5根K线，是否
                // $result = array_slice(原数组, 起始下标, 截取长度, [是否保留键名]);
                $patterns_slice = array_slice($processed_Klines, $prevPattern_index, $key - $prevPattern_index, false);
                $max_val = 0;
                $processed_kline_count = 0;
                $isDrop = false;
                // if($kline['trade_date'] == '2023-07-18'){
                //     var_dump($patterns_slice);exit;
                // }
                foreach($patterns_slice as $p_key => $p_val){
                    if($p_val['merged'] == false){
                        $processed_kline_count++;
                    }
                    $temp = floatval($p_val['c_high']) > floatval($p_val['high']) ? floatval($p_val['c_high']) : floatval($p_val['high']);
                    if($temp > $max_val){
                        $max_val = $temp;
                    }   
                }
                if($max_val > floatval($kline['c_high'])){
                    $error_msg = "TOP_CANCEL:\t" . $kline['trade_date'] . "\t" . $kline['c_high'] . "\tREASON:中间有大于当前值\n\n";
                    errorLOG($error_msg);
                    return [$retDelPrevPattern_keys, $retBPattern];
                }else{
                    if($processed_kline_count > 3){
                        $retBPattern = true;
                        $error_msg = "TOP_ADD:\t" . $kline['trade_date'] . "\t" . $kline['c_high'] . "\tFrom:" . $prevPattern_Kline['trade_date'] . "\tMAX_c_high" . $max_val . "\tREASON:中间K线大于3根\n\n";
                        errorLOG($error_msg);
                    }
                    else{
                        for($p_key = 0; $p_key < count($patterns_slice) - 1; $p_key++){
                            $p_val = $patterns_slice[$p_key];
                            $p_val_n = $patterns_slice[$p_key + 1];
                            if (floatval($p_val['high']) < floatval($p_val_n['low']) ||  
                                    floatval($p_val['low']) > floatval($p_val_n['high'])) {
                                $isDrop =true;
                            }
                        }

                        $bCheckDrop = floatval($kline['high']) < floatval($patterns_slice[count($patterns_slice) - 1]['low']) ||  floatval($kline['low']) > floatval($patterns_slice[count($patterns_slice) - 1]['high']);
                        if($isDrop || $bCheckDrop){
                            $error_msg = "TOP_ADD:\t，" . $kline['trade_date'] . "\t" . $kline['c_high'] . "\tREASON:跳空\n\n";
                            errorLOG($error_msg);
                            $retBPattern = true;
                        }
                        else{
                            $error_msg = "TOP_CANCEL:\t" . $kline['trade_date'] . "\t" . $kline['c_high'] . "\t From:" . $prevPattern_Kline['trade_date'] ."\tREASON:中间K线<4根\n\n";
                            errorLOG($error_msg);
                        }
                    }
                }

            }
        }

    }
    else{
        $error_msg = "TOP_END_NOTIN:\t" . $kline['trade_date'] . "\t" . $kline['c_high'] . "\tPRE:{$preKline['c_high']}\tNEXT:{$nextKline['c_high']}\n\n";
        // errorLOG($error_msg);
    }
    return [$retDelPrevPattern_keys, $retBPattern];

}
function errorLOG($error_msg){
    error_log(date('Y-m-d H:i:s') . "\t" . $error_msg, 3, ERROR_LOG_DIR);
}
/***
 * 
 * 
 * return : retDelPrevPattern_key 需删除index中的下标
 *              -1:不删除
 *              N:删除下标为N的分型
 *          retBPattern  返回是否为顶分型
 *              false:不是
 *              true:是
 */
function checkBottomPattern($key, $processed_Klines, $patterns_index, $patterns){
    $delMergedKlines = getDelMergedKlines($processed_Klines);
    $kline = $processed_Klines[$key];
    $preKline_key = getPreviousKey($delMergedKlines, $key);
    //循环得到第一个顶峰型
    $preKline = $processed_Klines[$preKline_key];
    $nextKline = $processed_Klines[getNextKey($delMergedKlines, $key)];
    $prevPattern_key = getLastPatternIndex($patterns_index);    //分型的下标
    $retDelPrevPattern_keys = -1;
    $retBPattern = false;
    // 顶分型：当前K线的高点高于前一个和后一个K线的高点
    if (floatval($kline['c_low']) < floatval($preKline['c_low']) && 
        floatval($kline['c_low']) < floatval($nextKline['c_low'])) {
        $error_msg = "BOTTOM_START:\t" . $kline['trade_date'] . "\t" . $kline['c_low'] . "\n";
        errorLOG($error_msg);

        if($prevPattern_key == -1){
            $error_msg = "BOTTOM_ADD:\t" . $kline['trade_date'] . "\t" . $kline['c_low'] . "\tREASON:第一个底分型\n\n";
            errorLOG($error_msg);    
            $retBPattern = true;
            return [$retDelPrevPattern_keys, $retBPattern];

        }
        $prevPattern_index = $patterns_index[$prevPattern_key];
        //取得上一个顶分型的kline
        $prevPattern_Kline = $patterns[$prevPattern_index];
        //上一个是顶分型，则对比，如果大小
        if($prevPattern_Kline['type'] == 'bottom'){
            //前一个比当前的大，当前这个不是顶分型
            if(floatval($prevPattern_Kline['c_low']) >= floatval($kline['c_low'])){
                $error_msg = "BOTTOM_DELETE:\t" .$kline['trade_date'] . "\t" . $kline['c_low'] . "\t前一K线：" . $prevPattern_Kline['trade_date'] . "\t" . $prevPattern_Kline['c_low'] . "\n";
                errorLOG($error_msg);    
                $retDelPrevPattern_keys = $prevPattern_key;

                //前一个是底分型，当前的NextK线如果 < 底分型，则不构成顶分型
                $prevPrevPattern_key = getLastPatternIndex($patterns_index, 2);    //取上上个分型的下标
                if($prevPrevPattern_key != -1)
                {
                    $prevPrevPattern_index = $patterns_index[$prevPrevPattern_key];
                    //取得上一个顶分型的kline
                    $prevPrevPattern_Kline = $patterns[$prevPrevPattern_index];
                    if(floatval($nextKline['c_high']) > floatval($prevPrevPattern_Kline['c_high']) && $prevPrevPattern_Kline['type'] == "top"){
                        $error_msg = "BOTTOM_CANCEL:\t" . $kline['trade_date'] . "\t" . $kline['c_low'] .  "\tFrom:" . $prevPrevPattern_Kline['trade_date'] . "\t" . $prevPrevPattern_Kline['c_high'] . "\t NEXT:" . "\t{$nextKline['trade_date']}\t{$nextKline['c_high']}\t" ."\tREASON:下一条K线大于上一个顶分型 - BOTTOM\n\n";
                        errorLOG($error_msg);
                    }
                    else{
                        $error_msg = "TOP_ADD:\t" . $kline['trade_date'] . "\t" . $kline['c_high'] . "\tREASON:大于前一个顶分型K线\n\n";
                        errorLOG($error_msg);
                        $retBPattern = true;
                    }
                }
                else{
                    $error_msg = "BOTTOM_ADD:\t" . $kline['trade_date'] . "\t" . $kline['c_low'] . "\tREASON:小于前一个底分型K线\n\n";
                    errorLOG($error_msg);    
                    $retBPattern = true;    
                }
            }
            else{
                $error_msg = "BOTTOM_CANCEL:\t" . $kline['trade_date'] . "\t" . $kline['c_low'] . "\t前一个顶分型：". $prevPattern_Kline['trade_date'] . ':' . $prevPattern_Kline['c_low'] . "\tREASON:未小于前一个底分型K线\n\n";
                errorLOG($error_msg);    
            }
        }
        else{
            //前一个是顶分型，当前的NextK线如果>顶分型，则不构成底分型
            if(floatval($nextKline['c_high']) > floatval($prevPattern_Kline['c_high'])){
                $error_msg = "BOTTOM_CANCEL:\t" . $kline['trade_date'] . "\t" . $kline['c_low'] .  "\tFrom:" . $prevPattern_Kline['trade_date'] . "\t" . $prevPattern_Kline['c_high'] . "\t NEXT:" . "\t{$nextKline['trade_date']}\t{$nextKline['c_high']}\t" ."\tREASON:下一条K线大于上一个顶分型 - TOP\n\n";
                errorLOG($error_msg);
            }
            else{
                //上一个是底分型，需要判断是否跳空，是否中单是5根K线，是否
                // $result = array_slice(原数组, 起始下标, 截取长度, [是否保留键名]);
                $patterns_slice = array_slice($processed_Klines, $prevPattern_index, $key - $prevPattern_index,false);
                // if($kline['trade_date'] == '2024-03-05')
                // {
                //     var_dump($patterns_slice);exit;
                // }

                $min_val = 0;
                $processed_kline_count = 0;
                $isDrop = false;
                foreach($patterns_slice as $p_key => $p_val){
                    if($p_val['merged'] == false){
                        $processed_kline_count++;
                    }
                    $temp = floatval($p_val['c_low']) > floatval($p_val['low']) ? floatval($p_val['c_low']) : floatval($p_val['low']);
                    if($min_val === 0){
                        $min_val = $temp;
                    }else{
                        if($temp < $min_val){
                            $min_val = $temp;
                        }       
                    }
                }
                // if($kline['trade_date'] == '2024-10-28'){
                //     $error_msg = "BOTTOM_CANCEL:---------------------" . $kline['trade_date'] . ' ' . $kline['c_low'] . " 中间最小值" . $min_val . "\n\n";
                //     errorLOG($error_msg);    
                // }
                if($min_val < floatval($kline['c_low'])){
                    $error_msg = "BOTTOM_CANCEL:\t" . $kline['trade_date'] . "\t" . $kline['c_low'] . "\tREASON:中间有小于底分开型K线\n\n";
                    errorLOG($error_msg);    
                    return [$retDelPrevPattern_keys, $retBPattern];
                }
                else{
                    if($processed_kline_count > 3){
                        $error_msg = "BOTTOM_ADD:\t" . $kline['trade_date'] . "\t" . $kline['c_low'] . "\tREASON:中间有4根K线\n\n";
                        errorLOG($error_msg);    
                        $retBPattern = true;
                    }
                    else{
                        for($p_key = 0; $p_key < count($patterns_slice) - 1; $p_key++){
                            $p_val = $patterns_slice[$p_key];
                            $p_val_n = $patterns_slice[$p_key + 1];
                            if (floatval($p_val['high']) < floatval($p_val_n['low']) ||  
                                    floatval($p_val['low']) > floatval($p_val_n['high'])) {
                                $isDrop =true;
                            }
                        }
                        $bCheckDrop = floatval($kline['high']) < floatval($patterns_slice[count($patterns_slice) - 1]['low']) ||  floatval($kline['low']) > floatval($patterns_slice[count($patterns_slice) - 1]['high']);
                        if($isDrop || $bCheckDrop){
                            $error_msg = "BOTTOM_ADD:\t" . $kline['trade_date'] . "\t" . $kline['c_low'] . "\tREASON:跳空\n\n";
                            errorLOG($error_msg);    
                            $retBPattern = true;
                        }
                        else{
                            $error_msg = "BOTTOM_CANCEL:\t" . $kline['trade_date'] . "\t" . $kline['c_low'] .  "\tFrom:" . $prevPattern_Kline['trade_date'] . "\tREASON:中间K线<4\n\n";
                            errorLOG($error_msg);  
                        }
                    }
                }
            }
        }
    }
    else{
        $error_msg = "BOTTOM_END_NOTIN:\t" . $kline['trade_date'] . "\t" . $kline['c_low'] . "\tPRE:{$preKline['c_low']}\tNEXT:{$nextKline['c_low']}\n\n";
        // errorLOG($error_msg);

    }
    return [$retDelPrevPattern_keys, $retBPattern];

}



//取得最后第$c个分型key
function getLastPatternIndex($patterns_index, $c = 1){
    if(count($patterns_index) < $c){
        return -1;
    }
    $keys = array_keys($patterns_index);
    return $keys[count($patterns_index) - $c]; // 最后一个元素
}

/**
 * 识别顶底分型
 * @param array $processedKlines 处理包含关系后的K线数据
 * @return array 顶底分型数据
 */
function findPatterns($processedKlines, $ori_Kdata) {
    $validKlines = getProcessedKLines($processedKlines);
    $patterns = getPatternStep1($validKlines);
    // $deletePatterns = delCloseSpacePatterns($patterns, $validKlines);
    // // var_dump($patterns);exit;
    // // 过滤分型
    if (count($patterns) > 0) {
        $delNum = 1;
        while($delNum > 0){
            $delNum = 0;
            $deletePatterns = delCloseSpacePatterns($patterns, $validKlines, $ori_Kdata);
            // var_dump($deletePatterns);
            if (count($deletePatterns) > 0){
                $patterns = delPatterns($deletePatterns, $patterns);
                $delNum = 1;
            }
            $newDeletePatterns = checkPrevDelPattern($deletePatterns, $patterns);
            if(count($newDeletePatterns) > 0){
                $patterns = delPatterns($newDeletePatterns, $patterns);
                $delNum = 1;
            }
            $allDeletePatterns = array_merge($deletePatterns, $newDeletePatterns);
            $patterns = getChangeVertexPatterns($patterns, $allDeletePatterns);
        }
    }
    return $patterns;
}

function delPatterns($delPatterns, $patterns) {
        // 移除需要删除的分型
    foreach ($delPatterns as $delPattern) {
            $key = array_search($delPattern, $patterns);
            if ($key !== false) {
                unset($patterns[$key]);
            }
    }
    $patterns = array_values($patterns);
    return $patterns;
}
//增加删除后一个pattern大于或小于前一个删除pattern
function checkPrevDelPattern($deletePatterns, $patterns){
    $newDeletePatterns = [];
    for($i = 1; $i < count($patterns); $i++){
        $currPattern = $patterns[$i];
        $prevPattern = $patterns[$i - 1];
        for($j = 0; $j < count($deletePatterns); $j++){
            if($deletePatterns[$j]['index']  > $currPattern['index']){
                break;
            }
            if($deletePatterns[$j]['index'] > $prevPattern['index'] && $deletePatterns[$j]['index'] < $currPattern['index']){
                if($deletePatterns[$j]['type'] == $currPattern['type'] ){
                    if($deletePatterns[$j]['type'] == 'top' && $deletePatterns[$j]['value'] > $currPattern['value']){
                        $newDeletePatterns[] = $patterns[$i];
                    }
                    if($deletePatterns[$j]['type'] == 'bottom' && $deletePatterns[$j]['value'] < $currPattern['value']){
                        $newDeletePatterns[] = $patterns[$i];
                    }
                }
            }
        }
    } 
    return $newDeletePatterns ;
// // var_dump($newDeletePatterns);exit;
//     foreach ($newDeletePatterns as $delPattern) {
//         $key = array_search($delPattern, $patterns);
//         if ($key !== false) {
//             // var_dump($key);
//             // var_dump($patterns[$key]);
//             unset($patterns[$key]);
//         }
//     }
//     $patterns = array_values($patterns);
//     return $patterns;
}



/**
 * 识别线段（笔）
 * @param array $patterns 顶底分型数据
 * @param array $processedKlines 处理包含关系后的K线数据
 * @return array 线段数据
 */
function findSegments($patterns, $processedKlines) {
    $segments = [];
    
    // 如果分型数量少于2个，无法构成笔
    if (count($patterns) < 2) {
        return $segments;
    }
    
    // 按索引（时间顺序）排序分型
    usort($patterns, function($a, $b) {
        return $a['index'] - $b['index'];
    });
    
    // 遍历分型，连接相邻的不同类型分型形成笔
    $i = 0;
    while ($i < count($patterns) - 1) {
        $startPattern = $patterns[$i];
        $endPattern = $patterns[$i + 1];
        
        // 检查笔的有效性
        if (isValidSegment($startPattern, $endPattern, $processedKlines)) {
            $segments[] = [
                'start' => [
                    'index' => $startPattern['index'],
                    'value' => $startPattern['value'],
                    'date' => $startPattern['date']
                ],
                'end' => [
                    'index' => $endPattern['index'],
                    'value' => $endPattern['value'],
                    'date' => $endPattern['date']
                ]
            ];
        }
        
        $i++;
    }
    
    return $segments;
}

/**
 * 检查是否构成有效的笔
 * @param array $startPattern 起始分型
 * @param array $endPattern 结束分型
 * @param array $processedKlines 处理后的K线数据
 * @return bool 是否是有效的笔
 */
function isValidSegment($startPattern, $endPattern, $processedKlines) {
    
    return true;
    
    $startIdx = $startPattern['index'];
    $endIdx = $endPattern['index'];
    
    // 检查是否是不同类型的分型（一个顶一个底）
    if ($startPattern['type'] === $endPattern['type']) {
        return false;
    }
    
    // 获取中间的未合并K线
    $middleKlines = [];
    for ($i = $startIdx + 1; $i < $endIdx; $i++) {
        if (isset($processedKlines[$i]) && $processedKlines[$i]['merged'] === false) {
            $middleKlines[] = $processedKlines[$i];
        }
    }
    
    // 检查中间是否有独立K线
    if (count($middleKlines) < 1) {
        return false;
    }
    
    // 检查方向
    if ($startPattern['type'] === 'bottom' && $endPattern['type'] === 'top') {
        // 上升笔：检查整体趋势是否上升
        if (floatval($startPattern['value']) >= floatval($endPattern['value'])) {
            return false;
        }
        
        // 检查中间K线的最低点是否低于起点
        $minLow = PHP_FLOAT_MAX;
        foreach ($middleKlines as $kline) {
            $minLow = min($minLow, floatval($kline['c_low']));
        }
        if ($minLow < floatval($startPattern['value'])) {
            return false;
        }
        
        return true;
    } else if ($startPattern['type'] === 'top' && $endPattern['type'] === 'bottom') {
        // 下降笔：检查整体趋势是否下降
        if (floatval($startPattern['value']) <= floatval($endPattern['value'])) {
            return false;
        }
        
        // 检查中间K线的最高点是否高于起点
        $maxHigh = 0;
        foreach ($middleKlines as $kline) {
            $maxHigh = max($maxHigh, floatval($kline['c_high']));
        }
        if ($maxHigh > floatval($startPattern['value'])) {
            return false;
        }
        
        return true;
    }
    
    return false;
}

/**
 * 检查是否形成中枢
 * @param array $segList 线段列表
 * @param array $macdData MACD数据
 * @return array [是否形成中枢, 中枢上沿, 中枢下沿, 进入DIFF最小值, 离开DIFF最小值]
 */
function checkCentralZone($segList, $macdData, $diff) {
    // 进入线和离开线
    $inLine = end($segList);
    $outLine = reset($segList);
    
    // 计算中枢区间
    $maxHigh = [];
    $minLow = [];
    
    // 只考虑中间的线段（不包括第一个和最后一个）
    for ($i = 1; $i < count($segList) - 1; $i++) {
        $startValue = floatval($segList[$i]['start']['value']);
        $endValue = floatval($segList[$i]['end']['value']);
        $maxHigh[] = max($startValue, $endValue);
        $minLow[] = min($startValue, $endValue);
    }
    
    // 如果中间没有足够的线段，无法形成中枢
    if (empty($maxHigh) || empty($minLow)) {
        return [false, 0, 0, 0, 0, 0, 0];
    }
    
    $zg = min(array_slice($maxHigh, -3)); // 中枢上沿 = 最高点的最小值
    $zd = max(array_slice($minLow, -3));  // 中枢下沿 = 最低点的最大值

    $zg_group = max($maxHigh);
    $zd_group = max($minLow);
    
    // 收集MACD数据
    $inStartIdx = $inLine['start']['index'];
    $inEndIdx = $inLine['end']['index'];
    $outStartIdx = $outLine['start']['index'];
    $outEndIdx = $outLine['end']['index'];
    
    // 进入段DIFF
    $inDiff = [];
    $minInDiff = NULL;
    $minInDiffIndex = NULL;
    $minInDiffLookForFlag = true;
    for ($i = min($inStartIdx, $inEndIdx); $i <= max($inStartIdx, $inEndIdx); $i++) {
        if (isset($diff[$i])) {
            $inDiff[] = $diff[$i];
            if(is_null($minInDiff)){
                $minInDiff =   $diff[$i];
                $minInDiffIndex = $i;
            }
            else{
                if($diff[$i] < $minInDiff ){
                    $minInDiff =   $diff[$i];
                    $minInDiffIndex = $i;
                }
            }
        }
    }
    
    // 出线DIFF
    $outDiff = [];
    for ($i = min($outStartIdx, $outEndIdx); $i <= max($outStartIdx, $outEndIdx); $i++) {
        if (isset($diff[$i])) {
            $outDiff[] = $diff[$i];
        }
    }
    
    // 中间区域DIFF
    $midDiff = [];
    for ($i = max($inStartIdx, $inEndIdx); $i <= min($outStartIdx, $outEndIdx); $i++) {
        if (isset($diff[$i])) {
            $midDiff[] = $diff[$i];
            //如果进入段，最小值还在延续
            if($minInDiffIndex == max($inStartIdx, $inEndIdx) && $minInDiffLookForFlag){
                if($minInDiff < $diff[$i]){
                    $minInDiff = $diff[$i];
                }
                else{
                    $minInDiffLookForFlag = false;
                }
            }
        }
    }
    
    // 如果没有足够的DIFF数据，无法判断
    if (empty($inDiff) || empty($outDiff) || empty($midDiff)) {
        return [false, 0, 0, 0, 0, 0, 0];
    }
    
    // $minInDiff = min($inDiff);
    $minOutDiff = min($outDiff);
    $maxOutDiff = max($outDiff);
    $minMidDiff = min($midDiff);
    $maxMidDiff = max($midDiff);
    $maxMidDiff = max($maxOutDiff, $maxMidDiff );
    
    // var_dump($minInDiff, $minOutDiff);exit;
    // 判断是否形成中枢
    if ($zg > $zd) {
        // 检查MACD条件
        if ((round($minInDiff, 2) <= round($minOutDiff, 2) && 
            (((round($minMidDiff, 2) < 0 && 
            round($maxMidDiff, 2) > 0)) || (round($maxMidDiff, 2) > -10)))) {
            return [true, $zg, $zd, round($minInDiff, 2), round($minOutDiff, 2), $zg_group, $zd_group];
        }
    }
    
    return [false, 0, 0, 0, 0, 0, 0];
}

/**
 * 识别中枢
 * @param array $segments 线段数据
 * @param array $macdData MACD数据
 * @return array 中枢数据
 */
function findCentralZone($segments, $diff) {
    $centralZones = [];
    
    // 如果线段不足3个，无法形成中枢
    if (count($segments) < 3) {
        return $centralZones;
    }
    
    $i = count($segments) - 1;
    while ($i >= 4) {
        // 1. 找到底分型
        $seg = $segments[$i];
        if (floatval($seg['start']['value']) < floatval($seg['end']['value'])) {
            $i--;
            continue;
        }
        
        // 2. 找出买入点1前的笔
        $segList = [$seg]; // 中枢的线段
        for ($j = $i - 1; $j > 0; $j--) {
            $tempSeg = $segments[$j];
            $minTemp = min(floatval($tempSeg['start']['value']), floatval($tempSeg['end']['value']));
            
            if (floatval($seg['end']['value']) > $minTemp) {
                break;
            }
            
            $segList[] = $tempSeg;
            
            if (count($segList) > 4 && 
                floatval($tempSeg['start']['value']) > floatval($tempSeg['end']['value'])) {
                // 3. 已达到4笔，可以结合MACD，判断是否形成中枢
                list($ret, $zg, $zd, $minInDiff, $minOutDiff, $zg_group, $zd_group) = checkCentralZone($segList, [], $diff);
                
                if ($ret) {
                    // 4. 加入中枢
                    $centralZones = [
                        'start_index' => $segList[count($segList) - 2]['start']['index'],
                        'end_index' => $segList[0]['start']['index'],
                        'zg' => $zg,
                        'zd' => $zd,
                        'min_in_diff' => $minInDiff,
                        'min_out_diff' => $minOutDiff,
                        'segments' => $segList,  // 保存构成中枢的线段
                        'zg_group'  => $zg_group,
                        'zd_group'  => $zd_group
                    ];
                    if(count($segList) > 9)
                        break;
                }
            }
        }
        
        if (!empty($centralZones)) {
            break;
        }
        
        $i--;
    }
    
    return [$centralZones];
}
