# ec_2021
## 概要
EC2021のシミュレーション・プログラムとサンプルコードを配布しています．

``` bin/ ```にLinux, Mac, Windows用のシミュレーション・プログラムがあります．
example_sop.pyは単目的部門の最適化用サンプルコード(python)です．
example_mop.py: 多目的部門の最適化用サンプルコード(python)です．

## プログラムの実行方法
シミュレーション・プログラムは引数を含め，次のように実行します．プログラムファイル(syn_pop.exe, syn_pop)のあるディレクトリに申込後にダウンロード可能となる仮想合成人口データとシナリオファイルを配置します．

```
.
..
data
syn_pop
```
dataディレクトリ以下は次のファイルが配置されていることを確認してください．

```
$ ls -al data
 .
 ..
 hakodate_01.pkl
 hakodate_02.pkl
 hakodate_03.pkl
 naha_01.pkl
 naha_02.pkl
 naha_03.pkl
 scenario.csv
```
Linux, MacOSの場合は，syn_popに実行権限を付与して,以下のように実行してください．
```
./syn_pop [query] [payment] [function id] [city] [seeds]
```

Windowsの場合は，コマンドプロンプトから以下のように実行してください．
```
syn_pop.exe [query] [payment] [function id] [city] [seeds]
```

なお，それぞれの引数の意味と形式は下の表に整理しています．

| 引数 | 意味 | 値の例 |
| -------- | -------- | -------- |
| query    | 支給対象の条件．DataFrameに対するqueryを定義する文字列．"" で囲んでください．スペースの入れ方，カンマの入れ方は例のようにしてください．| "family_type_id == [0,3,4] and role_household_type_id == [0,1,10,11] and industry_type_id == [-1,130,160] and employment_type_id == [20,30] and company_size_id == [-1,5,10]" |
| payment | 給付金額．float型の金額(単位:万円)． | 7.5 |
| function id | 解(query, payment)を評価する目的関数のidのリスト.""で囲んでください． | "[1,2]" |
| city | 都市を表す文字列．naha, hakodateのいずれか． | naha |
| seeds |  失業や収入変化における不確実性に関わる乱数シードのリスト．"" で囲む． |"[123,42,256]" |

## プログラムの戻り値
プログラムの戻り値は，引数で指定した**目的関数の設定**により異なることから，注意が必要です．
"[1,2]" として$F_1, F_2$共に求めたとき（多目的）には，
```
[f_1, f_1_list, f_2, f_2_list, judge, slack_list]
```
のリストが返されます．このとき，それぞれ以下の意味になります．
- f_1: $F_1$の目的関数値
- f_1_list: 各条件で実行した生の$F_1$値（この平均が$F_1$になります）のリスト
- f_2: $F_2$の目的関数値
- f_2_list: 各条件で実行した生の$F_2$値（この平均が$F_2$になります）のリスト
- judge: 解が支給対象の制約条件を満たすかどうか（T/F）
- slack_list: 各条件で実行したときの制約の上限金額と支給金額との差のリスト．マイナスなら制約を満たしておらず，違反している金額を表す．

一方，"[2]" として$F_2$のみ求めたとき（単目的）には，
```
[f_2, f_2_list, judge, slack_list]
```
のリストが返されます．各条件で実行した生の$F_{1 or 2}$値は，都市セットid(city) -> シナリオid(scenario) -> rnの順番の組合せで，
```
city = [1,2,3]
scenario = [opt, mid, psm]
rn = [123, 42, 256]

for p in city:
    for s in scenario:
        for r in rn:
            run_simulation(p, s, r)
```
の順番に計算してリストに値を追加していきます．計算順(追加順)はslack_listも同様です．

## 実行例

参加者に提供されているローカルでのプログラムの実行例とその結果は以下のようになります．動作確認のためにご参考になさってください．

### Linux, MacOSの場合
実行例1(単目的の場合):
```
./syn_pop "family_type_id == [0,3,4,60,70,80] and role_household_type_id == [0,1,21,30,31] and industry_type_id == [-1,10,20,30,50,60,80,90,100,160,170,200] and employment_type_id == [-1,20,30] and company_size_id == [-1,5,10]" 9 "[2]" hakodate "[123,42,256]"

[-0.08909989731653375, [-0.08927674775332958, -0.08925003493289528, -0.08917500309701121, -0.089201944833592, -0.08920122485185668, -0.08907908501146895, -0.08913219158225016, -0.08920759066964075, -0.08913160190401953, -0.08925849126971337, -0.08919716843345353, -0.08921094159900611, -0.08910518939071901, -0.08898908096163473, -0.08907074851444122, -0.0890393583368047, -0.08905363819076062, -0.0890325279807537, -0.08913515589765733, -0.08916806598752336, -0.08912171652899231, -0.08901188533831822, -0.08895436396329265, -0.08898942590086098, -0.08888056539540183, -0.08893022571909716, -0.08889325350191615], True, [1159980000.0, 1160250000.0, 1161060000.0, 1156560000.0, 1156920000.0, 1157460000.0, 1155120000.0, 1155120000.0, 1155660000.0, 1161240000.0, 1161420000.0, 1161600000.0, 1157100000.0, 1157460000.0, 1157730000.0, 1155390000.0, 1155930000.0, 1155930000.0, 1178250000.0, 1176720000.0, 1177980000.0, 1175010000.0, 1173480000.0, 1175100000.0, 1173210000.0, 1172580000.0, 1173660000.0]]
```

実行例2(多目的の場合):
```
./syn_pop "family_type_id == [0,3,4,60,70,80] and role_household_type_id == [0,1,21,30,31] and industry_type_id == [-1,10,20,30,50,60,80,90,100,160,170,200] and employment_type_id == [-1,20,30] and company_size_id == [-1,5,10]" 9 "[1,2]" naha "[123,42,256]"

[-0.04075319685619771, [-0.042095110146575035, -0.04204271443334482, -0.04134024346149211, -0.04188400557764809, -0.04114043841839017, -0.04105839416058395, -0.04166093061674009, -0.041356749311294766, -0.04093436386483742, -0.040648021945791125, -0.04039619678694069, -0.04015177647464643, -0.04075543290789928, -0.03967144656692724, -0.03980399937887127, -0.04020490530891027, -0.03994754188883712, -0.03971362028810489, -0.04136470831253335, -0.04100957234350252, -0.040334658885503284, -0.040449786471965836, -0.0402141541427809, -0.04052798237764163, -0.04064172964040418, -0.04090729172044676, -0.040080539684725], -0.08829235069428736, [-0.08841258082864605, -0.08839533187677918, -0.08856850162862956, -0.08831020805107345, -0.08830616996394351, -0.08836263938482418, -0.08827287802555749, -0.08826136975651588, -0.0882715141331174, -0.08826628140722327, -0.08843495410293327, -0.08828670103644314, -0.0882337313362899, -0.08827002040130143, -0.08820053911750123, -0.08805238854651883, -0.08817956664248616, -0.08814265666361654, -0.08835223215791574, -0.08838228681934988, -0.08836571559593753, -0.08825255604124789, -0.08831975163763439, -0.0883166306232555, -0.08825234785828068, -0.08813497731089744, -0.08828893779783929], True, [2078100000.0, 2076300000.0, 2074590000.0, 2073510000.0, 2071440000.0, 2069370000.0, 2070810000.0, 2068920000.0, 2067210000.0, 2081520000.0, 2078730000.0, 2080350000.0, 2077200000.0, 2074320000.0, 2075040000.0, 2075220000.0, 2071890000.0, 2073330000.0, 2072790000.0, 2069190000.0, 2070000000.0, 2068020000.0, 2064600000.0, 2064510000.0, 2066130000.0, 2062350000.0, 2062350000.0]]
```

### Windowsの場合
実行例1(単目的の場合):
```
syn_pop.exe "family_type_id == [0,3,4,60,70,80] and role_household_type_id == [0,1,21,30,31] and industry_type_id == [-1,10,20,30,50,60,80,90,100,160,170,200] and employment_type_id == [-1,20,30] and company_size_id == [-1,5,10]" 9 "[2]" hakodate "[123,42,256]"

[-0.08909989731653375, [-0.08927674775332958, -0.08925003493289528, -0.08917500309701121, -0.089201944833592, -0.08920122485185668, -0.08907908501146895, -0.08913219158225016, -0.08920759066964075, -0.08913160190401953, -0.08925849126971337, -0.08919716843345353, -0.08921094159900611, -0.08910518939071901, -0.08898908096163473, -0.08907074851444122, -0.0890393583368047, -0.08905363819076062, -0.0890325279807537, -0.08913515589765733, -0.08916806598752336, -0.08912171652899231, -0.08901188533831822, -0.08895436396329265, -0.08898942590086098, -0.08888056539540183, -0.08893022571909716, -0.08889325350191615], True, [1159980000.0, 1160250000.0, 1161060000.0, 1156560000.0, 1156920000.0, 1157460000.0, 1155120000.0, 1155120000.0, 1155660000.0, 1161240000.0, 1161420000.0, 1161600000.0, 1157100000.0, 1157460000.0, 1157730000.0, 1155390000.0, 1155930000.0, 1155930000.0, 1178250000.0, 1176720000.0, 1177980000.0, 1175010000.0, 1173480000.0, 1175100000.0, 1173210000.0, 1172580000.0, 1173660000.0]]
```

実行例2(多目的の場合):
```
syn_pop.exe "family_type_id == [0,3,4,60,70,80] and role_household_type_id == [0,1,21,30,31] and industry_type_id == [-1,10,20,30,50,60,80,90,100,160,170,200] and employment_type_id == [-1,20,30] and company_size_id == [-1,5,10]" 9 "[1,2]" naha "[123,42,256]"

[-0.04075319685619771, [-0.042095110146575035, -0.04204271443334482, -0.04134024346149211, -0.04188400557764809, -0.04114043841839017, -0.04105839416058395, -0.04166093061674009, -0.041356749311294766, -0.04093436386483742, -0.040648021945791125, -0.04039619678694069, -0.04015177647464643, -0.04075543290789928, -0.03967144656692724, -0.03980399937887127, -0.04020490530891027, -0.03994754188883712, -0.03971362028810489, -0.04136470831253335, -0.04100957234350252, -0.040334658885503284, -0.040449786471965836, -0.0402141541427809, -0.04052798237764163, -0.04064172964040418, -0.04090729172044676, -0.040080539684725], -0.08829235069428736, [-0.08841258082864605, -0.08839533187677918, -0.08856850162862956, -0.08831020805107345, -0.08830616996394351, -0.08836263938482418, -0.08827287802555749, -0.08826136975651588, -0.0882715141331174, -0.08826628140722327, -0.08843495410293327, -0.08828670103644314, -0.0882337313362899, -0.08827002040130143, -0.08820053911750123, -0.08805238854651883, -0.08817956664248616, -0.08814265666361654, -0.08835223215791574, -0.08838228681934988, -0.08836571559593753, -0.08825255604124789, -0.08831975163763439, -0.0883166306232555, -0.08825234785828068, -0.08813497731089744, -0.08828893779783929], True, [2078100000.0, 2076300000.0, 2074590000.0, 2073510000.0, 2071440000.0, 2069370000.0, 2070810000.0, 2068920000.0, 2067210000.0, 2081520000.0, 2078730000.0, 2080350000.0, 2077200000.0, 2074320000.0, 2075040000.0, 2075220000.0, 2071890000.0, 2073330000.0, 2072790000.0, 2069190000.0, 2070000000.0, 2068020000.0, 2064600000.0, 2064510000.0, 2066130000.0, 2062350000.0, 2062350000.0]]
```