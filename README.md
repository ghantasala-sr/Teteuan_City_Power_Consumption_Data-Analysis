Teteuan City Power Consumption Data Analysis

Tetouan,‬‭ located‬‭ in‬‭ northern‬‭ Morocco‬‭ with‬‭ an‬‭ area‬‭ of‬‭ approximately‬‭ 10,375‬‭ km²
‬and‬‭ a‬‭ population‬‭ of‬‭ about‬‭ 550,374‬‭ as‬‭ per‬ the‬‭ 2014‬‭ census,‬‭ serves‬‭ as‬‭ the‬‭ case‬‭ study‬‭ area‬‭ [2].‬‭ The‬‭ city‬‭ experiences‬‭ mild,‬‭ rainy‬‭ winters‬‭ and‬‭ hot,‬‭ dry‬‭ summers,‬
‭ influencing‬‭ power‬‭ use‬‭ patterns.‬‭ Data‬‭ was‬‭ sourced‬‭ from‬‭ the‬‭ Supervisory‬‭ Control‬‭ and‬‭ Data‬‭ Acquisition‬‭ (SCADA)‬‭ system‬‭ of‬
‭ Amendis,‬‭ a‬‭ public‬‭ service‬‭ operator‬‭ distributing‬‭ electricity‬‭ since‬‭ 2002.‬‭ The‬‭ electricity‬‭ is‬‭ supplied‬‭ by‬‭ the‬‭ National‬‭ Office‬‭ of‬
‭ Electricity‬‭ and‬‭ Drinking‬‭ Water,‬‭ transformed‬‭ from‬‭ high‬‭ voltage‬‭ (63‬‭ kV)‬‭ to‬‭ medium‬‭ voltage‬‭ (20‬‭ kV),‬‭ and‬‭ distributed‬
‭ through‬‭ three‬‭ source‬‭ stations:‬‭ Quads,‬‭ Smir,‬‭ and‬‭ Boussafou.‬‭ The‬‭ dataset‬‭ [1]‬‭ spans‬‭ from‬‭ January‬‭ 1,‬‭ 2017,‬‭ to‬‭ December‬‭ 31,‬
‭ 2017,‬‭ recorded‬‭ every‬‭ 10‬‭ minutes,‬‭ with‬‭ no‬‭ missing‬‭ data,‬‭ and‬‭ includes‬‭ variables‬‭ like‬‭ date,‬‭ time,‬‭ and‬‭ consumption‬‭ for‬‭ each‬
‭ network.‬‭ Weather‬‭ data,‬‭ collected‬‭ every‬‭ 5‬‭ minutes‬‭ from‬‭ sensors‬‭ at‬‭ the‬‭ city’s‬‭ airport‬‭ and‬‭ the‬‭ Faculty‬‭ of‬‭ Science,‬‭ was‬
‭ resampled‬‭ to‬‭ 10-minute‬‭ intervals‬‭ by‬‭ averaging,‬‭ including‬‭ variables‬‭ like‬‭ temperature,‬‭ humidity,‬‭ wind‬‭ speed,‬‭ diffuse‬‭ flows,‬
‭ and‬‭ global‬‭ diffuse‬‭ flows.‬‭ Calendar‬‭ variables‬‭ such‬‭ as‬‭ month,‬‭ day,‬‭ hour,‬‭ and‬‭ week‬‭ were‬‭ also‬‭ analyzed‬‭ to‬‭ capture‬‭ seasonal‬
‭ and temporal effects.‬

The‬‭ efficient‬‭ management‬‭ and‬‭ accurate‬‭ prediction‬‭ of‬‭ electrical‬‭ energy‬‭ consumption‬‭ are‬‭ crucial‬‭ for‬‭ optimizing‬‭ power‬
‭ distribution‬‭ and‬‭ planning‬‭ in‬‭ urban‬‭ areas.‬‭ This‬‭ project‬‭ analyzes‬‭ the‬‭ Tetouan‬‭ City‬‭ Power‬‭ Consumption‬‭ dataset,‬
‭ encompassing‬‭ power‬‭ usage‬‭ data‬‭ across‬‭ three‬‭ distinct‬‭ zones‬‭ within‬‭ the‬‭ city.‬‭ Initially,‬‭ exploratory‬‭ data‬‭ analysis‬‭ (EDA)‬‭ was‬
‭ conducted‬‭ to‬‭ understand‬‭ data‬‭ distribution,‬‭ detect‬‭ patterns,‬‭ and‬‭ identify‬‭ any‬‭ anomalies‬‭ or‬‭ missing‬‭ values.‬‭ Subsequently,‬
‭ statistical‬‭ tests—including‬‭ independent‬‭ sample‬‭ t-tests‬‭ and‬‭ Analysis‬‭ of‬‭ Variance‬‭ (ANOV A)—were‬‭ employed‬‭ to‬‭ determine‬
‭ if‬‭ significant‬‭ differences‬‭ exist‬‭ among‬‭ the‬‭ zones'
‬‭ power‬‭ consumption‬‭ patterns.‬‭ Finally,‬‭ multiple‬‭ linear‬‭ regression‬‭ models‬
‭ were‬‭ developed‬‭ to‬‭ predict‬‭ power‬‭ consumption‬‭ accurately‬‭ in‬‭ each‬‭ zone,‬‭ facilitating‬‭ improved‬‭ energy‬‭ management‬
‭ strategies.‬
‭ At the end of this report you will be able to find out the :‬
‭ (i)‬‭ How‬‭ does‬‭ the‬‭ power‬‭ consumption‬‭ change‬‭ from‬‭ zone‬‭ to‬‭ zone,‬‭ correlation‬‭ between‬‭ the‬‭ features‬‭ of‬‭ the‬‭ physical‬‭ conditions‬
‭ affecting the consumption of the power in the city through various statistical testing analysis ?‬
‭ (ii)‬‭ Extracting‬‭ the‬‭ features‬‭ from‬‭ the‬‭ time-series‬‭ dataset‬‭ and‬‭ finding‬‭ out‬‭ the‬‭ hidden‬‭ patterns‬‭ between‬‭ new‬‭ features‬‭ ,‬
‭ predicting the power consumption based on the factors of each zone through the machine learning methodology?‬
‭ (iii)‬‭ Proving‬‭ the‬‭ relations‬‭ between‬‭ the‬‭ continuous‬‭ data‬‭ variables‬‭ (features)‬‭ using‬‭ hypothesis‬‭ tests‬‭ such‬‭ as‬‭ ANOVA‬‭ test,‬
‭ t-test (or) paired t-test, chi-square tests.‬
‭ (iv) Why is Zone 1 power consumption greater compared to Zone 2 and 3 power consumption ?‬
‭
‭
