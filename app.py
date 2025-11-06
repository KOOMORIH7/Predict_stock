import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

import sklearn.linear_model
import sklearn.model_selection
import yfinance
from PIL import Image
import yfinance as yf
from PIL.TiffImagePlugin import COPYRIGHT
from pygments.styles.dracula import green
# from pyparsing.diagram import template

st.title("AIで株価予測アプリ")
st.write('AIを使って、株価を予測してみましょう。')

#トップ画像の表示
image = Image.open('stock_predict.png')
st.image(image, width="stretch")

st.write('＊あくまでAIによる予測です（参考値）。コチラのアプリによる損害や損失は一切保証しかねます。')
st.header("株価銘柄のティッカーシンボルを入力してください。")
stock_name = st.text_input("例:AAPL,FB,SFTBY(大文字・小文字どちらでも可)", "AAPL")
stock_name = stock_name.upper()

link = 'https://search.sbisec.co.jp/v2/popwin/info/stock/pop6040_usequity_list.html'
st.markdown(link)
st.write('ティイッカーシンボルについては上のリンク（SBI証券）をご参照ください。')

df_stock = yf.download(stock_name, '2021-01-05')
if isinstance(df_stock.columns, pd.MultiIndex):
    df_stock.columns = [col[0] for col in df_stock.columns]
st.header(stock_name + "２０２２年１月５日から現在までの価格（USD）")
st.write(df_stock)

st.header(stock_name + "終値と１４日間の平均（USD）")
df_stock['SMA'] = df_stock['Close'].rolling(window=14).mean()
df_stock2 = df_stock[['Close', 'SMA']]
st.line_chart(df_stock2)

st.header(stock_name + "値動き（USD）")
df_stock['change'] = (((df_stock['Close'] - df_stock['Open'])) / (df_stock['Open']) * 100)
st.line_chart(df_stock['change'].tail(100))

fig = go.Figure(
    data=[go.Candlestick(
        x=df_stock.index,
        open=df_stock['Open'],
        high=df_stock['High'],
        low=df_stock['Low'],
        close=df_stock['Close'],
        increasing_line_color='green',
        decreasing_line_color='red', )
    ]
)

st.header(stock_name + "キャンドルスティック")
st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)

df_stock['label'] = df_stock['Close'].shift(-30)
st.header(stock_name + '1か月後を予想しよう（USD）')

try:
    def stock_predict():
        #機械学習（マシンラーニング）
        X = np.array(df_stock.drop(['label', 'SMA'], axis=1))
        X = sklearn.preprocessing.scale(X)
        predict_data = X[-30:]
        X = X[:-30]
        y = np.array(df_stock['label'])
        y = y[:-30]
        #データの分割
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.2)
        #訓練データを用いて学習する
        model = sklearn.linear_model.LinearRegression()
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        #小数点第一位で四捨五入
        st.write(f'正答率は{round((accuracy) * 100, 1)}%です。')

        #accurcyより信頼度を表示
        if accuracy > 0.75:
            st.write('信頼度：高')
        elif accuracy > 0.5:
            st.write('信頼度：中')
        else:
            st.write('信頼度：低')
        st.write('オレンジの線（Predict）が予想値です。')

        #検証データを用いて検証してみる
        predicted_data = model.predict(predict_data)
        df_stock['Predict'] = np.nan
        last_date = df_stock.iloc[-1].name
        one_day = 86400
        next_unix = last_date.timestamp() + one_day

        for pred in predicted_data:
            next_date = datetime.fromtimestamp(next_unix)
            next_unix += one_day
            new_row = pd.Series(np.nan, index=df_stock.columns)
            new_row['Predict'] = pred
            df_stock.loc[next_date] = new_row

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_stock.index, y=df_stock['Close'],mode='lines',name='Close',line=dict(color='green')))
        fig2.add_trace(go.Scatter(x=df_stock.index, y=df_stock['Predict'], mode='lines', name='Predict', line=dict(color='orange',dash='dot')))

        fig2.update_layout(title=f"{stock_name}予測結果", xaxis_title="日付",yaxis_title="株価(USD)",template="plotly_white")
        st.plotly_chart(fig2,use_container_width=True)


    #ボタンを押すとstock_predict()が発動
    if st.button('予測する'):
        stock_predict()


except:
    st.error(
        "エラーが起きているようです。"
    )
st.write('COPYRIGHT  © 2021 Tomoyuki Yoshikawa. ALL rights Reserved.')