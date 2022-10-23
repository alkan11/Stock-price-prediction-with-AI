from multiprocessing.sharedctypes import Value
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from plotly import graph_objs as go
import numpy as np
import joblib
import pickle 
from xgboost import XGBRegressor
import keras
from sklearn.preprocessing import MinMaxScaler
from pandas.io.formats.style import Styler
from st_aggrid import AgGrid,GridUpdateMode,JsCode
from st_aggrid import GridOptionsBuilder
import functools



if __name__ == '__main__':
   
    option = st.sidebar.selectbox("Hisse Senedi Seçiniz?", ('Ana Sayfa','Apple','Walmart','Ford', 'THY', 'Aselsan'),index=1)
    if option=='Ana Sayfa':
        st.subheader('Hisse Senedi Fiyat Tahmini')
        
        with st.expander('Yöntem ve Materyaller'):
            st.write('alkan')
        Kapanış=[]
        dfapple=pd.read_csv('C:\\Users\\alkan\\Desktop\\VSCODE\\Hisseler\\AAPL.csv')
        dfa=dfapple.copy()
        dfa.reset_index(inplace=True)
        Kapanış.append(np.float64(dfa.Close[-1:].values))
        dfWmt=pd.read_csv('C:\\Users\\alkan\\Desktop\\VSCODE\\Hisseler\\WMT.csv')
        dfw=dfWmt.copy()
        dfw.reset_index(inplace=True)
        Kapanış.append(np.float64(dfw.Close[-1:].values))
        dfaselsan=pd.read_csv('C:\\Users\\alkan\\Desktop\\VSCODE\\Hisseler\\ASELS.IS.csv')
        dfas=dfaselsan.copy()
        dfas.reset_index(inplace=True)
        Kapanış.append(np.float64(dfas.Close[-1:].values))
        dfF=pd.read_csv('C:\\Users\\alkan\\Desktop\\VSCODE\\Hisseler\\F.csv')
        dff=dfF.copy()
        dff.reset_index(inplace=True)
        Kapanış.append(np.float64(dff.Close[-1:].values))
        dfThy=pd.read_csv('C:\\Users\\alkan\\Desktop\\VSCODE\\Hisseler\\THYAO.IS.csv')
        dft=dfThy.copy()
        dft.reset_index(inplace=True)
        Kapanış.append(np.float64(dft.Close[-1:].values))

        hisse=[dfapple,dfWmt,dfaselsan,dfF,dfThy]
        agrlk=['C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\KNN\\KNN_Apple.joblib','C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\XGB\\XGB2_Apple_model','C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\LSTM\\LSTM_apple.h5','CAT_model_dosyasi.joblib','C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\RF\\RF_Apple_model.joblib',
        'C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\KNN\\KNN_WMT.joblib','C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\XGB\\XGB2_WMT_model','C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\LSTM\\LSTM_Walmart.h5','CATWMT_model_dosyasi.joblib','C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\RF\\RF_WMT_model.joblib',
        'C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\KNN\\KNN_Aselsan.joblib','C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\XGB\\XGB2_Aselsan_model','C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\LSTM\\LSTM_Aselsan.h5','CAT2_Aselsan_model_dosyasi.joblib','C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\RF\\RF_Aselsan_model.joblib',
        'C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\KNN\\KNN_F.joblib','C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\XGB\\XGB2_F_model','C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\LSTM\\LSTM_F.h5','CAT2_F_model_dosyasi.joblib','C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\RF\\RF_F_model.joblib',
        'C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\KNN\\KNN_THY.joblib','C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\XGB\\XGB2_THY_model','C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\LSTM\\LSTM_THY.h5','CAT_model_dosyasi.joblib','C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\RF\\RF_THY_model.joblib']
        KNN_pred=[]
        XGB_pred=[]
        LSTM_pred=[]
        CATB_pred=[]
        RF_pred=[]
        y=[]
        def modeller(hisse,j):

                    model=joblib.load(agrlk[j])
                    day=[[len(hisse)+1]]    
                    y_pred=model.predict(day)
                    y_pred= np.float64(y_pred)
                    KNN_pred.append(y_pred)
                    
                    filename=agrlk[j+1]
                    modelXGB=pickle.load(open(filename,'rb'))
                    x=[]
                    x=filename.split("_", 1)
                    if x[0]=='XGB2':
                        x=np.array(hisse.Close[-30:])
                        x=x.reshape((1,30))
                    else:
                        x=np.array(hisse.Close[-30:])
                        x=x.reshape((1,30))
                    y_predXgb=modelXGB.predict(x)
                    y_predXgb= np.float64(y_predXgb)
                    XGB_pred.append(y_predXgb)

                    new_model=keras.models.load_model(agrlk[j+2])
                    new_model.summary()
                    dfc=hisse.copy()
                    dfc=dfc.drop(['Date','Close','Volume','Adj Close'],axis=1)
                    pred=dfc[-1:]
                    pred=np.array(pred)
                    pred=np.reshape(pred,(3,1))
                    s=MinMaxScaler(feature_range=(0,1))
                    s_pred=s.fit_transform(pred)
                    s_pred=np.reshape(s_pred,(1,3,1))
                    y_preds=new_model.predict(s_pred)
                    y_predLSTM=s.inverse_transform(y_preds)
                    y_predLSTM= np.float64(y_predLSTM)
                    LSTM_pred.append(y_predLSTM)

                    filenamecat=agrlk[j+3]
                    modelCat=joblib.load(filenamecat)
                    x=[]
                    x=filenamecat.split("_", 1)
                    if x[0]=='CAT2':
                        c2=np.array(hisse.Close[-15:])
                        c2=c2.reshape((1,15))
                        c2_pred=modelCat.predict(c2)
                        c2_pred=np.float64(c2_pred)
                        CATB_pred.append(c2_pred)
                        y.append(1)
                    else:
                        y.append(2)
                        dfc=hisse.copy()
                        dfc=dfc.drop(['Date','Close','Volume','Adj Close'],axis=1)
                        pred=dfc[-1:]
                        pred=np.array(pred)
                        pred=np.reshape(pred,(3,1))
                        s=MinMaxScaler(feature_range=(0,1))
                        s_pred=s.fit_transform(pred)
                        s_pred=s_pred.reshape((1,3))
                        y_preds=modelCat.predict(s_pred) 
                        y_preds=np.reshape(y_preds,(y_preds.shape[0],1))
                        y_preds=s.inverse_transform(y_preds)
                        y_predCat= np.float64(y_preds)
                        CATB_pred.append(y_predCat)

                    modelR=joblib.load(agrlk[j+4])
                    day=[[len(hisse)+1]]    
                    y_pred=modelR.predict(day)
                    y_pred= np.float64(y_pred)
                    RF_pred.append(y_pred)
        j=0       
        for i in range(5):
                modeller(hisse[i],j)
                j=j+5
        algo=[RF_pred,KNN_pred,XGB_pred,CATB_pred,LSTM_pred]
        ortalama=[]
        ort_change=[]
        for i in range(5):
            ortalama.append(round(((RF_pred[i]+KNN_pred[i]+XGB_pred[i]+CATB_pred[i]+LSTM_pred[i])/5),3))
        for i in range(5):
            ort_change.append(round(((RF_pred[i]-Kapanış[i])+(KNN_pred[i]-Kapanış[i])+(XGB_pred[i]-Kapanış[i])+(CATB_pred[i]-Kapanış[i])+(LSTM_pred[i]-Kapanış[i])/5),3)) 
        frame={'Name':['Apple','Walmart','Aselsan','Ford','THY'],'Symbol':['APPL','WMT','ASELS.IS','F','THYAO'],
        'Kapanış':[Kapanış[0],Kapanış[1],Kapanış[2],Kapanış[3],Kapanış[4]],'RF Prediction':[RF_pred[0],RF_pred[1],RF_pred[2],RF_pred[3],RF_pred[4]],'RF Change Price':[round(RF_pred[0]-Kapanış[0],3),round(RF_pred[1]-Kapanış[1],3),round(RF_pred[2]-Kapanış[2],3),round(RF_pred[3]-Kapanış[3],3),round(RF_pred[4]-Kapanış[4],3)],'KNN':[KNN_pred[0],KNN_pred[1],KNN_pred[2],KNN_pred[3],KNN_pred[4]],'KNN Change Price':[round(KNN_pred[0]-Kapanış[0],3),round(KNN_pred[1]-Kapanış[1],3),round(KNN_pred[2]-Kapanış[2],3),round(KNN_pred[3]-Kapanış[3],3),round(KNN_pred[4]-Kapanış[4],3)],'XGB':[XGB_pred[0],XGB_pred[1],XGB_pred[2],XGB_pred[3],XGB_pred[4]],'XGB Change Price':[round(XGB_pred[0]-Kapanış[0],3),round(XGB_pred[1]-Kapanış[1],3),round(XGB_pred[2]-Kapanış[2],3),round(XGB_pred[3]-Kapanış[3],3),round(XGB_pred[4]-Kapanış[4],3)],'CATB':[CATB_pred[0],CATB_pred[1],CATB_pred[2],CATB_pred[3],CATB_pred[4]],'CATB Change Price':[round(CATB_pred[0]-Kapanış[0],3),round(CATB_pred[1]-Kapanış[1],3),round(CATB_pred[2]-Kapanış[2],3),round(CATB_pred[3]-Kapanış[3],3),round(CATB_pred[4]-Kapanış[4],3)],
        'LSTM':[LSTM_pred[0],LSTM_pred[1],LSTM_pred[2],LSTM_pred[3],LSTM_pred[4]],'LSTM Change Price':[round(LSTM_pred[0]-Kapanış[0],3),round(LSTM_pred[1]-Kapanış[1],3),round(LSTM_pred[2]-Kapanış[2],3),round(LSTM_pred[3]-Kapanış[3],3),round(LSTM_pred[4]-Kapanış[4],3)],
        'Risk':[0.15,0.30,0.45,0.11,0.08],'Ortalama':[ortalama[0],ortalama[1],ortalama[2],ortalama[3],ortalama[4]],'Ortalama_Change_Price':[ort_change[0],ort_change[1],ort_change[2],ort_change[3],ort_change[4]]}
        df=pd.DataFrame(data=frame)
        @st.cache(suppress_st_warning=True)
        def plot_data():
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=dfapple['Date'],y=dfapple['Close'],name='Apple'))
            fig.add_trace(go.Scatter(x=dfF['Date'],y=dfF['Close'],name='Ford'))
            fig.add_trace(go.Scatter(x=dfWmt['Date'],y=dfWmt['Close'],name='Walmart'))
            fig.add_trace(go.Scatter(x=dfaselsan['Date'],y=dfaselsan['Close'],name='Aselsan'))
            fig.add_trace(go.Scatter(x=dfThy['Date'],y=dfThy['Close'],name='Thy'))
            fig.layout.update(title_text='Hisse Grafikleri', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig,xaxis_rangeslider_visible=True)
        plot_data()    

        cellsytle_jscode = JsCode(
        """
    function(params) {
        if (params.value > 0) {
            return {
                'color': 'white',
                'backgroundColor': 'green'
            }
        } else if (params.value <0) {
            return {
                'color': 'white',
                'backgroundColor': 'crimson'
            }
        } else {
            return {
                'color': 'white',
                'backgroundColor': 'slategray'
            }
        }
    };
    """
    
    )
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_columns(
        (
            "RF Change Price",
            "KNN Change Price",
            'XGB Change Price',
            'CATB Change Price',
            'LSTM Change Price',
            'Ortalama_Change_Price',
            'Kapanış',
        ),
        cellStyle=cellsytle_jscode,
    )
        COMMON_ARGS = {
        "color": "symbol",
        "color_discrete_sequence": px.colors.sequential.Greens,
        "hover_data": [
        "Name",
        "Kapanış",
        "RF","KNN","XGB",'CATB','LSTM','Risk','Ortalama_Change_Price'
    ],
}       
        gb.configure_pagination()
        gb.configure_columns(("Name", "Symbol"), pinned=True)
        gridOptions = gb.build()
        chart = functools.partial(st.plotly_chart, use_container_width=True)
        AgGrid(df, gridOptions=gridOptions, allow_unsafe_jscode=True,height=250)

        st.subheader(f"Value of Account")
        total =df[:3]
        i=0
        for column, row in zip(st.columns(len(total)), total.itertuples()):
            column.metric(
            row.Name,
            f"${row.Kapanış:.2f}",
            f"${row.Ortalama_Change_Price:.2f}",
            )
        totals=df[3:]
        for columnt, rows in zip(st.columns(len(totals)), totals.itertuples()):
            columnt.metric(
            rows.Name,
            f"${rows.Kapanış:.2f}",
            f"${rows.Ortalama_Change_Price:.2f}"
            )  
            
            
        
    if option=='Apple':
        st.subheader('Apple Stock Price Prediction')
        with st.expander('Yöntem ve Materyaller'):
            st.write('alkan')
        dfapple=pd.read_csv('C:\\Users\\alkan\\Desktop\\VSCODE\\Hisseler\\AAPL.csv')
        gb = GridOptionsBuilder.from_dataframe(dfapple)
        gb.configure_pagination()
        gridOptions = gb.build()
        chart = functools.partial(st.plotly_chart, use_container_width=True)
        AgGrid(dfapple, gridOptions=gridOptions, allow_unsafe_jscode=True,height=250)
        
        import math
        @st.cache(suppress_st_warning=True)
        def CATB_plot_data():
            modelCat=joblib.load('CAT_model_dosyasi.joblib')
            dfk=dfapple.copy()
            close=dfk.Close[-1:]
            cut=math.ceil(len(dfk)*(.85))
            trainx=dfk.drop(['Close','Adj Close','Date','Volume'],axis=1)
            trainy=dfk.drop(['High','Low','Open','Volume','Adj Close','Date'],axis=1)
            x_test=trainx[cut:]
            pred=x_test[-1:]
            x_test=x_test[:-1]
            col=x_test.columns.to_list()
            col=['High','Low','Open']
            x_test=x_test[col]
            x_test=np.array(x_test)
            x_test=np.reshape(x_test,(656,3))
            scale=MinMaxScaler(feature_range=(0,1))
            scaled_x=scale.fit_transform(x_test)
            y_test=trainy[cut:]
            y_test=y_test[:-1]
            y_test=np.array(y_test)
            scaled_y=scale.fit_transform(y_test)
            y_preds=modelCat.predict(scaled_x)
            y_preds=np.reshape(y_preds,(y_preds.shape[0],1))
            y_preds=scale.inverse_transform(y_preds)
            grafik=dfk[:cut]
            valid=dfk[cut:-1]
            valid['y_pred']=y_preds
            fig=plt.figure(figsize=(8,6))
            plt.title('CATB Model')
            plt.plot(dfk.Close)
            plt.xlabel('date')
            plt.ylabel('Close Price USD')
            plt.plot(grafik['Close'])
            plt.plot(valid[['Close','y_pred']])
            plt.show()
            st.plotly_chart(fig,use_container_width=True)
            pred=np.array(pred)
            pred=np.reshape(pred,(3,1))
            s=MinMaxScaler(feature_range=(0,1))
            s_pred=s.fit_transform(pred)
            s_pred=s_pred.reshape((1,3))
            y_preds=modelCat.predict(s_pred) 
            y_preds=np.reshape(y_preds,(y_preds.shape[0],1))
            y_preds=s.inverse_transform(y_preds)
            y_predCat= np.float64(y_preds)
            st.metric(
               label="CATB Tahmin", value=np.round(y_predCat,3), delta=(np.round(np.float64(y_predCat-close),3))
            )
        
        def XGB_plot_data():
            filename='C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\XGB\\XGB2_Apple_model'
            modelXGB=pickle.load(open(filename,'rb'))
            dfk=dfapple.copy()
            close=dfk.Close[-1:]
            x_train=[]
            y_train=[]
            for i in range(30,len(dfk)):
                    x_train.append(dfk[i-30:i]['Close'])
                    y_train.append(dfk.Close[i])
            cut=math.ceil(len(dfk)*(.80))
            x_test=x_train[cut:]
            y_test=y_train[cut:]
            x_test,y_test=np.array(x_test),np.array(y_test)
            y_test=np.reshape(y_test,(y_test.shape[0],1))
            y_pred=modelXGB.predict(x_test)
            y_pred=np.reshape(y_pred,(y_pred.shape[0],1))
            grafik=dfk[:cut]
            index=dfk[cut:-30]
            index=index.set_index(pd.Index(np.arange(3530,(3530+846),1)))
            index['y_pred']=y_pred
            fig=plt.figure(figsize=(8,6))
            plt.title('XGB Model')
            plt.plot(dfk.Close)
            plt.xlabel('date')
            plt.ylabel('Close Price USD')
            plt.plot(index['y_pred'])
            plt.show()
            st.plotly_chart(fig,use_container_width=True)
            x=np.array(dfk.Close[-30:])
            x=x.reshape((1,30))
            y_predXgb=modelXGB.predict(x)
            y_predXgb=np.float64(y_predXgb)
            st.metric(
               label="XGB Tahmin", value=np.round(y_predXgb,3), delta=(np.round(np.float64(y_predXgb-close),3))
            )

        def LSTM_plot_data():
            new_model=keras.models.load_model('C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\LSTM\\LSTM_apple.h5')
            dfc=dfapple.copy()
            close=dfc.Close[-1:]
            dfc=dfc[:-1]
            cut=math.ceil(len(dfc)*(.85))
            test=dfc.iloc[cut:,:]
            x_test=test.drop(['Close','Adj Close','Date','Volume'],axis=1)
            y_test=test.drop(['High','Low','Open','Adj Close','Date','Volume'],axis=1)
            deneme=x_test[-1:]
            x_test=x_test[:-1]
            y_test=y_test[1:]
            x_test=x_test.values
            y_test=y_test.values
            s=MinMaxScaler(feature_range=(0,1))
            s_pred=s.fit_transform(x_test)
            Y=s.fit_transform(y_test)
            s_pred=s_pred.reshape((x_test.shape[0],3,1))
            y_pred=new_model.predict(s_pred)
            y_predLSTM=s.inverse_transform(y_pred)
            y_predLSTM= np.float64(y_predLSTM)
            train=dfc[:cut]
            valid=dfc[cut:]
            valid=valid[:-1]
            valid['y_pred']=y_predLSTM
            fig=plt.figure(figsize=(8,6))
            plt.title('LSTM Model')
            plt.plot(dfc.Close)
            plt.xlabel('date')
            plt.ylabel('Close Price USD')
            plt.plot(train['Close'])
            plt.plot(valid[['Close','y_pred']])
            plt.show()
            deneme=np.array(deneme)
            deneme=np.reshape(deneme,(3,1))
            s=MinMaxScaler(feature_range=(0,1))
            s_pred=s.fit_transform(deneme)
            s_pred=np.reshape(s_pred,(1,3,1))
            y_preds=new_model.predict(s_pred)
            predLSTM=s.inverse_transform(y_preds)
            predLSTM= np.float64(predLSTM)
            st.plotly_chart(fig,use_container_width=True)
            st.metric(
               label="LSTM Tahmin", value=np.round(predLSTM,3), delta=(np.round(np.float64(predLSTM-close),3))
            )
        def KNN_plot_data():
            modelKNN=joblib.load('C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\KNN\\KNN_Apple.joblib')
            dfk=dfapple.copy()
            day=[[len(dfk)+1]]      
            days=[]
            a_close=[]
            df_days=dfk.loc[:,'Date']
            df_close=dfk.loc[:,'Close']
            for a in range(len(df_days)):
                    days.append([df_close.index[a]])
            for a in df_close:
                    a_close.append(float(a))        
            y_pred=modelKNN.predict(days)
            val=modelKNN.predict(day)
            fig=plt.figure(figsize=(8,6))
            plt.title('KNN Model')
            plt.plot(days,a_close,color='red',label='data')
            plt.plot(days,y_pred,color='blue',label='Predict') 
            plt.show() 
            st.plotly_chart(fig,use_container_width=True)
            st.metric(
               label="KNN Tahmin", value=np.round(val,3), delta=(np.round(np.float64(val-a_close[-1]),3))
            )
        def RF_plot_data():
            modelRF=joblib.load('C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\RF\\RF_Apple_model.joblib')
            dfk=dfapple.copy() 
            day=[[len(dfk)+1]]     
            days=[]
            a_close=[]
            df_days=dfk.loc[:,'Date']
            df_close=dfk.loc[:,'Close']
            for a in range(len(df_days)):
                    days.append([df_close.index[a]])
            for a in df_close:
                    a_close.append(float(a))        
            y_pred=modelRF.predict(days)
            val=modelRF.predict(day)
            fig=plt.figure(figsize=(8,6))
            plt.title('RF Model')
            plt.plot(days,a_close,color='red',label='data')
            plt.plot(days,y_pred,color='blue',label='Predict') 
            plt.show() 
            st.plotly_chart(fig,use_container_width=True)
            st.metric(
               label="RF Tahmin", value=np.round(val,3), delta=(np.round(np.float64(val-a_close[-1]),3))
            )
        RF_plot_data()      
        KNN_plot_data()        
        LSTM_plot_data()        
        XGB_plot_data()
        CATB_plot_data()   
        
    if option=='Walmart':
        st.subheader('WALMART Stock Price Prediction')
        with st.expander('Yöntem ve Materyaller'):
            st.write('alkan')
        dfWmt=pd.read_csv('C:\\Users\\alkan\\Desktop\\VSCODE\\Hisseler\\WMT.csv')
        gb = GridOptionsBuilder.from_dataframe(dfWmt)
        gb.configure_pagination()
        gridOptions = gb.build()
        chart = functools.partial(st.plotly_chart, use_container_width=True)
        AgGrid(dfWmt, gridOptions=gridOptions, allow_unsafe_jscode=True,height=250)
        
        import math
        @st.cache(suppress_st_warning=True)
        def CATB_plot_data():
            modelCat=joblib.load('CAT_model_dosyasi.joblib')
            dfk=dfWmt.copy()
            close=dfk.Close[-1:]
            cut=math.ceil(len(dfk)*(.85))
            trainx=dfk.drop(['Close','Adj Close','Date','Volume'],axis=1)
            trainy=dfk.drop(['High','Low','Open','Volume','Adj Close','Date'],axis=1)
            x_test=trainx[cut:]
            pred=x_test[-1:]
            x_test=x_test[:-1]
            col=x_test.columns.to_list()
            col=['High','Low','Open']
            x_test=x_test[col]
            x_test=np.array(x_test)
            x_test=np.reshape(x_test,(656,3))
            scale=MinMaxScaler(feature_range=(0,1))
            scaled_x=scale.fit_transform(x_test)
            y_test=trainy[cut:]
            y_test=y_test[:-1]
            y_test=np.array(y_test)
            scaled_y=scale.fit_transform(y_test)
            y_preds=modelCat.predict(scaled_x)
            y_preds=np.reshape(y_preds,(y_preds.shape[0],1))
            y_preds=scale.inverse_transform(y_preds)
            grafik=dfk[:cut]
            valid=dfk[cut:-1]
            valid['y_pred']=y_preds
            fig=plt.figure(figsize=(8,6))
            plt.title('CATB Model')
            plt.plot(dfk.Close)
            plt.xlabel('date')
            plt.ylabel('Close Price USD')
            plt.plot(grafik['Close'])
            plt.plot(valid[['Close','y_pred']])
            plt.show()
            st.plotly_chart(fig,use_container_width=True)
            pred=np.array(pred)
            pred=np.reshape(pred,(3,1))
            s=MinMaxScaler(feature_range=(0,1))
            s_pred=s.fit_transform(pred)
            s_pred=s_pred.reshape((1,3))
            y_preds=modelCat.predict(s_pred) 
            y_preds=np.reshape(y_preds,(y_preds.shape[0],1))
            y_preds=s.inverse_transform(y_preds)
            y_predCat= np.float64(y_preds)
            st.metric(
               label="CATB Tahmin", value=np.round(y_predCat,3), delta=(np.round(np.float64(y_predCat-close),3))
            )
        
        def XGB_plot_data():
            filename='C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\XGB\\XGB2_WMT_model'
            modelXGB=pickle.load(open(filename,'rb'))
            dfk=dfWmt.copy()
            close=dfk.Close[-1:]
            x_train=[]
            y_train=[]
            for i in range(30,len(dfk)):
                    x_train.append(dfk[i-30:i]['Close'])
                    y_train.append(dfk.Close[i])
            cut=math.ceil(len(dfk)*(.80))
            x_test=x_train[cut:]
            y_test=y_train[cut:]
            x_test,y_test=np.array(x_test),np.array(y_test)
            y_test=np.reshape(y_test,(y_test.shape[0],1))
            y_pred=modelXGB.predict(x_test)
            y_pred=np.reshape(y_pred,(y_pred.shape[0],1))
            grafik=dfk[:cut]
            index=dfk[cut:-30]
            index=index.set_index(pd.Index(np.arange(3530,(3530+846),1)))
            index['y_pred']=y_pred
            fig=plt.figure(figsize=(8,6))
            plt.title('XGB Model')
            plt.plot(dfk.Close)
            plt.xlabel('date')
            plt.ylabel('Close Price USD')
            plt.plot(index['y_pred'])
            plt.show()
            st.plotly_chart(fig,use_container_width=True)
            x=np.array(dfk.Close[-30:])
            x=x.reshape((1,30))
            y_predXgb=modelXGB.predict(x)
            y_predXgb=np.float64(y_predXgb)
            st.metric(
               label="XGB Tahmin", value=np.round(y_predXgb,3), delta=(np.round(np.float64(y_predXgb-close),3))
            )

        def LSTM_plot_data():
            new_model=keras.models.load_model('C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\LSTM\\LSTM_Walmart.h5')
            dfc=dfWmt.copy()
            close=dfc.Close[-1:]
            dfc=dfc[:-1]
            cut=math.ceil(len(dfc)*(.85))
            test=dfc.iloc[cut:,:]
            x_test=test.drop(['Close','Adj Close','Date','Volume'],axis=1)
            y_test=test.drop(['High','Low','Open','Adj Close','Date','Volume'],axis=1)
            deneme=x_test[-1:]
            x_test=x_test[:-1]
            y_test=y_test[1:]
            x_test=x_test.values
            y_test=y_test.values
            s=MinMaxScaler(feature_range=(0,1))
            s_pred=s.fit_transform(x_test)
            Y=s.fit_transform(y_test)
            s_pred=s_pred.reshape((x_test.shape[0],3,1))
            y_pred=new_model.predict(s_pred)
            y_predLSTM=s.inverse_transform(y_pred)
            y_predLSTM= np.float64(y_predLSTM)
            train=dfc[:cut]
            valid=dfc[cut:]
            valid=valid[:-1]
            valid['y_pred']=y_predLSTM
            fig=plt.figure(figsize=(8,6))
            plt.title('LSTM Model')
            plt.plot(dfc.Close)
            plt.xlabel('date')
            plt.ylabel('Close Price USD')
            plt.plot(train['Close'])
            plt.plot(valid[['Close','y_pred']])
            plt.show()
            deneme=np.array(deneme)
            deneme=np.reshape(deneme,(3,1))
            s=MinMaxScaler(feature_range=(0,1))
            s_pred=s.fit_transform(deneme)
            s_pred=np.reshape(s_pred,(1,3,1))
            y_preds=new_model.predict(s_pred)
            predLSTM=s.inverse_transform(y_preds)
            predLSTM= np.float64(predLSTM)
            st.plotly_chart(fig,use_container_width=True)
            st.metric(
               label="LSTM Tahmin", value=np.round(predLSTM,3), delta=(np.round(np.float64(predLSTM-close),3))
            )
        def KNN_plot_data():
            modelKNN=joblib.load('C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\KNN\\KNN_WMT.joblib')
            dfk=dfWmt.copy()
            day=[[len(dfk)+1]]      
            days=[]
            a_close=[]
            df_days=dfk.loc[:,'Date']
            df_close=dfk.loc[:,'Close']
            for a in range(len(df_days)):
                    days.append([df_close.index[a]])
            for a in df_close:
                    a_close.append(float(a))        
            y_pred=modelKNN.predict(days)
            val=modelKNN.predict(day)
            fig=plt.figure(figsize=(8,6))
            plt.title('KNN Model')
            plt.plot(days,a_close,color='red',label='data')
            plt.plot(days,y_pred,color='blue',label='Predict') 
            plt.show() 
            st.plotly_chart(fig,use_container_width=True)
            st.metric(
               label="KNN Tahmin", value=np.round(val,3), delta=(np.round(np.float64(val-a_close[-1]),3))
            )
        def RF_plot_data():
            modelRF=joblib.load('C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\RF\\RF_WMT_model.joblib')
            dfk=dfWmt.copy() 
            day=[[len(dfk)+1]]     
            days=[]
            a_close=[]
            df_days=dfk.loc[:,'Date']
            df_close=dfk.loc[:,'Close']
            for a in range(len(df_days)):
                    days.append([df_close.index[a]])
            for a in df_close:
                    a_close.append(float(a))        
            y_pred=modelRF.predict(days)
            val=modelRF.predict(day)
            fig=plt.figure(figsize=(8,6))
            plt.title('RF Model')
            plt.plot(days,a_close,color='red',label='data')
            plt.plot(days,y_pred,color='blue',label='Predict') 
            plt.show() 
            st.plotly_chart(fig,use_container_width=True)
            st.metric(
               label="RF Tahmin", value=np.round(val,3), delta=(np.round(np.float64(val-a_close[-1]),3))
            )
        RF_plot_data()      
        KNN_plot_data()        
        LSTM_plot_data()        
        XGB_plot_data()
        CATB_plot_data()

    if option=='Ford':
        st.subheader('Ford Stock Price Prediction')
    
        dfF=pd.read_csv('C:\\Users\\alkan\\Desktop\\VSCODE\\Hisseler\\F.csv')
        gb = GridOptionsBuilder.from_dataframe(dfF)
        gb.configure_pagination()
        gridOptions = gb.build()
        chart = functools.partial(st.plotly_chart, use_container_width=True)
        AgGrid(dfF, gridOptions=gridOptions, allow_unsafe_jscode=True,height=250)
        
        import math
        @st.cache(suppress_st_warning=True)
        def CATB_plot_data():
            modelCat=joblib.load('CAT_model_dosyasi.joblib')
            dfk=dfF.copy()
            close=dfk.Close[-1:]
            cut=math.ceil(len(dfk)*(.85))
            trainx=dfk.drop(['Close','Adj Close','Date','Volume'],axis=1)
            trainy=dfk.drop(['High','Low','Open','Volume','Adj Close','Date'],axis=1)
            x_test=trainx[cut:]
            pred=x_test[-1:]
            x_test=x_test[:-1]
            col=x_test.columns.to_list()
            col=['High','Low','Open']
            x_test=x_test[col]
            x_test=np.array(x_test)
            x_test=np.reshape(x_test,(656,3))
            scale=MinMaxScaler(feature_range=(0,1))
            scaled_x=scale.fit_transform(x_test)
            y_test=trainy[cut:]
            y_test=y_test[:-1]
            y_test=np.array(y_test)
            scaled_y=scale.fit_transform(y_test)
            y_preds=modelCat.predict(scaled_x)
            y_preds=np.reshape(y_preds,(y_preds.shape[0],1))
            y_preds=scale.inverse_transform(y_preds)
            grafik=dfk[:cut]
            valid=dfk[cut:-1]
            valid['y_pred']=y_preds
            fig=plt.figure(figsize=(8,6))
            plt.title('CATB Model')
            plt.plot(dfk.Close)
            plt.xlabel('date')
            plt.ylabel('Close Price USD')
            plt.plot(grafik['Close'])
            plt.plot(valid[['Close','y_pred']])
            plt.show()
            st.plotly_chart(fig,use_container_width=True)
            pred=np.array(pred)
            pred=np.reshape(pred,(3,1))
            s=MinMaxScaler(feature_range=(0,1))
            s_pred=s.fit_transform(pred)
            s_pred=s_pred.reshape((1,3))
            y_preds=modelCat.predict(s_pred) 
            y_preds=np.reshape(y_preds,(y_preds.shape[0],1))
            y_preds=s.inverse_transform(y_preds)
            y_predCat= np.float64(y_preds)
            st.metric(
               label="CATB Tahmin", value=np.round(y_predCat,3), delta=(np.round(np.float64(y_predCat-close),3))
            )
        
        def XGB_plot_data():
            filename='C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\XGB\\XGB2_F_model'
            modelXGB=pickle.load(open(filename,'rb'))
            dfk=dfF.copy()
            close=dfk.Close[-1:]
            x_train=[]
            y_train=[]
            for i in range(30,len(dfk)):
                    x_train.append(dfk[i-30:i]['Close'])
                    y_train.append(dfk.Close[i])
            cut=math.ceil(len(dfk)*(.80))
            x_test=x_train[cut:]
            y_test=y_train[cut:]
            x_test,y_test=np.array(x_test),np.array(y_test)
            y_test=np.reshape(y_test,(y_test.shape[0],1))
            y_pred=modelXGB.predict(x_test)
            y_pred=np.reshape(y_pred,(y_pred.shape[0],1))
            grafik=dfk[:cut]
            index=dfk[cut:-30]
            index=index.set_index(pd.Index(np.arange(3530,(3530+846),1)))
            index['y_pred']=y_pred
            fig=plt.figure(figsize=(8,6))
            plt.title('XGB Model')
            plt.plot(dfk.Close)
            plt.xlabel('date')
            plt.ylabel('Close Price USD')
            plt.plot(index['y_pred'])
            plt.show()
            st.plotly_chart(fig,use_container_width=True)
            x=np.array(dfk.Close[-30:])
            x=x.reshape((1,30))
            y_predXgb=modelXGB.predict(x)
            y_predXgb=np.float64(y_predXgb)
            st.metric(
               label="XGB Tahmin", value=np.round(y_predXgb,3), delta=(np.round(np.float64(y_predXgb-close),3))
            )

        def LSTM_plot_data():
            new_model=keras.models.load_model('C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\LSTM\\LSTM_F.h5')
            dfc=dfF.copy()
            close=dfc.Close[-1:]
            dfc=dfc[:-1]
            cut=math.ceil(len(dfc)*(.85))
            test=dfc.iloc[cut:,:]
            x_test=test.drop(['Close','Adj Close','Date','Volume'],axis=1)
            y_test=test.drop(['High','Low','Open','Adj Close','Date','Volume'],axis=1)
            deneme=x_test[-1:]
            x_test=x_test[:-1]
            y_test=y_test[1:]
            x_test=x_test.values
            y_test=y_test.values
            s=MinMaxScaler(feature_range=(0,1))
            s_pred=s.fit_transform(x_test)
            Y=s.fit_transform(y_test)
            s_pred=s_pred.reshape((x_test.shape[0],3,1))
            y_pred=new_model.predict(s_pred)
            y_predLSTM=s.inverse_transform(y_pred)
            y_predLSTM= np.float64(y_predLSTM)
            train=dfc[:cut]
            valid=dfc[cut:]
            valid=valid[:-1]
            valid['y_pred']=y_predLSTM
            fig=plt.figure(figsize=(8,6))
            plt.title('LSTM Model')
            plt.plot(dfc.Close)
            plt.xlabel('date')
            plt.ylabel('Close Price USD')
            plt.plot(train['Close'])
            plt.plot(valid[['Close','y_pred']])
            plt.show()
            deneme=np.array(deneme)
            deneme=np.reshape(deneme,(3,1))
            s=MinMaxScaler(feature_range=(0,1))
            s_pred=s.fit_transform(deneme)
            s_pred=np.reshape(s_pred,(1,3,1))
            y_preds=new_model.predict(s_pred)
            predLSTM=s.inverse_transform(y_preds)
            predLSTM= np.float64(predLSTM)
            st.plotly_chart(fig,use_container_width=True)
            st.metric(
               label="LSTM Tahmin", value=np.round(predLSTM,3), delta=(np.round(np.float64(predLSTM-close),3))
            )
        def KNN_plot_data():
            modelKNN=joblib.load('C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\KNN\\KNN_F.joblib')
            dfk=dfF.copy()
            day=[[len(dfk)+1]]      
            days=[]
            a_close=[]
            df_days=dfk.loc[:,'Date']
            df_close=dfk.loc[:,'Close']
            for a in range(len(df_days)):
                    days.append([df_close.index[a]])
            for a in df_close:
                    a_close.append(float(a))        
            y_pred=modelKNN.predict(days)
            val=modelKNN.predict(day)
            fig=plt.figure(figsize=(8,6))
            plt.title('KNN Model')
            plt.plot(days,a_close,color='red',label='data')
            plt.plot(days,y_pred,color='blue',label='Predict') 
            plt.show() 
            st.plotly_chart(fig,use_container_width=True)
            st.metric(
               label="KNN Tahmin", value=np.round(val,3), delta=(np.round(np.float64(val-a_close[-1]),3))
            )
        def RF_plot_data():
            modelRF=joblib.load('C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\RF\\RF_F_model.joblib')
            dfk=dfF.copy() 
            day=[[len(dfk)+1]]     
            days=[]
            a_close=[]
            df_days=dfk.loc[:,'Date']
            df_close=dfk.loc[:,'Close']
            for a in range(len(df_days)):
                    days.append([df_close.index[a]])
            for a in df_close:
                    a_close.append(float(a))        
            y_pred=modelRF.predict(days)
            val=modelRF.predict(day)
            fig=plt.figure(figsize=(8,6))
            plt.title('RF Model')
            plt.plot(days,a_close,color='red',label='data')
            plt.plot(days,y_pred,color='blue',label='Predict') 
            plt.show() 
            st.plotly_chart(fig,use_container_width=True)
            st.metric(
               label="RF Tahmin", value=np.round(val,3), delta=(np.round(np.float64(val-a_close[-1]),3))
            )
        RF_plot_data()      
        KNN_plot_data()        
        LSTM_plot_data()        
        XGB_plot_data()
        CATB_plot_data()
        
    if option=='THY':
        st.subheader('THY Stock Price Prediction')
        
        dfThy=pd.read_csv('C:\\Users\\alkan\\Desktop\\VSCODE\\Hisseler\\THYAO.IS.csv')
        gb = GridOptionsBuilder.from_dataframe(dfThy)
        gb.configure_pagination()
        gridOptions = gb.build()
        chart = functools.partial(st.plotly_chart, use_container_width=True)
        AgGrid(dfThy, gridOptions=gridOptions, allow_unsafe_jscode=True,height=250)
        
        import math
        @st.cache(suppress_st_warning=True)
        def CATB_plot_data():
            modelCat=joblib.load('CAT_THY.joblib')
            dfk=dfThy.copy()
            close=dfk.Close[-1:]
            cut=math.ceil(len(dfk)*(.80))
            trainx=dfk.drop(['Close','Adj Close','Date','Volume'],axis=1)
            trainy=dfk.drop(['High','Low','Open','Volume','Adj Close','Date'],axis=1)
            x_test=trainx[cut:]
            pred=x_test[-1:]
            x_test=x_test[:-1]
            col=x_test.columns.to_list()
            col=['High','Low','Open']
            x_test=x_test[col]
            x_test=np.array(x_test)
            x_test=np.reshape(x_test,(671,3))
            scale=MinMaxScaler(feature_range=(0,1))
            scaled_x=scale.fit_transform(x_test)
            y_test=trainy[cut:]
            y_test=y_test[:-1]
            y_test=np.array(y_test)
            scaled_y=scale.fit_transform(y_test)
            y_preds=modelCat.predict(scaled_x)
            y_preds=np.reshape(y_preds,(y_preds.shape[0],1))
            y_preds=scale.inverse_transform(y_preds)
            grafik=dfk[:cut]
            valid=dfk[cut:-1]
            valid['y_pred']=y_preds
            fig=plt.figure(figsize=(8,6))
            plt.title('CATB Model')
            plt.plot(dfk.Close)
            plt.xlabel('date')
            plt.ylabel('Close Price USD')
            plt.plot(grafik['Close'])
            plt.plot(valid[['Close','y_pred']])
            plt.show()
            st.plotly_chart(fig,use_container_width=True)
            pred=np.array(pred)
            pred=np.reshape(pred,(3,1))
            s=MinMaxScaler(feature_range=(0,1))
            s_pred=s.fit_transform(pred)
            s_pred=s_pred.reshape((1,3))
            y_preds=modelCat.predict(s_pred) 
            y_preds=np.reshape(y_preds,(y_preds.shape[0],1))
            y_preds=s.inverse_transform(y_preds)
            y_predCat= np.float64(y_preds)
            st.metric(
               label="CATB Tahmin", value=np.round(y_predCat,3), delta=(np.round(np.float64(y_predCat-close),3))
            )
        
        def XGB_plot_data():
            filename='C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\XGB\\XGB2_THY_model'
            modelXGB=pickle.load(open(filename,'rb'))
            dfk=dfThy.copy()
            close=dfk.Close[-1:]
            x_train=[]
            y_train=[]
            for i in range(30,len(dfk)):
                    x_train.append(dfk[i-30:i]['Close'])
                    y_train.append(dfk.Close[i])
            cut=math.ceil(len(dfk)*(.80))
            x_test=x_train[cut:]
            y_test=y_train[cut:]
            x_test,y_test=np.array(x_test),np.array(y_test)
            y_test=np.reshape(y_test,(y_test.shape[0],1))
            y_pred=modelXGB.predict(x_test)
            y_pred=np.reshape(y_pred,(y_pred.shape[0],1))
            grafik=dfk[:cut]
            index=dfk[cut:-30]
            index=index.set_index(pd.Index(np.arange(3630,(3630+866),1)))
            index['y_pred']=y_pred
            fig=plt.figure(figsize=(8,6))
            plt.title('XGB Model')
            plt.plot(dfk.Close)
            plt.xlabel('date')
            plt.ylabel('Close Price USD')
            plt.plot(index['y_pred'])
            plt.show()
            st.plotly_chart(fig,use_container_width=True)
            x=np.array(dfk.Close[-30:])
            x=x.reshape((1,30))
            y_predXgb=modelXGB.predict(x)
            y_predXgb=np.float64(y_predXgb)
            st.metric(
               label="XGB Tahmin", value=np.round(y_predXgb,3), delta=(np.round(np.float64(y_predXgb-close),3))
            )

        def LSTM_plot_data():
            new_model=keras.models.load_model('C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\LSTM\\LSTM_THY.h5')
            dfc=dfThy.copy()
            close=dfc.Close[-1:]
            dfc=dfc[:-1]
            cut=math.ceil(len(dfc)*(.85))
            test=dfc.iloc[cut:,:]
            x_test=test.drop(['Close','Adj Close','Date','Volume'],axis=1)
            y_test=test.drop(['High','Low','Open','Adj Close','Date','Volume'],axis=1)
            deneme=x_test[-1:]
            x_test=x_test[:-1]
            y_test=y_test[1:]
            x_test=x_test.values
            y_test=y_test.values
            s=MinMaxScaler(feature_range=(0,1))
            s_pred=s.fit_transform(x_test)
            Y=s.fit_transform(y_test)
            s_pred=s_pred.reshape((x_test.shape[0],3,1))
            y_pred=new_model.predict(s_pred)
            y_predLSTM=s.inverse_transform(y_pred)
            y_predLSTM= np.float64(y_predLSTM)
            train=dfc[:cut]
            valid=dfc[cut:]
            valid=valid[:-1]
            valid['y_pred']=y_predLSTM
            fig=plt.figure(figsize=(8,6))
            plt.title('LSTM Model')
            plt.plot(dfc.Close)
            plt.xlabel('date')
            plt.ylabel('Close Price USD')
            plt.plot(train['Close'])
            plt.plot(valid[['Close','y_pred']])
            plt.show()
            deneme=np.array(deneme)
            deneme=np.reshape(deneme,(3,1))
            s=MinMaxScaler(feature_range=(0,1))
            s_pred=s.fit_transform(deneme)
            s_pred=np.reshape(s_pred,(1,3,1))
            y_preds=new_model.predict(s_pred)
            predLSTM=s.inverse_transform(y_preds)
            predLSTM= np.float64(predLSTM)
            st.plotly_chart(fig,use_container_width=True)
            st.metric(
               label="LSTM Tahmin", value=np.round(predLSTM,3), delta=(np.round(np.float64(predLSTM-close),3))
            )
        def KNN_plot_data():
            modelKNN=joblib.load('C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\KNN\\KNN_THY.joblib')
            dfk=dfThy.copy()
            day=[[len(dfk)+1]]      
            days=[]
            a_close=[]
            df_days=dfk.loc[:,'Date']
            df_close=dfk.loc[:,'Close']
            for a in range(len(df_days)):
                    days.append([df_close.index[a]])
            for a in df_close:
                    a_close.append(float(a))        
            y_pred=modelKNN.predict(days)
            val=modelKNN.predict(day)
            fig=plt.figure(figsize=(8,6))
            plt.title('KNN Model')
            plt.plot(days,a_close,color='red',label='data')
            plt.plot(days,y_pred,color='blue',label='Predict') 
            plt.show() 
            st.plotly_chart(fig,use_container_width=True)
            st.metric(
               label="KNN Tahmin", value=np.round(val,3), delta=(np.round(np.float64(val-a_close[-1]),3))
            )
        def RF_plot_data():
            modelRF=joblib.load('C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\RF\\RF_THY_model.joblib')
            dfk=dfThy.copy() 
            day=[[len(dfk)+1]]     
            days=[]
            a_close=[]
            df_days=dfk.loc[:,'Date']
            df_close=dfk.loc[:,'Close']
            for a in range(len(df_days)):
                    days.append([df_close.index[a]])
            for a in df_close:
                    a_close.append(float(a))        
            y_pred=modelRF.predict(days)
            val=modelRF.predict(day)
            fig=plt.figure(figsize=(8,6))
            plt.title('RF Model')
            plt.plot(days,a_close,color='red',label='data')
            plt.plot(days,y_pred,color='blue',label='Predict') 
            plt.show() 
            st.plotly_chart(fig,use_container_width=True)
            st.metric(
               label="RF Tahmin", value=np.round(val,3), delta=(np.round(np.float64(val-a_close[-1]),3))
            )
        RF_plot_data()      
        KNN_plot_data()        
        LSTM_plot_data()        
        XGB_plot_data()
        CATB_plot_data() 
        
    if option=='Aselsan':
        st.subheader('Aselsan Stock Price Prediction')
        with st.expander('Yöntem ve Materyaller'):
            st.write('alkan')
        dfAselsan=pd.read_csv('C:\\Users\\alkan\\Desktop\\VSCODE\\Hisseler\\ASELS.IS.csv')
        gb = GridOptionsBuilder.from_dataframe(dfAselsan)
        gb.configure_pagination()
        gridOptions = gb.build()
        chart = functools.partial(st.plotly_chart, use_container_width=True)
        AgGrid(dfAselsan, gridOptions=gridOptions, allow_unsafe_jscode=True,height=250)
        
        import math
        @st.cache(suppress_st_warning=True)
        def CATB_plot_data():
            modelCat=joblib.load('CAT_model_dosyasi.joblib')
            dfk=dfAselsan.copy()
            close=dfk.Close[-1:]
            cut=math.ceil(len(dfk)*(.85))
            trainx=dfk.drop(['Close','Adj Close','Date','Volume'],axis=1)
            trainy=dfk.drop(['High','Low','Open','Volume','Adj Close','Date'],axis=1)
            x_test=trainx[cut:]
            pred=x_test[-1:]
            x_test=x_test[:-1]
            col=x_test.columns.to_list()
            col=['High','Low','Open']
            x_test=x_test[col]
            x_test=np.array(x_test)
            x_test=np.reshape(x_test,(671,3))
            scale=MinMaxScaler(feature_range=(0,1))
            scaled_x=scale.fit_transform(x_test)
            y_test=trainy[cut:]
            y_test=y_test[:-1]
            y_test=np.array(y_test)
            scaled_y=scale.fit_transform(y_test)
            y_preds=modelCat.predict(scaled_x)
            y_preds=np.reshape(y_preds,(y_preds.shape[0],1))
            y_preds=scale.inverse_transform(y_preds)
            grafik=dfk[:cut]
            valid=dfk[cut:-1]
            valid['y_pred']=y_preds
            fig=plt.figure(figsize=(8,6))
            plt.title('CATB Model')
            plt.plot(dfk.Close)
            plt.xlabel('date')
            plt.ylabel('Close Price USD')
            plt.plot(grafik['Close'])
            plt.plot(valid[['Close','y_pred']])
            plt.show()
            st.plotly_chart(fig,use_container_width=True)
            pred=np.array(pred)
            pred=np.reshape(pred,(3,1))
            s=MinMaxScaler(feature_range=(0,1))
            s_pred=s.fit_transform(pred)
            s_pred=s_pred.reshape((1,3))
            y_preds=modelCat.predict(s_pred) 
            y_preds=np.reshape(y_preds,(y_preds.shape[0],1))
            y_preds=s.inverse_transform(y_preds)
            y_predCat= np.float64(y_preds)
            st.metric(
               label="CATB Tahmin", value=np.round(y_predCat,3), delta=(np.round(np.float64(y_predCat-close),3))
            )
        
        def XGB_plot_data():
            filename='C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\XGB\\XGB2_Aselsan_model'
            modelXGB=pickle.load(open(filename,'rb'))
            dfk=dfAselsan.copy()
            close=dfk.Close[-1:]
            x_train=[]
            y_train=[]
            for i in range(30,len(dfk)):
                    x_train.append(dfk[i-30:i]['Close'])
                    y_train.append(dfk.Close[i])
            cut=math.ceil(len(dfk)*(.80))
            x_test=x_train[cut:]
            y_test=y_train[cut:]
            x_test,y_test=np.array(x_test),np.array(y_test)
            y_test=np.reshape(y_test,(y_test.shape[0],1))
            y_pred=modelXGB.predict(x_test)
            y_pred=np.reshape(y_pred,(y_pred.shape[0],1))
            grafik=dfk[:cut]
            index=dfk[cut:-30]
            index=index.set_index(pd.Index(np.arange(3630,(3630+866),1)))
            index['y_pred']=y_pred
            fig=plt.figure(figsize=(8,6))
            plt.title('XGB Model')
            plt.plot(dfk.Close)
            plt.xlabel('date')
            plt.ylabel('Close Price USD')
            plt.plot(index['y_pred'])
            plt.show()
            st.plotly_chart(fig,use_container_width=True)
            x=np.array(dfk.Close[-30:])
            x=x.reshape((1,30))
            y_predXgb=modelXGB.predict(x)
            y_predXgb=np.float64(y_predXgb)
            st.metric(
               label="XGB Tahmin", value=np.round(y_predXgb,3), delta=(np.round(np.float64(y_predXgb-close),3))
            )

        def LSTM_plot_data():
            new_model=keras.models.load_model('C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\LSTM\\LSTM_Aselsan.h5')
            dfc=dfAselsan.copy()
            close=dfc.Close[-1:]
            dfc=dfc[:-1]
            cut=math.ceil(len(dfc)*(.85))
            test=dfc.iloc[cut:,:]
            x_test=test.drop(['Close','Adj Close','Date','Volume'],axis=1)
            y_test=test.drop(['High','Low','Open','Adj Close','Date','Volume'],axis=1)
            deneme=x_test[-1:]
            x_test=x_test[:-1]
            y_test=y_test[1:]
            x_test=x_test.values
            y_test=y_test.values
            s=MinMaxScaler(feature_range=(0,1))
            s_pred=s.fit_transform(x_test)
            Y=s.fit_transform(y_test)
            s_pred=s_pred.reshape((x_test.shape[0],3,1))
            y_pred=new_model.predict(s_pred)
            y_predLSTM=s.inverse_transform(y_pred)
            y_predLSTM= np.float64(y_predLSTM)
            train=dfc[:cut]
            valid=dfc[cut:]
            valid=valid[:-1]
            valid['y_pred']=y_predLSTM
            fig=plt.figure(figsize=(8,6))
            plt.title('LSTM Model')
            plt.plot(dfc.Close)
            plt.xlabel('date')
            plt.ylabel('Close Price USD')
            plt.plot(train['Close'])
            plt.plot(valid[['Close','y_pred']])
            plt.show()
            deneme=np.array(deneme)
            deneme=np.reshape(deneme,(3,1))
            s=MinMaxScaler(feature_range=(0,1))
            s_pred=s.fit_transform(deneme)
            s_pred=np.reshape(s_pred,(1,3,1))
            y_preds=new_model.predict(s_pred)
            predLSTM=s.inverse_transform(y_preds)
            predLSTM= np.float64(predLSTM)
            st.plotly_chart(fig,use_container_width=True)
            st.metric(
               label="LSTM Tahmin", value=np.round(predLSTM,3), delta=(np.round(np.float64(predLSTM-close),3))
            )
        def KNN_plot_data():
            modelKNN=joblib.load('C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\KNN\\KNN_Aselsan.joblib')
            dfk=dfAselsan.copy()
            day=[[len(dfk)+1]]      
            days=[]
            a_close=[]
            df_days=dfk.loc[:,'Date']
            df_close=dfk.loc[:,'Close']
            for a in range(len(df_days)):
                    days.append([df_close.index[a]])
            for a in df_close:
                    a_close.append(float(a))        
            y_pred=modelKNN.predict(days)
            val=modelKNN.predict(day)
            fig=plt.figure(figsize=(8,6))
            plt.title('KNN Model')
            plt.plot(days,a_close,color='red',label='data')
            plt.plot(days,y_pred,color='blue',label='Predict') 
            plt.show() 
            st.plotly_chart(fig,use_container_width=True)
            st.metric(
               label="KNN Tahmin", value=np.round(val,3), delta=(np.round(np.float64(val-a_close[-1]),3))
            )
        def RF_plot_data():
            modelRF=joblib.load('C:\\Users\\alkan\\Desktop\\VSCODE\\modeller\\RF\\RF_Aselsan_model.joblib')
            dfk=dfAselsan.copy() 
            day=[[len(dfk)+1]]     
            days=[]
            a_close=[]
            df_days=dfk.loc[:,'Date']
            df_close=dfk.loc[:,'Close']
            for a in range(len(df_days)):
                    days.append([df_close.index[a]])
            for a in df_close:
                    a_close.append(float(a))        
            y_pred=modelRF.predict(days)
            val=modelRF.predict(day)
            fig=plt.figure(figsize=(8,6))
            plt.title('RF Model')
            plt.plot(days,a_close,color='red',label='data')
            plt.plot(days,y_pred,color='blue',label='Predict') 
            plt.show() 
            st.plotly_chart(fig,use_container_width=True)
            st.metric(
               label="RF Tahmin", value=np.round(val,3), delta=(np.round(np.float64(val-a_close[-1]),3))
            )
        RF_plot_data()      
        KNN_plot_data()        
        LSTM_plot_data()        
        XGB_plot_data()
        CATB_plot_data()   
    
                 