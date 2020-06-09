# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash.dash import no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import base64
from PIL import Image
from io import BytesIO
import numpy as np


app = dash.Dash(__name__)
server = app.server

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    return image


app.layout = html.Div(className='row',
                    children=[
    # Banner
    html.Div(children=[html.Div(html.H1("Clasificador con Deep Learning")), html.H4("Detector de neumonía")],
            className='row header',
            style={'textAlign':'center', 'padding-top':'30px', 'padding-bottom':'10px'}),

    # Body
    html.Div(children=[dbc.Row(html.Div(children=[
                # First column, (Upload image and show image)
                dbc.Col(
                    html.Div(
                        className='seven columns mini_container',
                        children=[dbc.Row(dcc.Upload(
                                                id='input-img',
                                                className='button',
                                                children=html.Div(['Arrastra y suelta o ', html.A('Subir imagen')])
                                                )),

                                #IMAGEN SUBIDA
                                dbc.Row(html.Img(id='output-img',
                                                className='img',
                                                src='assets/no-image2.png',
                                                style={'width':'100%',
                                                    'margin-top':'100px',
                                                    'box-shadow': '9px 10px 23px -7px rgba(0,0,0,0.67)',
                                                    'max-width':'50%'}))],
                                        style={'textAlign': 'center', 
                                                'height':'798px',
                                                'padding':'30px',
                                                'max-width': '100%',
                                                'max-height': '100%'})),
        
                # Second column (Model select, model info)
                dbc.Col(
                    html.Div(className='five columns mini_container',
                        children=[
                            dcc.Dropdown(id = 'select-model',
                                        options=[
                                            {'label':'VGG16', 'value':'vgg16'},
                                            {'label':'VGG19', 'value':'vgg19'},
                                            {'label':'Xception', 'value':'xception'},
                                            {'label':'InceptionV3', 'value':'inceptionv3'}],
                                        placeholder='Selecciona un modelo...'),
                            html.Div(id='predict-result', 
                                    children=[dcc.Loading(id='prediction-loading', children=[
                                        html.Div([html.P("Neumonía:"), html.P(id='prediction_c1')]),
                                        html.Div([html.P("Normal:"), html.P(id='prediction_c2')])
                                        ])],
                                    style = {'padding-top':'10px', 'padding-left':'5px'}),
                                    html.Div(html.Button(id='predict-button',
                                    children='Clasificar!'),
                                    style={'textAlign':'center'})
                                ]
                                
                        )),

            # Confusion matrix img
            dbc.Col(html.Div(className='five columns mini_container',
                        children=[html.H4("Matriz de confusión del modelo"),
                                  html.Img(
                                        id="confmatrix-image",
                                        className="img",
                                        style={'width':'80%', 'margin-top':'10px','max-width': '80%','max-height': '90%'})],
                    style={'textAlign': 'center',
                            'height':'45vh',
                            'padding':'30px',
                            })
                        )
    ]), 
    
        style={'height':'900px', 'padding-left': '50px'}),
    
     ])
])

# Mostrar imagen subida
@app.callback(Output('output-img', 'src'),
             [Input('input-img', 'contents')])
def upload_img(contents):
    if contents != None:
        print("Log: Imagen subida correctamente")
        return contents
    else:
        PreventUpdate 

## Acciones al elegir modelo
@app.callback(Output('confmatrix-image', 'src'),
              [Input('select-model','value')])

def upadte_cm_image(selected_model):
    if selected_model != None:
        src = 'assets/models/{}/{}_MatrizConfusion.png'.format(selected_model, selected_model)
        print(src)
        return src
    else:
        PreventUpdate


#Predicción al pulsar el botón
@app.callback([Output('prediction_c1', 'children'),
            Output('prediction_c2', 'children')],
            [Input('select-model','value'),
            Input('predict-button', 'n_clicks'),
            Input('input-img', 'contents')])
def prediction(predict_n_clicks, selected_model, img_coded):
    model= None
    if predict_n_clicks is None:
        raise PreventUpdate
    else:
        print("------------- Entramos en predicción----------")
        Carga el modelo 
        if selected_model=='vgg16':
            model = load_model('assets/models/vgg16/vgg16.h5')
            print("Cargado correctamente")

        Decodifica la imagen
        base64_input = img_coded.split(',')[1]
        im = Image.open(BytesIO(base64.b64decode(base64_input)))
        ready_img = prepare_image(im, (224,224))

        Realiza la predicción
        prediction = round(model.predict(ready_img, steps=1)[0], 2)
        class_0_percent = str(prediction*100) + ' %'
        class_1_percent = str(100-prediction*100) + ' %'
        print("Predicción realizada correctamente")
        print(prediction)

    return class_0_percent, class_1_percent


if __name__ == '__main__':
    app.run_server(debug=True)