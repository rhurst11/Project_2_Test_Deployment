import numpy as np
from flask import Flask, render_template, url_for, request

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/model")
def model():
    return render_template("model.html")


#To use the predict button in our web-app
@app.route("/predict", methods=["POST"])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)   
    
    return render_template("predict.html", prediction_text = 'Classifier result for end user is : {}'.format(output))



if __name__ == "__main__":
    
    app.run(debug = True)






# import numpy as np
# from flask import Flask, render_template, url_for, request, jsonify
# import pandas as pd

# import tensorflow as tf
# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Sequential
# from sklearn.preprocessing import StandardScaler

# import pickle
# from pickle import load


# # loading in model
# file_path = ('modelling/r_oversampled_8_features.json')

# with open(file_path, "r") as json_file:
#     model_json = json_file.read()
# loaded_model = model_from_json(model_json)


# # loading in model weights to model

# file_path_weights = ('modelling/r_oversampled_8_features.h5')
# loaded_model.load_weights(file_path_weights)

# # loading in dataframe for scaling
# web_df = pd.read_csv("modelling/web_deployment.csv")
# data_df.dropna(inplace=True)

# # loading in scaler
# scaler_final = pickle.load(open('scaler_4.pkl', 'rb'))





# app = Flask(__name__)
# # run_with_ngrok(app)



# # REMOVE AFTER TESTING
# #default page of our web-app

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route("/about")
# def about():
#     return render_template("about.html")

# @app.route("/model")
# def model():
#     return render_template("model.html")


# #To use the predict button in our web-app
# @app.route('/predict',methods=['POST'])
# def predict():
#     '''
#     For rendering results on HTML GUI
#     '''
   
#     '''
#     outlining process for scaling / integrating user input to scale
#     '''

#     '''
#     User Input itegration through pandas DataFrame manipulation
#     '''

#     # # loading in dataframe for scaling
#     # web_df = pd.read_csv("modelling/web_deployment.csv",index_col=[0])
#     # data_df.dropna(inplace=True)

#     # # loading in scaler
#     # scaler_final = pickle.load(open('scaler_3.pkl', 'rb'))
    
#     # setting up user input integration
#     user_inputs = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_EMPLOYED', 'MONTHS_BALANCE', 'AMT_PAYMENT_CURRENT', 'CNT_INSTALMENT_MATURE_CUM']
#     int_features = [float(x) for x in request.form.values()]
#     # final_features = np.array(int_features).reshape(-1, 8)

#     # creating user input df
#     user_input_df = pd.DataFrame(int_features)
#     user_input_df = user_input_df.T
#     user_input_df.columns = (user_inputs)

#     # appending user data to original dataframe for scale transformation
#     web_df.iloc[-1] = user_input_df.iloc[0]

#     '''
#     Standard Scaller Implementation
#     '''

#     # scaler_final = pickle.load(open('scaler_3.pkl', 'rb'))

#     user_input_scaled = scaler_final.transform(web_df)
   
#     final_df = pd.DataFrame(user_input_scaled, index=web_df.index,columns=web_df.columns)

#     '''
#     isolating row with user input, converting to array for model prediction
#     '''

#     prediction_df = final_df.iloc[-1]
#     final_features = np.array(prediction_df).reshape(-1, 8)


#     prediction = loaded_model.predict(final_features)

#     final_prediction  = str(prediction)

#     prediction_deliv = ''
    
#     if final_prediction == '[[0.]]':
#       prediction_deliv = 'be Approved!'

#     else:
#         prediction_deliv = 'be denied. I am sorry.'

#     output = prediction_deliv
#     return render_template('model.html', prediction_text='Your loan application will likely {}'.format(output))


# if __name__ == "__main__":
    
#     app.run()