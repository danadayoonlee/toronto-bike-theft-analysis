'''
5. Deploying the model
5.1	Using flask framework arrange to turn your selected machine-learning model into an API.
5.2	Using pickle module arrange for Serialization & Deserialization of your model.
5.3	Build a client to test your model API service. Use the test data, which was not previously used to train the module.
You can use simple Jinja HTML templates with or without Java script, REACT or any other technology but at minimum use POSTMAN Client API.
'''

from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
import sys
import logging 
logging.basicConfig(format='%(asctime)s %(message)s')


# Creating an object
logger=logging.getLogger()

# Setting the threshold of logger to DEBUG 
logger.setLevel(logging.DEBUG) 

# Your API definition
app = Flask(__name__)
@app.route("/lg9", methods=['GET','POST']) # Use decorator pattern for the route
def predict():
    if lr:
        try:
            json_ = request.json
            logger.info(json_)
            print(json_, file=sys.stderr)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            print(query,file=sys.stderr)
            logger.info(query)
            from sklearn import preprocessing
            scaler = preprocessing.StandardScaler()
            # Fit your data on the scaler object
            scaled_df = scaler.fit_transform(query)
            # Return to data frame
            query = pd.DataFrame(scaled_df, columns=model_columns)
            print(query,file=sys.stderr)
            logger.info(query)
            prediction = list(lr.predict(query))
            print({'prediction': str(prediction)}, file=sys.stderr)
            logger.info(prediction)
            return jsonify({'prediction': str(prediction)})            

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        logger.info('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12999 # If you don't provide any port the port will be set to 12345
 
    # Change to your local path
    modelpath = "C:/Users/User/Desktop/model_lr.pkl"
    lr = joblib.load(modelpath) # Load "model file model.pkl"
    print ('Model loaded')
    logger.info('Model loaded')

    # Change to your local path
    modelcolumnpath = 'C:/Users/User/Desktop/model_columns_lr.pkl'
    model_columns = joblib.load(modelcolumnpath) # Load "model_columns.pkl"
    print ('Model columns loaded')
    logger.info('Model columns loaded')
    app.run(port=port, debug=True)