from loguru import logger

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report

from services.spam_detect import SpamEmailDetect

@logger.catch
def main(
    data_source = './assets/spam_data.csv'
    ):

    # Load data from ./assets/spam_data.csv
    df = pd.read_csv(data_source)

    # Downsampling 'Ham' data to avoid overfitting
    df_ham = df[df['Category'] == 'ham']
    df_spam = df[df['Category'] == 'spam']

    df_ham_downsampled = df_ham.sample(df_spam.shape[0])
    df_balanced = pd.concat([df_spam , df_ham_downsampled])

    # Preprocessing data
    df_balanced['spam'] = df_balanced['Category'].apply(lambda x:1 if x == 'spam' else 0)

    # Loading train-test-split
    X_train, X_test , y_train, y_test = train_test_split(df_balanced['Message'], df_balanced['spam'],
                                                    stratify = df_balanced['spam'])
    
    # Instantiate model
    sed = SpamEmailDetect()
    logger.debug("----------Model Summary----------")
    sed.summary()

    # Compile model
    sed.compile()

    # Fit data into model
    logger.debug("----------Model Fitting----------")
    sed.fit(X_train, y_train, epochs=10)

    # Make prediction and evaluate
    logger.debug("----------Model Evaluation----------")
    sed.evaluate(X_test, y_test)

    y_pred = sed.predict(X_test)
    y_pred = np.round(y_pred.flatten())

    # Confusion matrix
    logger.debug("----------Confusion Matrix----------")
    cm = confusion_matrix(y_test, y_pred)
    df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
    print(df)

    # Classification report
    logger.debug("----------Classification Report----------")
    cr = classification_report(y_test, y_pred)
    print(cr)

if __name__ == "__main__":
    main()