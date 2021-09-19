from loguru import logger

import pandas as pd

@logger.catch
def main(
    data_source = './assets/spam_data.csv'
    ):

    # Load data from ./assets/spam_data.csv
    logger.info("Load Data")
    df = pd.read_csv(data_source)

    # Downsampling 'Ham' data to avoid overfitting
    df_ham = df[df['Category'] == 'ham']
    df_spam = df[df['Category'] == 'spam']

    df_ham_downsampled = df_ham.sample(df_spam.shape[0])
    df_balanced = pd.concat([df_spam , df_ham_downsampled])

    logger.info(df_balanced['Category'].value_counts())


if __name__ == "__main__":
    main()