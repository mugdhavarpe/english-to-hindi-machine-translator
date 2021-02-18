
import string
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import re
import seaborn as sns
from nltk.stem import PorterStemmer
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np


def main():
    """
    This is the main method. It is used for driving all the other methods.
    :return: None
    """
    warnings.filterwarnings("ignore")
    # Load in the dataframe
    df = pd.read_csv("Hindi_English_Truncated_Corpus.csv", encoding='utf-8')
    df = df.sample(n=65000)
    new_data = clean_data(df)
    new_data = [line.split("\t") for line in new_data]
    lines = [("\t" + line[0] + "\n", "\t" + line[1] + "\n") for line in new_data]
    english_words, hindi_words = get_words(lines)
    visualize_data(df, english_words, hindi_words)
    max_en_length = max([len(word) for word in [(line[0]) for line in lines]])
    max_de_length = max([len(word) for word in [(line[1]) for line in lines]])
    input_dict = create_dict(english_words)
    output_dict = create_dict(hindi_words)
    encoder_input_data = create_np_array([(line[0]) for line in lines], max_en_length, len(english_words))
    decoder_input_data = create_np_array([(line[1]) for line in lines], max_de_length, len(hindi_words))
    decoder_target_data = decoder_input_data
    for idx, (input_text, target_text) in enumerate(zip([(line[0]) for line in lines], [(line[1]) for line in lines])):
        for idx2, char in enumerate(input_text):
            encoder_input_data[idx, idx2, input_dict[char]] = 1.
        for idx2, char in enumerate(target_text):
            decoder_input_data[idx, idx2, output_dict[char]] = 1.
            if idx2 > 0:
                decoder_target_data[idx, idx2 - 1, output_dict[char]] = 1.
    model(english_words, hindi_words, encoder_input_data, decoder_input_data, decoder_target_data)


def model(english_words, hindi_words, encoder_input_data, decoder_input_data, decoder_target_data):
    """
    This is the implementation form the LSTM model, it trains the model and saves the same.
    :param english_words: Set of English words
    :param hindi_words: Set of Hindi Words
    :param encoder_input_data: Encoded input data
    :param decoder_input_data: Decoded input data
    :param decoder_target_data: Decoded target data
    :return:
    """
    encoder_inputs = Input(shape=(None, len(english_words)))
    encoder = LSTM(256, return_state=True, dropout=0.4)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, len(hindi_words)))
    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)
    decoder_dense = Dense(len(hindi_words), activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=128, epochs=200,
              validation_split=0.25)
    model.save('s2s.h5')


def create_dict(words):
    """
    Method for creating dictionary.
    :param words: English or Hindi words
    :return: dictionary
    """
    dictionary = {}
    for i, char in enumerate(words):
        dictionary[char] = i
    return dictionary


def create_np_array(english_texts, sequence_length, tokens):
    # This method initializes the numpy array.
    return np.zeros((len(english_texts), sequence_length, tokens), dtype='float32')


def make_word_clouds(df, stop_words, language):
    """
    This method is used to plot word clouds for english or hindi sentences given in the dataframe.
    :param df: dataframe of hindi or english sentences
    :param stopwords: stopwords for the hindi language
    :param language: language for which word cloud has to be formed
    :return:None
    """
    word_string = ' '
    increment = 0
    for sentence in df:
        sentence = str(sentence)
        tokens = sentence.split()
        for index in range(len(tokens)):
            tokens[index] = tokens[index].lower()
        word_string = word_string + " ".join(tokens) + " "
        increment += 1
        if increment == 3000:
            break
    if language == 'english':
        word_cloud = WordCloud(width=800, height=800,
                               background_color='white', collocations=False, stopwords=stop_words,
                               min_font_size=10).generate(word_string)
    else:
        # Create and generate a word cloud image:
        word_cloud = WordCloud(font_path='Lohit-Devanagari.ttf', width=800, height=800,
                               background_color='white', collocations=False, stopwords=stop_words,
                               min_font_size=10).generate(word_string)
    # Display the generated image:
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def hindi_stop_words_processing():
    """
    This method is used to process the stop words list for hindi language.
    :return: stopwords list
    """
    words = open("hindi_stopwords.txt", 'r', encoding='UTF-8')
    stop_words = []
    # pre-process hindi stopword
    for word in words:
        word = re.sub('[\n]', '', word)
        stop_words.append(word)
    return stop_words


def clean_data(df):
    """
     This method is used to clean the data sentences.
     Data cleaning is done by: removing null values,
     converting the data to string, convertaing the sentences to lower cases,
     removing numbers, and by removing punctuations for both English and Hindi
     sentences
    :param df: dataframe of hindi or english sentences
    :return:  Clean dataframe
    """
    # Cleaning english sentences
    df = df[df['english_sentence'].notnull()]
    df['english_sentence'].astype(str)
    df.drop_duplicates(inplace=True)
    df['english_sentence'] = df['english_sentence'].str.lower()
    df['english_sentence'] = df['english_sentence'].str.replace('\d+', '')
    df['english_sentence'] = df['english_sentence'].str.replace('[{}]'.format(string.punctuation), '')

    # Cleaning hindi sentences
    df = df[df['hindi_sentence'].notnull()]
    df['hindi_sentence'].astype(str)
    df['hindi_sentence'] = df['hindi_sentence'].str.lower()
    df['hindi_sentence'] = df['hindi_sentence'].str.replace('\d+', '')
    df['hindi_sentence'] = df['hindi_sentence'].str.replace('[{}]'.format(string.punctuation), '')

    # Concatenating english and hindi sentences
    df['Concat'] = df["english_sentence"].astype(str) + "\t" + df["hindi_sentence"]
    new_data = df['Concat'].values.tolist()
    # print(df)
    return new_data


def visualize_data(df, english_words, hindi_words):
    """
    This method is used to visualize the data.
    :param df: Data-frame of sentences
    :param english_words: English words obtained from the sentences.
    :param hindi_words: Hindi words obtained from the sentences.
    :return: None
    """
    # Get total sentences present in different sources
    print("Data count for the respective data sources: \n" + str(df['source'].value_counts()))

    # plot the graph for distribution of sentences in the sources
    sns.barplot(df['source'].value_counts().index, df['source'].value_counts().values)
    plt.ylabel('Data count', fontsize=10)
    plt.xlabel('Sources', fontsize=10)
    plt.show()

    # Getting Hindi stopwords
    stop_word = open("hindi_stopwords.txt", 'r', encoding='utf-8')
    stop_words = []
    # pre-process stopword
    for i in stop_word:
        i = re.sub('[\n]', '', i)
        stop_words.append(i)

    stopwords_hindi = set(stop_words)
    stopwords_english = set(STOPWORDS)

    # Generating word clouds for English and Hindi sentences from data-set
    make_word_clouds(df.english_sentence, stopwords_english, 'english')
    make_word_clouds(df.hindi_sentence, stopwords_hindi, 'hindi')

    # Displaying total number of unique words present for sentences of languages.
    print("\nNumber of unique English words: " + str(len(english_words)))
    print("Number of unique Hindi words: " + str(len(hindi_words)))


def get_words(lines, english_words=set(), hindi_words=set()):
    """
    This method is used for obtaining unique words from the sentences.
    :param df: dataframe of hindi or english sentences
    :param english_words: Set to obtain unique English words
    :param hindi_words: Set to obtain unique Hindi words
    :return:
    """
    ps = PorterStemmer()
    # Obtaining English words
    for eng in [(line[0]) for line in lines]:
        arr = [ps.stem(word) for word in eng if word not in english_words]
        english_words.update(arr)

    # Obtaining Hindi words
    for hin in [(line[1]) for line in lines]:
        arr = [ps.stem(word) for word in hin if word not in hindi_words]
        hindi_words.update(arr)
    # print(sorted(list(english_words)), sorted(list(hindi_words)))
    return sorted(list(english_words)), sorted(list(hindi_words))


if __name__ == '__main__':
    main()
