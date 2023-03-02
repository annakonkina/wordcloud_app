import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import wordnet
import string

def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_sentence(sentence):

    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), nltk_tagged)
    lemmatized_sentence = []
    
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:        
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


lemmatizer = WordNetLemmatizer() #lemmatizer.lemmatize("rocks")
tokenizer = RegexpTokenizer(r'\b\w{3,}\b')


# https://docs.streamlit.io/library/api-reference 

# nltk.download('stopwords') #IMPORTANT TO DOWNLOAD FIRST TIME
# nltk.download('punkt') #IMPORTANT TO DOWNLOAD FIRST TIME

st.set_page_config(page_title = 'Verbatims analysis', layout="wide")
st.header('Verbatims analysis')
st.subheader('Wordcloud:')

# st.sidebar.header('Options')

### --- LOAD DATA
st.markdown('The excel table should have columns: uid, answer, question, experiment_name plus filter/breakout columns')
uploaded_file = st.file_uploader('Drag a verbatims file here', type=['xlsx'])
sheet_name = st.text_input('Type which sheet you want to open')

if uploaded_file and sheet_name:

        df = pd.read_excel(uploaded_file,
                   sheet_name=sheet_name,
                #    usecols='A:F',
                   header=0)


        # ---DISPLAY AS 2 COLUMNS (picture and the table)
        col1, col2 = st.columns(2)

        image = Image.open('images/hands-keyboard.jpg')

        col1.image(image,
                #  caption='got from Freepick',
                #  use_column_width=True,
                width = 200
                )

        col2.dataframe(df)
        # SELECTION BOX AND WORDCLOUD
        col1, col2 = st.columns(2)

        nb_cols = len([i for i in df.columns if i not in ['uid', 'answer']])
        df_cols = [i for i in df.columns if i not in ['uid', 'answer']]
        mask = []
        for i in range(nb_cols):
                globals()[f'{i}_options'] = df[df_cols[i]].unique().tolist()
                globals()[f'{i}_selection'] = col1.multiselect(f'{df_cols[i]}:',
                                        globals()[f'{i}_options'],
                                        default = globals()[f'{i}_options'])
        
        # --- FILTER DATAFRAME BASED ON SELECTION
        for i in range(nb_cols):
                mask.append((df[df_cols[i]].isin(globals()[f'{i}_selection'])))

        df_filtered = df.copy()
        for cond in mask:
                df_filtered = df_filtered[cond]
        
        number_of_result = df_filtered.shape[0]
        col2.markdown(f'**Available results:** {number_of_result}')
        # col2.markdown(f'**Check df len:** {df.shape[0]}')

        # ---- ADD WORDCLOUD
        corpus = df_filtered.answer.unique().tolist()
        for i in ['-', '  ', '’', "\'"]: # drop extra symbols
                corpus = [a.replace(i, '').lower() 
                          for a in corpus]
        text = ' '.join(corpus)
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # LEMMATIZE
        try:
              text = lemmatize_sentence(text)
        except:
              nltk.download()
              text = lemmatize_sentence(text)

        # Create and generate a word cloud image:
        stop_words = STOPWORDS

        wordcloud = WordCloud(background_color='white',
                        width=1600, height=1000, 
                        max_words=len(text),
                        max_font_size=210, 
                        relative_scaling=.01,
                        collocations=False,
                        stopwords = stop_words).generate(text)

        # Display the generated image:
        # https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        # Save to file first or an image file has already existed.
        wc = 'wordcloud.png'
        plt.savefig(wc, pad_inches=None , dpi=1200)
        col2.pyplot()

        
        with open(wc, "rb") as img:
                btn = col2.download_button(
                        label="Download image",
                        data=img,
                        file_name=wc,
                        mime="image/png",
                        
                )

# #  --- GROUP DATAFRAME
# df_grouped = df[mask].groupby('experiment_name').uid.nunique().reset_index()
# df_grouped.rename(columns={'uid':'nb_uids'}, inplace=True)


# bar_chart = px.bar(df_grouped,
#                    x='experiment_name',
#                    y = 'nb_uids',
#                    color_discrete_sequence = ['#F63366']*len(df_grouped),
#                    template='plotly_white',

#                    )
# st.plotly_chart(bar_chart)

# age_selection = (30,40)
# mask = df['age'].between(*age_selection)
# df[mask]


# ages = df.age.unique().tolist()
# age_selection = st.slider('Age:',
#           min_value = min(ages),
#           max_value = max(ages),
#           value = (min(ages), max(ages)))





