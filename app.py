import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import wordnet
import string
import itertools


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

@st.cache(suppress_st_warning=False)
def calculate_wordcloud(text):
    word_cloud = WordCloud(background_color='white',
                            width=1600, height=1000, 
                            max_words=len(text),
                            max_font_size=210, 
                            relative_scaling=.01,
                            collocations=False,
                            stopwords = stop_words).generate(text)
    return word_cloud


def display_wordcloud(wc):
    # Display the generated image:
    # https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    # Save to file first or an image file has already existed.
    wc_png = 'wordcloud.png'
    plt.savefig(wc_png, pad_inches=None , dpi=1200)
    col2.pyplot()

    with open(wc_png, "rb") as img:
        btn = col2.download_button(
                            label="Download image",
                            data=img,
                            file_name=wc_png,
                            mime="image/png")
        
lemmatizer = WordNetLemmatizer() #lemmatizer.lemmatize("rocks")
tokenizer = RegexpTokenizer(r'\b\w{3,}\b')

# https://docs.streamlit.io/library/api-reference 
# Load the model (only executed once!)
# Don't set ttl or max_entries in this case

nltk.download('stopwords')
nltk.download('punkt') 

st.set_page_config(page_title = 'Verbatims analysis', layout="wide")
st.header('Verbatims analysis')
st.subheader('Wordcloud:')

# st.sidebar.header('Options')

### --- LOAD DATA
st.markdown('The excel table should have columns: uid, answer, question, experiment_name plus filter/breakout columns')
uploaded_file = st.file_uploader('Drag a verbatims file here', type=['xlsx'])
sheet_name = st.text_input('Type which sheet you want to open')
submit = st.button('Submit')

if uploaded_file and sheet_name and submit:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.sheet_name = sheet_name
elif uploaded_file and sheet_name and not submit:
    st.text('You can upload another excel file or press "Submit"')
elif not uploaded_file and not sheet_name and not submit and 'uploaded_file' in st.session_state and 'sheet_name' in  st.session_state:
    st.text('You can upload another excel file or press "Submit"')
else:
    #  inputs are not filled
    st.text('Please upload excel file or press "Submit"')

if 'uploaded_file' in st.session_state and 'sheet_name' in st.session_state:
    #  only if input is in session we continue
    if not submit:
            # case after coming back to page withing the session
        if 'df_filtered' in st.session_state and ('refresh_filters' not in st.session_state or not st.session_state.refresh_filters):
            df = st.session_state.df_filtered
        elif 'df_filtered' in st.session_state and st.session_state.refresh_filters:
            df = pd.read_excel(st.session_state.uploaded_file,
                            sheet_name=st.session_state.sheet_name,
                        #    usecols='A:F',
                            header=0)
            # in case the uploaded file has a structure of excel with LINKS and the button = 'Go to Summary':
            df = df.dropna(how = 'all').reset_index(drop=True)
            if df.iloc[0, 0] == 'Go to Summary':
                df.columns = df.iloc[0, :].values
                df = df.iloc[1:, :]
                df = df.drop(columns = ['Go to Summary'])

            st.session_state.df  = df
        else:
            if 'df' not in st.session_state:
                # first input before df is defined
                df = pd.read_excel(st.session_state.uploaded_file,
                            sheet_name=st.session_state.sheet_name,
                        #    usecols='A:F',
                            header=0)
                if df.iloc[0, 0] == 'Go to Summary':
                    # in case the uploaded file has a structure of excel with LINKS and the button = 'Go to Summary':
                    df.columns = df.iloc[0, :].values
                    df = df.iloc[1:, :]
                    df = df.drop(columns = ['Go to Summary'])

                st.session_state.df  = df
            else:
                st.text('smth not expected line 79, because df_filtered should be defined anyway')

    elif submit:
        # means second or more input so we need to redefine our inputs and read df again
        st.session_state.uploaded_file = uploaded_file
        st.session_state.sheet_name = sheet_name
        df = pd.read_excel(st.session_state.uploaded_file,
                            sheet_name=st.session_state.sheet_name,
                        #    usecols='A:F',
                            header=0)
        # in case the uploaded file has a structure of excel with LINKS and the button = 'Go to Summary':
        df = df.dropna(how = 'all').reset_index(drop=True)
        if df.iloc[0, 0] == 'Go to Summary':
            df.columns = df.iloc[0, :].values
            df = df.iloc[1:, :]
            df = df.drop(columns = ['Go to Summary'])

        st.session_state.df  = df

    else:
        #  inputs are not filled
        st.text('You can upload another excel file or press "Submit"')

    

    # ADDING IMAGE AND DISPLAYING THE DF
    col1, col2 = st.columns(2)
    image = Image.open('images/hands-keyboard.jpg')
    col1.image(image,
            #  caption='got from Freepick',
            #  use_column_width=True,
            width = 400
            )
    col2.dataframe(st.session_state.df)

    # EXTRA input form
    extra_form = st.form(key="user_form")
    stopwords_to_add = extra_form.text_input('What stopwords do you want to add? (type words separated by commas)')
    stopwords_to_remove = extra_form.text_input('What stopwords you would like to remove? (type words separated by commas)')
    language = extra_form.text_input('Stopwords of which language do you want to use? \
                                (type f.e. "english", "french" etc):')
    # default
    if 'language' not in st.session_state:
        st.session_state.language = 'english'
    if 'stopwords_to_add' not in st.session_state:
        st.session_state.stopwords_to_add = {}
    if 'stopwords_to_remove' not in st.session_state:
        st.session_state.stopwords_to_remove = {}

    if extra_form.form_submit_button('Submit extra input'):
        stopwords_to_add_set = set([i.strip().lower() for i in stopwords_to_add.split(',')])
        stopwords_to_remove_set = set([i.strip().lower() for i in stopwords_to_remove.split(',')])
        if len(language) != 0:
            st.session_state.language = language
        if len(stopwords_to_add_set) > 0:
            st.session_state.stopwords_to_add = stopwords_to_add_set 
        if len(stopwords_to_remove_set) > 0:
            st.session_state.stopwords_to_remove = stopwords_to_remove_set 
        
        st.markdown(f'**Extra stopwords added**: {st.session_state.stopwords_to_add}')
        st.markdown(f'**Extra stopwords removed**: {st.session_state.stopwords_to_remove}')
        st.markdown(f'**Language added**: {st.session_state.language}')
        st.markdown("*Standard language is English, if no language added") 
    else:
         st.markdown('Default or previously set parameters are being used')
         st.markdown(f'**Extra stopwords added**: {st.session_state.stopwords_to_add}')
         st.markdown(f'**Extra stopwords removed**: {st.session_state.stopwords_to_remove}')
         st.markdown(f'**Language added**: {st.session_state.language}')

    ## _____________________________________________________________________________________________________________
    # SELECTION BOX AND WORDCLOUD
    col1, col2 = st.columns(2)
    df['answer'] = df['answer'].fillna('-')
    nb_ = len(df[df['answer']=='-'])
    if 'empty' not in st.session_state:
        st.session_state.empty = nb_
    col2.markdown(f'Number of empty answers in the data: {st.session_state.empty} >> drop for the analysis') 

    st.session_state.df['answer'] = st.session_state.df['answer'].fillna('-')
    st.session_state.df = st.session_state.df[st.session_state.df.answer != '-']
    col2.markdown(f'Nb of respondents in the data: {st.session_state.df.uid.nunique()}') #ok
    col2.markdown(f'Shape of current data: {st.session_state.df.shape}') #ok

    # lock the options in the first run
    if 'nb_cols' not in st.session_state and 'df_cols' not in st.session_state:
        st.session_state.nb_cols = len([i for i in st.session_state.df.columns if i not in ['uid', 'answer']])
        st.session_state.df_cols = [i for i in st.session_state.df.columns if i not in ['uid', 'answer']]


    # refresh_all_filters = st.button('Refresh all the filters', key  = 'refresh_filters')
    # if refresh_all_filters:
    #     st.session_state.df_filtered = st.session_state.df.copy()


    for col in st.session_state.df_cols:
        if not any(' | ' in str(i) for i in st.session_state.df[col].unique()):
            globals()[f'{col}_options'] = st.session_state.df[col].unique().tolist()
        else:
            options_ = list(itertools.chain.from_iterable([a.split(' | ') 
                                for a in set([i for i in st.session_state.df[col].unique()])]))
            globals()[f'{col}_options'] = [*set(options_)]

        # adding MULTISELECT for the specific breakout/question:
        globals()[f'{col}_selection'] = col1.multiselect(f'{col}:',
                                globals()[f'{col}_options'],
                                default = globals()[f'{col}_options'],
                                label_visibility = "collapsed")
        
    # so far we just created the multiselct objects themselves, which are not connected to the data. 
    # next we need to actually filter out dataframe and connect it to the wordcloud function
        
    # --- FILTER DATAFRAME BASED ON SELECTION
    mask = []
    for col in st.session_state.df_cols:
        if not any(' | ' in str(i) for i in st.session_state.df[col].unique()):
            mask.append((st.session_state.df[col].isin(globals()[f'{col}_selection'])))
        else:
            multi_mask = []
            for opt in globals()[f'{col}_selection']:
                multi_mask.append((st.session_state.df[col].str.contains(opt)))
            mask.append(multi_mask)
            
    # df_filtered = df.copy()

    # ADD df_filtered to the current session state:
    if 'df_filtered' not in st.session_state:
        st.session_state.df_filtered = st.session_state.df.copy() #was df_filtered.copy()
    
    #here df_filtered is still ok
    uids_filter = []
    for cond in mask:
        if type(cond) == list:
            cond_multi = pd.concat(cond, axis=1)
            cond_x = cond_multi.any(axis='columns')
            df_filtered_x = st.session_state.df_filtered[cond_x]
            st.text(df_filtered_x.uid.nunique())
            uids_filter += df_filtered_x.uid.unique().tolist()
            # st.session_state.df_filtered = df_filtered_x
        else:
            df_filtered_x = st.session_state.df_filtered[cond]
            st.text(df_filtered_x.uid.nunique())
            uids_filter += df_filtered_x.uid.unique().tolist()
            # st.session_state.df_filtered = df_filtered_x
    uids_filter = [*set(uids_filter)]
    st.text(len(uids_filter))

    # number_of_result = df_filtered_x.shape[0]
    # col2.markdown(f'**Available results:** {number_of_result}')


    # stop_words = set(stopwords.words(st.session_state.language))

    # if len(st.session_state.stopwords_to_add) > 0:
    #     stop_words.update(st.session_state.stopwords_to_add)

    # if len(st.session_state.stopwords_to_remove) > 0:
    #     stop_words = stop_words - st.session_state.stopwords_to_remove

    # st.session_state.stop_words = stop_words

    # # ---- ADD WORDCLOUD
    
    # corpus = df_filtered_x.answer.unique().tolist() #was df_filtered, change 07.07.23 16:34
    # corpus = [i.lower() for i in corpus]
    # text = ' '.join(corpus)
    # col2.markdown(f'Total nb of words: {len(text)}')

    # for i in ['-', '  ', '’', "\'"]: # drop extra symbols
    #     if i != '’':
    #         text = text.replace(i, '')
    #     else:
    #         text = text.replace(i, "'")
    # # st.text(text)
    # text = text.translate(str.maketrans('', '', string.punctuation))

    # # LEMMATIZE
    # try:
    #     text = lemmatize_sentence(text)
    # except:
    #     nltk.download('all')
    #     text = lemmatize_sentence(text)


    # # Create and generate a word cloud image:
    # if 'wordcloud' not in st.session_state:    
    #     with st.spinner('Wait for it...'):
    #         wordcloud = calculate_wordcloud(text)
    #     st.session_state.wordcloud = wordcloud
    #     with st.spinner('Wait for it...'):
    #         display_wordcloud(st.session_state.wordcloud)
            
    
    # regenerate_wordcloud = col2.button('Generate wordcloud (or regenerate to refresh)')
    # if regenerate_wordcloud:
    #     with st.spinner('Wait for it...'):
    #         wordcloud = calculate_wordcloud(text)
    #     st.session_state.wordcloud = wordcloud
    #     with st.spinner('Wait for it...'):
    #         display_wordcloud(st.session_state.wordcloud)




