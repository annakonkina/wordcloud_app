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
from st_aggrid import AgGrid, GridUpdateMode, DataReturnMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder


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
# Load the model (only executed once!)
# Don't set ttl or max_entries in this case

# nltk.download('stopwords')
# nltk.download('punkt') 

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
    st.text('You can upload another excel file')
elif not uploaded_file and not sheet_name and not submit and 'uploaded_file' in st.session_state and 'sheet_name' in  st.session_state:
    st.text('You can upload another excel file')
else:
    #  inputs are not filled
    st.text('Please upload excel file')

if 'uploaded_file' in st.session_state and 'sheet_name' in st.session_state:
    #  only if input is in session we continue
    if not submit:
            # case after coming back to page withing the session
        if 'df_filtered' in st.session_state:
            df = st.session_state.df_filtered
        else:
            if 'df' not in st.session_state:
                # first input before df is defined
                df = pd.read_excel(st.session_state.uploaded_file,
                            sheet_name=st.session_state.sheet_name,
                        #    usecols='A:F',
                            header=0)
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
        st.session_state.df  = df

    else:
        #  inputs are not filled
        st.text('You can upload another excel file')

    # aggrid
    gd = GridOptionsBuilder.from_dataframe(df)
    gd.configure_pagination(enabled=True)
    gd.configure_default_column(groupable=True)
    gd.configure_selection(selection_mode='single',
                        #    use_ckeckbox=True
                            )
    gridOptions = gd.build()
    grid_table = AgGrid(df,
                        gridOptions = gridOptions,
            fit_columns_on_grid_load=True,
            height=500,
            width='100%',
            theme='streamlit',
            update_mode = GridUpdateMode.GRID_CHANGED,
            reload_data=True,
            allow_unsafe_jscode=True,
            editable=True)
    df_interactive = grid_table['data']

    # ADDING IMAGE AND DISPLAYING THE DF
    col1, col2 = st.columns(2)
    image = Image.open('images/hands-keyboard.jpg')
    col1.image(image,
            #  caption='got from Freepick',
            #  use_column_width=True,
            width = 400
            )
    if 'submit_extra_input' not in st.session_state:
        st.session_state.submit_extra_input = False
    if st.session_state.submit_extra_input:
        col2.markdown(f'**Extra stopwords added**: {st.session_state.stopwords_to_add}')
        col2.markdown(f'**Extra stopwords removed**: {st.session_state.stopwords_to_remove}')
        col2.markdown(f'**Language added**: {st.session_state.language}')
    else:
        col2.markdown(f'No extra stopwords or language have been input >> standard stopwords and English language')

    

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
    # ADD df_filtered to the current session state:
    if 'df_filtered' not in st.session_state:
        st.session_state.df_filtered = df_filtered

    for cond in mask:
        df_filtered = df_filtered[cond]
        st.session_state.df_filtered = df_filtered

    number_of_result = df_filtered.shape[0]
    col2.markdown(f'**Available results:** {number_of_result}')
    col2.markdown('If you want to add or remove extra stopwords and set language, please, fill the fields below (pressing Enter) \
                  and then press [Submit extra stopwords and language] button above')

    # STOPWORDS
    language = col2.text_input('Stopwords of which language do you want to use? \
                                (type f.e. "english", "french" etc)')
    if language and st.session_state.submit_extra_input:
        st.session_state.language = language
    else:
        if 'language' not in st.session_state or not st.session_state.submit_extra_input:
            st.session_state.language = 'english'

    stop_words = set(stopwords.words(st.session_state.language))

    stopwords_to_add = col2.text_input('What stopwords do you want to add? (type words separated by commas)')
    stopwords_to_remove = col2.text_input('What stopwords you would like to remove? (type words separated by commas)')

    stopwords_to_add_set = set([i.strip().lower() for i in stopwords_to_add.split(',')])
    stopwords_to_remove_set = set([i.strip().lower() for i in stopwords_to_remove.split(',')])

    # updating session state
    st.session_state.submit_extra_input = False
    submit_extra_input = col2.button('Submit extra stopwords and language')
    if submit_extra_input:
        st.session_state.submit_extra_input = True

    if stopwords_to_add and st.session_state.submit_extra_input:
        st.session_state.stopwords_to_add = stopwords_to_add_set
    else:
        if 'stopwords_to_add' not in st.session_state or not st.session_state.submit_extra_input:
            st.session_state.stopwords_to_add = {}
    if stopwords_to_remove and st.session_state.submit_extra_input:
        st.session_state.stopwords_to_remove = stopwords_to_remove_set
    else:
        if 'stopwords_to_remove' not in st.session_state or not st.session_state.submit_extra_input:
            st.session_state.stopwords_to_remove = {}

    if len(st.session_state.stopwords_to_add) > 0 and st.session_state.stopwords_to_add != set(['']):
        stop_words.update(st.session_state.stopwords_to_add)

    if len(st.session_state.stopwords_to_remove) > 0 and st.session_state.stopwords_to_remove != set(['']):
        stop_words = stop_words - st.session_state.stopwords_to_remove
    # reset
    st.session_state.submit_extra_input = False
    # ---- ADD WORDCLOUD
    col2.text(f'Number of empty answers in the data: {df_filtered.answer.isna().sum()}') 
    df_filtered['answer'] = df_filtered['answer'].fillna('-')
    corpus = df_filtered.answer.unique().tolist()
    corpus = [i.lower() for i in corpus]
    text = ' '.join(corpus)

    for i in ['-', '  ', '’', "\'"]: # drop extra symbols
        if i != '’':
            text = text.replace(i, '')
        else:
            text = text.replace(i, "'")
    # st.text(text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # LEMMATIZE
    try:
            text = lemmatize_sentence(text)
    except:
            nltk.download('all')
            text = lemmatize_sentence(text)

    regenerate_wordcloud = col2.button('Regenerate wordcloud')

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
        # download button
        with open(wc_png, "rb") as img:
            btn = col2.download_button(
                        label="Download image",
                        data=img,
                        file_name=wc_png,
                        mime="image/png")
         
    # Create and generate a word cloud image:
    if 'wordcloud' not in st.session_state:
        with st.spinner('Wait for it...'):
            wordcloud = WordCloud(background_color='white',
                            width=1600, height=1000, 
                            max_words=len(text),
                            max_font_size=210, 
                            relative_scaling=.01,
                            collocations=False,
                            stopwords = stop_words).generate(text)
        st.session_state.wordcloud = wordcloud
        with st.spinner('Wait for it...'):
            display_wordcloud(st.session_state.wordcloud)
    
    regenerate_wordcloud = col2.button('Regenerate wordcloud')
    if regenerate_wordcloud:
        with st.spinner('Wait for it...'):
            wordcloud = WordCloud(background_color='white',
                            width=1600, height=1000, 
                            max_words=len(text),
                            max_font_size=210, 
                            relative_scaling=.01,
                            collocations=False,
                            stopwords = stop_words).generate(text)
        st.session_state.wordcloud = wordcloud
        with st.spinner('Wait for it...'):
            display_wordcloud(st.session_state.wordcloud)





