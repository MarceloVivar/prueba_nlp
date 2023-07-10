import streamlit as st
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import nltk
import matplotlib.pyplot as plt
from heapq import nlargest
import requests
import seaborn as sns
from wordcloud import WordCloud
from io import BytesIO

def main():
    st.title("Análisis Cuantitativo de Textos")
    st.sidebar.title("Menu web app")
    #st.sidebar.markdown("Parámetros")

    st.set_option('deprecation.showPyplotGlobalUse', False)
    texto = st.text_input("Ingresa tu texto")
    

    if not texto:
        st.info("Pega tu texto en el campo de arriba.")
    elif st.button("ver texto"):
        st.write("Texto ingresado:", texto)
        
    if st.button("borrar texto"):   
            texto=""
            
    freq= st.sidebar.button("Frecuencia de palabras")
    max_freq=st.sidebar.button("Maxima frecuencia")
    plot_barras=st.sidebar.button("Gráfico de barras")
    nube=st.sidebar.button("Gráfico de nube de palabras")
    resumen=st.sidebar.button("Generar resumen")
    plot_pos=st.sidebar.button("Gráfico de palabras positivas")
    plot_neg=st.sidebar.button("Gráfico de palabras negativas")
    
    if texto:
    
        nlp = spacy.load('es_core_news_sm')
        nlp = es_core_news_sm.load()
        doc = nlp(texto)        

        if freq:
            frequencies = word_frequencies(doc)
            st.write("Frecuencia de palabras:", frequencies)

        if max_freq:
            max_frequency_value = max_frequency(doc)
            st.write("Maxima frecuencia de palabras:", max_frequency_value)

        df = transform_df(word_frequencies(doc))
                # st.write(df)

        if plot_barras:
                st.title("Plot de gráfico de barras")
                fig_barras = plot_frecuencia(df)
                # Convertir la figura a bytes
                buffer = BytesIO()
                fig_barras.savefig(buffer, format="png")
                buffer.seek(0)
                st.image(buffer)
                # Botón de descarga
                st.download_button(
                    label="Descargar gráfico de barras",
                    data=buffer,
                    file_name="grafico_barras.png",
                    mime="image/png"
                )


        if nube:
                st.title("Nube de Palabras desde DataFrame")
                fig_nube = generate_wordcloud_from_df(df)
                # Convertir la figura a bytes
                buffer = BytesIO()
                fig_nube.savefig(buffer, format="png")
                buffer.seek(0)
                st.image(buffer)
                # Botón de descarga
                st.download_button(
                    label="Descargar gráfico de nube de palabras",
                    data=buffer,
                    file_name="grafico_wordcloud.png",
                    mime="image/png"
                )

                
        normalized_frequencies = normalize(word_frequencies(doc))

        sentence_tokens = [sent for sent in doc.sents]

        sentence_scores = calculate_sentence_scores(sentence_tokens, normalized_frequencies)
        # df_scores = pd.DataFrame.from_dict(sentence_scores, orient='index', columns=['Puntaje'])
        # st.write(df_scores)

        select_length = int(len(sentence_tokens) * 0.02)
        summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
        # st.write("Resumen:", summary)

        if resumen:
                generated_summary = generate_summary(doc, summary)
                st.write("Resumen generado:", generated_summary)

        afinn = load_data()
        # st.write(afinn)

        reviews_afinn = join_afinn_scores(df, afinn)
        # st.write(reviews_afinn)

        top_words = filter_top_words(reviews_afinn, 30)
        # st.write(top_words)

        positive_words, negative_words = separate_positive_negative(top_words)
        
                    
        if plot_pos:
                st.title("Plot de palabras positivas")
                fig_pos = (plot_positivas(positive_words))
                # Convertir la figura a bytes
                buffer = BytesIO()
                fig_pos.savefig(buffer, format="png")
                buffer.seek(0)
                st.image(buffer)
                # Botón de descarga
                st.download_button(
                    label="Descargar gráfico de palabras positivas",
                    data=buffer,
                    file_name="grafico_positivas.png",
                    mime="image/png"
                )

        if plot_neg:
                st.title("Plot de palabras negativas")
                fig_neg = (plot_negativas(negative_words))
                # Convertir la figura a bytes
                buffer = BytesIO()
                fig_neg.savefig(buffer, format="png")
                buffer.seek(0)
                st.image(buffer)
                # Botón de descarga
                st.download_button(
                    label="Descargar gráfico de palabras Negativas",
                    data=buffer,
                    file_name="grafico_negativas.png",
                    mime="image/png"
                )
  

       

def load_stopwords():
    try:
        data = open('stopwords-es.txt', 'r', encoding='utf-8').read()
        return data
    except FileNotFoundError:
        return None
    except Exception as e:
        return str(e)

def word_frequencies(doc):
    stopwords = load_stopwords()
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1
    return word_frequencies

def max_frequency(doc):
    frequencies = word_frequencies(doc)
    max_frequency_value = max(frequencies.values())
    return max_frequency_value

def transform_df(frequencies):
    df = pd.DataFrame.from_dict(frequencies, orient='index', columns=['Frecuencia'])
    df.index.name = 'Palabra'
    df.reset_index(inplace=True)
    return df

def plot_frecuencia(df):
    fig, ax = plt.subplots()
    df[df['Frecuencia'] > 2].sort_values("Frecuencia", ascending=False).plot.bar(x="Palabra", y="Frecuencia", ax=ax)
    ax.set_title("Plot de gráfico de barras")
    ax.set_xlabel("Palabra")
    ax.set_ylabel("Frecuencia")
    return fig

def generate_wordcloud_from_df(df):
    text = " ".join(df['Palabra'].astype(str).tolist())
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", colormap='bone').generate(text)
    plt.figure(figsize=(11, 9))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return plt.gcf()

def normalize(frequencies):
    max_frequency = max(frequencies.values())
    normalized_frequencies = {word: frequency / max_frequency for word, frequency in frequencies.items()}
    return normalized_frequencies

def calculate_sentence_scores(sentence_tokens, normalized_frequencies):
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in normalized_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = normalized_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += normalized_frequencies[word.text.lower()]
    return sentence_scores

def generate_summary(doc, summary):
    summary_sentences = [str(sent) for sent in summary]
    generated_summary = " ".join(summary_sentences)
    return generated_summary

def load_data():
    data = pd.read_csv("lexico_afinn_en_es.csv", encoding="ISO-8859-1")
    return data

def join_afinn_scores(df, afinn):
    reviews_afinn = pd.merge(df, afinn, on='Palabra', how='inner')
    reviews_afinn = reviews_afinn.groupby('Palabra').agg(
        occurences=('Palabra', 'count'),
        contribution=('Puntuacion', 'sum')
    ).reset_index()
    return reviews_afinn

def filter_top_words(reviews_afinn, num_words):
    top_words = reviews_afinn.nlargest(num_words, 'contribution')
    top_words = top_words.sort_values('contribution')
    return top_words

def separate_positive_negative(top_words):
    positive_words = top_words[top_words['contribution'] > 0]
    negative_words = top_words[top_words['contribution'] < 0]
    return positive_words, negative_words

def plot_positivas(positive_words):
    fig, ax = plt.subplots()
    sns.barplot(data=positive_words, y='Palabra', x='contribution', color='green', ax=ax)
    ax.set_title("Plot de palabras positivas")
    ax.set_xlabel("Contribución")
    ax.set_ylabel("Palabra")
    return fig

def plot_negativas(negative_words):
    fig, ax = plt.subplots()
    sns.barplot(data=negative_words, y='Palabra', x='contribution', color='red', ax=ax)
    ax.set_title("Plot de palabras negativas")
    ax.set_xlabel("Contribución")
    ax.set_ylabel("Palabra")
    return fig


if __name__ == '__main__':
    main()
