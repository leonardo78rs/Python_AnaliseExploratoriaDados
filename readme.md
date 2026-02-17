# 🎯 Análise Exploratória de dados 
Um pequeno script em Python para análise de dados. Focando nos primeiros passos quando estamos explorando as condições e possibilidades do nosso conjunto de dados. 

## 🚀 Funcionalidades
    - Ler arquivo CSV
    - Investigar rapidamente os tipos e intervalos de valores  
    - Valores únicos, frequência, somas, médias e medianas
    - Plotar alguns tipos de gráficos: Histograma, Diagrama de caixa (bloxplot), Gráfico de barras, entre outros
    - Identificar as propriedades da amostra de dados para melhor entendimento
    
## 🛠️ Tecnologias 
    Linguagem: Python
    Libs: pandas, seaborn, matplotlib, numpy
       
## 📋 Pré-requisitos

    Python ou Google Colaboratory

## ⚙ Instalação e Uso
    O que você precisa para rodar este projeto:
    1. Clonar o repositório (git clone ....) e executar
    2. ou carregar o notebook jupiter deste repositório diretamente no colab.google.  
    
## 📸 Visualizar o projeto com a sua execução

Pelo próprio GitHub, você pode abrir diretamente o arquivo Jupiter:
[Note-Jupiter: Analise Exploratoria de Dados](https://github.com/leonardo78rs/Python_AnaliseExploratoriaDados/blob/main/ds-expl-analis-dados-completo.ipynb)


# Conteúdo do projeto

### Início: Importar um arquivo e entender que tipo de informações temos
import pandas as pd
notas = pd.read_csv("https://raw.githubusercontent.com/alura-cursos/data-science-analise-exploratoria/main/Aula_0/ml-latest-small/ratings.csv")
notas
notas.shape

notas.columns = ["usuarioId", "filmeId", "nota", "momento"]
notas.head()

notas["nota"].unique()

notas["nota"].value_counts()

notas["nota"].mean()

notas["nota"].plot(kind='hist')

notas["nota"].median()

mediana = notas["nota"].median()
media = notas["nota"].mean()
print(f"Mediana é {mediana}")
print(f"Média é {media}")

notas["nota"].describe()

import seaborn as sns

sns.boxplot(notas["nota"])


### Explorando os filmes

filmes = pd.read_csv("https://raw.githubusercontent.com/alura-cursos/data-science-analise-exploratoria/main/Aula_0/ml-latest-small/movies.csv")
filmes.columns = ["filmeId", "titulo", "generos"]
filmes.head()

notas.head()

notas.query("filmeId==1")["nota"].mean()

notas.query("filmeId==2")["nota"].mean()

medias_por_filme = notas.groupby("filmeId")["nota"].mean()
medias_por_filme.head()

medias_por_filme.plot(kind="hist")

sns.boxplot(medias_por_filme)

medias_por_filme.describe()

import matplotlib.pyplot as plt

sns.displot(medias_por_filme, kde=True)
plt.title("Histograma das médias dos filmes")


### Outros filmes

tmdb = pd.read_csv("https://raw.githubusercontent.com/alura-cursos/data-science-analise-exploratoria/main/Aula_0/tmdb_5000_movies.csv")
tmdb.head()

sns.displot(tmdb["revenue"])
plt.title("Distribuição da receita dos filmes")
plt.show()

### prompt: gráfico de distribuicao do orçamento dos filmes (budget)

import matplotlib.pyplot as plt
sns.displot(tmdb["budget"])
plt.title("Distribuição do orçamento dos filmes")
plt.show()


tmdb.info()

tmdb.describe()

com_faturamento = tmdb.query("revenue > 0")
sns.displot(com_faturamento["revenue"])

tmdb["original_language"].unique()

tmdb["original_language"].value_counts()

### lingua => categorica... sem ordem...

### budget (orcamento) => quantitativa continua

### nota do movielens => 0.5, 1, 1.5, ... ,5 => nao tem 2.5
### quantidade de votos => 1,2,3,4,5... nao existe 2.5

tmdb["original_language"].value_counts().index

tmdb["original_language"].value_counts().values

contagem_de_lingua = tmdb["original_language"].value_counts().to_frame().reset_index()
contagem_de_lingua.columns = ["original_language", "total"]
contagem_de_lingua.head()

sns.barplot(data = contagem_de_lingua, x="original_language", y="total")

sns.countplot(data=tmdb, x="original_language")

### gráfico de pizza não é adequado neste caso

contagem_de_lingua.plot(kind="pie", y="total", labels=contagem_de_lingua["original_language"])

### Melhorando nossa visualização

total_por_lingua = tmdb["original_language"].value_counts()
total_geral = total_por_lingua.sum()
total_de_ingles = total_por_lingua.loc["en"]
total_do_resto = total_geral - total_de_ingles
print(total_geral, total_de_ingles, total_do_resto)

dados = {
  "lingua" : ["ingles", "outros"],
  "total" : [total_de_ingles, total_do_resto]
}
dados = pd.DataFrame(dados)
dados

sns.barplot(data=dados, x="lingua", y="total")

dados.plot(kind="pie", y="total", labels=dados["lingua"])

total_de_outros_filmes_por_lingua = tmdb.query("original_language != 'en'")["original_language"].value_counts()
total_de_outros_filmes_por_lingua.head()

sns.countplot(data=tmdb.query("original_language != 'en'"),
              x="original_language")

sns.countplot(data=tmdb.query("original_language != 'en'"),
              order=total_de_outros_filmes_por_lingua.index,
              hue="original_language",
              x="original_language")

sns.color_palette("mako")

tmdb.query("original_language != 'en'")["original_language"].value_counts(normalize=True)


plt.figure(figsize=(16, 8))
sns.countplot(data=tmdb.query("original_language != 'en'"),
              order=total_de_outros_filmes_por_lingua.index,
              palette="mako",
              hue="original_language",
              hue_order=total_de_outros_filmes_por_lingua.index,
              stat="percent",
              x="original_language")
plt.title("Distribuição da língua original nos filmes exceto em inglês")
plt.show()

### Comparar duas distribuições visualmente ou através de medidas

filmes.head(2)

### prompt: extraia as notas dos dois filmes em variaveis distintas
notas_do_toy_story = notas.query("filmeId==1")["nota"]
notas_do_jumanji = notas.query("filmeId==2")["nota"]

media_do_toy_story = notas_do_toy_story.mean()
media_do_jumanji = notas_do_jumanji.mean()

print(media_do_toy_story, media_do_jumanji)

### prompt: mesmo mas com a mediana

mediana_do_toy_story = notas_do_toy_story.median()
mediana_do_jumanji = notas_do_jumanji.median()

print(mediana_do_toy_story, mediana_do_jumanji)


import numpy as np

filme1 = [2.5] * 10 + [3.5] * 10
filme2 = [5] * 10 + [1] * 10




media_filme1 = np.mean(filme1)
mediana_filme1 = np.median(filme1)

media_filme2 = np.mean(filme2)
mediana_filme2 = np.median(filme2)

print("Filme 1:")
print("Média:", media_filme1)
print("Mediana:", mediana_filme1)

print("\nFilme 2:")
print("Média:", media_filme2)
print("Mediana:", mediana_filme2)


plt.hist(filme1)
plt.hist(filme2)

plt.boxplot([filme1, filme2])

filme0 = [3.0] * 20 # espalha 0
np.mean(filme0)

np.std(filme0), np.std(filme1), np.std(filme2)


plt.boxplot([notas_do_toy_story, notas_do_jumanji])


sns.boxplot(data=notas.query("filmeId in [1,2,3,4,5]"),
            x="filmeId",
            y="nota")

### prompt: mesmo grafico mas colorido

sns.boxplot(data=notas.query("filmeId in [1,2,3,4,5]"),
            x="filmeId",
            y="nota",
            palette="Set2")


notas.groupby("filmeId").count()

notas["filmeId"].value_counts().tail()

notas.groupby("filmeId").count().query("nota == 1")









