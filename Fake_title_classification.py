#!/usr/bin/env python
# coding: utf-8

# # Проверка заголовков на фейковость

# # Тестовое задание
# 
# Для выполнения тестового задания требуется разработать модель, которая будет способна различать заголовки реальных и выдуманных новостей.
# Для обучения модели используйте данные из файла `train.tsv`. 
# 
# В файле находится таблица, состоящая из двух колонок: 
# - В колонке title записан заголовок новости. 
# - В колонке is_fake содержатся метки: 0 – новость реальная; 1 – новость выдуманная.
# 
# Для демонстрации работы модели используйте данные тестового набора из файла `test.tsv`. В нем также есть колонка title, данные которой являются входными для вашей модели.
# Вам нужно скопировать файл `test.tsv`, переименовать его в `predictions.tsv` и заполнить колонку is_fake значениями предсказаний вашей модели, аналогично `train.tsv`. 
# Изначально колонка заполнена значением 0.
# 
# # Критерии оценки
# 1. Для оценки финального решения будет использоваться метрика F1 score.
# 2. Чистота кода, оформление и понятность исследования.
# 
# # Требования к решению
# В качестве решения мы ожидаем zip-архив со всеми *.py и *.ipynb файлами в папке solution и файлом `predictions.tsv` в корне. Формат имени zip-архива: LastName_FirstName.zip (пример Ivanov_Ivan.zip).
# 
# Файл `predictions.tsv` должен включать в себя колонку title, содержащую те же данные, что и исходный файл `test.tsv`, а также колонку is_fake, содержащую значения 0 или 1.
# Разметка тестового набора данных и включение его в обучение/валидацию запрещены.
# 
# В папке solution должно быть отражено исследование и весь код, необходимый для воспроизведения исследования.
# 
# Успехов!
# 

# ## Загрузка данных

# In[101]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')


# In[102]:


data = pd.read_csv('dataset/train.tsv', sep='\t')
data.head(10)


# In[103]:


data.info()
data.describe()


# In[104]:


data.isna().sum()


# In[105]:


display(all(data.duplicated()))


# In[106]:


data['is_fake'].value_counts()


# **Вывод:** данные хорошие, пустые значения и дубликаты отсутствуют, в колонке ***is_fake*** все значения корректны, баланс классов не нарушен.

# # Лемматизация текста

# In[107]:


def clear_text(text):
    text = text.lower()
    clear_text = re.sub(r'[^А-ЯЁа-яёA-Za-z ]', ' ', text) 
    clear_text = clear_text.split()
    return " ".join(clear_text)


# In[108]:


data['clear_title'] = data['title'].apply(lambda x: clear_text(x)) 
data = data.drop(['title'], axis=1)


# In[109]:


import pymorphy2
morph = pymorphy2.MorphAnalyzer()

def lemmatize(text):
    words = text.split() # разбиваем текст на слова
    res = list()
    for word in words:
        p = morph.parse(word)[0]
        res.append(p.normal_form)

    return " ".join(res)


# In[110]:


data['lemmatize_title'] = data['clear_title'].apply(lambda x: lemmatize(x))
data = data.drop(['clear_title'], axis=1)
data.head(5)


# **Вывод:** текст очищен и лемматизирован

# In[111]:


#обозначение признаков, деление на обучающую и валидационную выборки

features = data['lemmatize_title']
target = data['is_fake']

features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)


# # Векторизация

# In[112]:


nltk.download('stopwords')
stopwords = set(nltk_stopwords.words('russian'))


# In[113]:


tf_idf_model = TfidfVectorizer(stop_words=stopwords)
tf_idf_model.fit_transform(features_train)
res_valid = tf_idf_model.transform(features_valid)
res_train = tf_idf_model.transform(features_train)


# **Вывод:** данные векторизованы и приведены к нужному формату для дальнейшего использования в обучении.

# # Обучение

# ### LogisticRegression

# In[114]:


LR_model = LogisticRegression(random_state=12345) 
grid_values = {'C': [0.0001, 5, 10, 15, 100, 1000]}

grid_search = GridSearchCV(LR_model, param_grid=grid_values, scoring='f1', cv=5)
grid_search.fit(res_train, target_train)

print('F1: ', grid_search.best_score_)
print('Параметры : ', grid_search.best_params_)


# ### RandomForestClassifier

# In[115]:


RF_model = RandomForestClassifier(random_state=12345)

grid_values = { 'n_estimators': [40, 60, 80],
                'max_depth': [ 10, 20, 16]}

grid_search = GridSearchCV(RF_model, param_grid=grid_values, scoring= 'f1', cv = 5)
grid_search.fit(res_train, target_train)

print('F1: ', grid_search.best_score_)
print('Параметры : ', grid_search.best_params_)


# **Вывод:** исходя из показателей F1-меры, модель ***логистической регрессии*** показала наилучший результат на обучающих данных и будет проверена на валидационой выборке.

# ## Проверка на валидационной выборке

# In[116]:


model_LR = LogisticRegression(C=5, random_state=12345)
model_LR.fit(res_train, target_train)
predictions = model_LR.predict(res_valid)
print('F1 на валидационной выборке у LogisticRegression: {:.2f}'.format(f1_score(target_valid, predictions)))


# ## Демонстрация работы модели на тестовых данных 

# In[125]:


test = pd.read_csv('dataset/test.tsv', sep='\t')
test.head(5)


# In[126]:


features_test = test['title']


# In[127]:


tf_idf_model = TfidfVectorizer(stop_words=stopwords)
tf_idf_model.fit_transform(features_train)
res_test = tf_idf_model.transform(features_test)


# In[128]:


predictions_test = model_LR.predict(res_test)


# In[129]:


test['is_fake'] = predictions_test
test.head(20)


# In[130]:


test.to_csv('predictions.tsv', index=True)


# ## Вывод
# В ходе работы было выполнено следующее:
# 
# - Изучены и подготовлены данные.
# - Выполнены лемматизация и векторизация текста.
# - Данные поделены на обучающую и валидационную выборки.
# - Выбраны и обучены модели для задачи классификации.
# - Работа лучшей модели LogisticRegression проверена на валидационной выборке.
# - Сделаны предсказания для тестовой выборки.

# In[ ]:




