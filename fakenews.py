import pandas as pd
import numpy as np
import seaborn as sns
import string
import contractions
import re
import nltk
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from matplotlib import rcParams
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten


sns.set_theme(style="darkgrid")
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [10,10]

def preprocess_text(x):
	cleaned_text = re.sub(r'[^a-zA-Z\d\s\']+', '', x)
	word_list = []
	for each_word in cleaned_text.split(' '):
		try:
			word_list.append(contractions.fix(each_word).lower())
		except:
			print(x)

	return " ".join(word_list)

fake_df = pd.read_csv('Fake.csv', header=0)
fake_df['label'] = 1
#print(fake_df.shape)

true_df = pd.read_csv('True.csv', header=0)
true_df['label'] = 0
#print(true_df.shape)
#print(fake_df.isna().sum())
#print(true_df.isna().sum())



text_cols = ['title', 'text']
for col in text_cols:
	print("Preprocessing column: {}".format(col))
	fake_df[col] = fake_df[col].apply(lambda x: preprocess_text(x))
	true_df[col] = true_df[col].apply(lambda x: preprocess_text(x))

for col in text_cols:
	print("Processing column: {}".format(col))
	fake_df[col] = fake_df[col].apply(word_tokenize)
	true_df[col] = true_df[col].apply(word_tokenize)

sw = stopwords.words('english')
for col in text_cols:
	print("Processing column: {}".format(col))
	fake_df[col] = fake_df[col].apply(lambda x: [each_word for each_word in x if each_word not in sw])
	true_df[col] = true_df[col].apply(lambda x: [each_word for each_word in x if each_word not in sw])

train_df = fake_df.append(true_df)
#print(train_df.isna().sum())
#print(train_df.shape)

train_df['all_info'] = train_df['text'] + train_df['title']

tokenizer = Tokenizer(oov_token = "<OOV>", num_words=6000)
tokenizer.fit_on_texts(train_df['all_info'])

target = train_df['label'].values

max_length = 40
vocab_size = 6000

sequences_train = tokenizer.texts_to_sequences(train_df['all_info'])

padded_train = pad_sequences(sequences_train, padding = 'post', maxlen=max_length)

X_train, X_test, y_train, y_test = train_test_split(padded_train, target, test_size=0.2)

train_df['text_joined'] = train_df['text'].apply(lambda x: " ".join(x))

# join all texts in resective labels
all_texts_gen = " ".join(train_df[train_df['label']==0]['text_joined'])
all_texts_fake = " ".join(train_df[train_df['label']==1]['text_joined'])

# Wordcloud for Genuine News
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords.words('english'),
                min_font_size = 10).generate(all_texts_gen)                       
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

# Worldcloud for Fake News
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords.words('english'),
                min_font_size = 10).generate(all_texts_fake)                    
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
  
plt.show()

embeddings_index = dict()

f = open('glove.6B.300d.txt', encoding='utf-8')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded {} word vectors.'.format(len(embeddings_index)))

print('Get vocab_size')
vocab_size = len(tokenizer.word_index) + 1

print('Create the embedding matrix')
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

best_model_file_name = "best_model_simple_with_GloVe.hdf5"

def get_simple_GloVe_model():
	model = Sequential()
	model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_length, trainable=False))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	return model

callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, verbose=1, mode="min", restore_best_weights=True),
	keras.callbacks.ModelCheckpoint(filepath=best_model_file_name, verbose=1)]

model = get_simple_GloVe_model()
print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=callbacks)

metric_to_plot = "loss"
plt.plot(range(1, max(history.epoch) + 2), history.history[metric_to_plot], ".:", label="Training loss")
plt.plot(range(1, max(history.epoch) + 2), history.history["val_" + metric_to_plot], ".:", label="Validation loss")
plt.title('Training and Validation Loss')
plt.xlim([1,max(history.epoch) + 2])
plt.xticks(range(1, max(history.epoch) + 2))
plt.legend()
plt.show()


model = keras.models.load_model(best_model_file_name)
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F1 Score: ', f1_score(y_test, y_pred))