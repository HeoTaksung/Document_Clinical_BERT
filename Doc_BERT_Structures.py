import os
from transformers import *
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])

class Doc_Bert_LSTM(tf.keras.Model):
	def __init__(self, model_name, dir_path, num_class):
		self.num_class = num_class
		self.bert = TFBertModel.from_pretrained(model_name, cache_dir=dir_path, from_pt=True)
		self.dropout1 = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
		self.lstm = tf.keras.layers.LSTM(self.bert.config.hidden_size)
		self.dropout2 = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
		self.classifier = tf.keras.layers.Dense(self.num_class, acitvation='sigmoid',
							kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range))

	def call(self, inputs):  # inputs shape = (inputs_ids, attention_mask, token_type)
		ids = inputs[0]  # (the number of sentences, batch_size, sentence length)
		mask = inputs[1]  # (the number of sentences, batch_size, sentence length)
		token = inputs[2]  # (the number of sentences, batch_size, sentence length)

		cls_tokens = []

		for i in range(ids.shape[0]):
			cls_tokens.append(self.dropout1(self.bert(ids[i], mask[i], token[i])))

		cls_tokens = tf.stack(cls_tokens, axis=1)  # (the number of sentence, hidden_size)

		output = self.lstm(cls_tokens)  # (hidden_size)

		output = self.dropout2(output)

		output = self.classifier(output)

		return output


class Doc_Bert_MEAN(tf.keras.Model):
	def __init__(self, model_name, dir_path, num_class):index
		self.num_class = num_class
		self.bert = TFBertModel.from_pretrained(model_name, cache_dir=dir_path, from_pt=True)
		self.dropout1 = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
		self.dropout2 = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
		self.classifier = tf.keras.layers.Dense(self.num_class, acitvation='sigmoid',
							kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range))

	def call(self, inputs):  # inputs shape = (inputs_ids, attention_mask, token_type)
		ids = inputs[0]  # (the number of sentences, batch_size, sentence length)
		mask = inputs[1]  # (the number of sentences, batch_size, sentence length)
		token = inputs[2]  # (the number of sentences, batch_size, sentence length)

		cls_tokens = []

		for i in range(ids.shape[0]):
			cls_tokens.append(self.dropout1(self.bert(ids[i], mask[i], token[i])))

		cls_tokens = tf.stack(cls_tokens, axis=1)  # (the number of sentence, hidden_size)

		output = tf.keras.layers.GlobalAveragePooling1D()(cls_tokens) # (hidden_size)

		output = self.dropout2(output)

		output = self.classifier(output)

		return output


# Model Execute

model_name = "emilyalsentzer/Bio_ClinicalBERT"

with strategy.scope():
	model = Doc_Bert_LSTM(model_name=model_name, dir_path='bert_ckpt', num_class=num_class)
	# model = Doc_Bert_MEAN(model_name=model_name, dir_path='bert_ckpt', num_class=num_class)

	F1_macro = tfa.metrics.F1Score(num_classes=num_class, average='macro',threshold=0.5, name='f1_macro')
	F1_micro = tfa.metrics.F1Score(num_classes=num_class, average='micro',threshold=0.5, name='f1_micro')

	optimizer = tf.keras.optimizers.Adam(3e-5)
	loss = tf.keras.losses.BinaryCrossentropy()
	metric = tf.keras.metrics.BinaryAccuracy()

	model.compile(optimizer=optimizer, loss=loss, metrics=[metric, F1_macro, F1_micro])

earlystop_callback = EarlyStopping(monitor='val_f1_micro', verbose=1, min_delta=0.0001, patience=5, mode='max', restore_best_weights=True)

checkpoint_path = os.path.join('./', '', 'weights.h5')
checkpoint_dir = os.path.dirname(checkpoint_path)

if os.path.exists(checkpoint_dir):
	print("{} -- Folder already exists \n".format(checkpoint_dir))
else:
	os.makedirs(checkpoint_dir, exist_ok=True)
	print("{} -- Folder create complete \n".format(checkpoint_dir))

cp_callback = ModelCheckpoint(checkpoint_path, monitor='val_f1_macro', mode='max', verbose=1, save_best_only=True, save_weights_only=True)

model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,
		  validation_data=(dev_X, dev_y), callbacks=[earlystop_callback, cp_callback])

test_loss, test_acc, test_macro, test_micro = model.evaluate(test_X, test_y, batch_size=batch_size)

print("TEST Loss : {:.6f}".format(test_loss))
print("TEST ACC : {:.6f}".format(test_acc))
print("TEST F1-macro : {:.6f}".format(test_macro))
print("TEST F1-micro : {:.6f}".format(test_micro))
