"""
Deep Learning Platform
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

from __future__ import print_function, division

from math import ceil

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization

from Managers.common_functions import create_decoder
from Trainers import Trainer


class LSTMTrainer(Trainer):
    def build(self, preparator, epoch_counter):
        super().build(preparator, epoch_counter)
        activity_decoder = create_decoder(preparator.orchestrator.encoder_descriptions[1], True)
        self.decoders.append(activity_decoder)
        ts_decoder = create_decoder(preparator.orchestrator.encoder_descriptions[2], True)
        self.decoders.append(ts_decoder)

    def get_prediction(self, input, leftovers):
        output = self.model.predict(input)
        result = []
        for i in range(len(self.decoders)):
            decoder = self.decoders[i]
            if decoder.leftover_name:
                leftover = leftovers[decoder.leftover_name]
            else:
                leftover = None
            result.append(decoder.encode_single_result(input, output[i], leftover))
        return result

    def core(self):

        # build the model:
        print('Build model...')
        main_input = Input(
            shape=(self.preparator.orchestrator.max_case_length, self.preparator.orchestrator.features_counter),
            name='main_input')
        # train a 2-layer LSTM with one shared layer
        l1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(
            main_input)  # the shared layer
        b1 = BatchNormalization()(l1)
        l2_1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(
            b1)  # the layer specialized in activity prediction
        b2_1 = BatchNormalization()(l2_1)
        act_output = Dense(self.preparator.orchestrator.activity_counter, activation='softmax',
                           kernel_initializer='glorot_uniform', name='act_output')(
            b2_1)

        model = Model(inputs=[main_input], outputs=[act_output])

        opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

        model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=opt)

        early_stopping = EarlyStopping(monitor='loss', patience=42)
        model_checkpoint = ModelCheckpoint('Output/'+self.preparator.orchestrator.output_name+'/Trainers/model_{epoch:02d}-{loss:.2f}.h5', monitor='loss',
                                           verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0)
        return model, [early_stopping, model_checkpoint, lr_reducer]

    def train_model_online(self):
        model, callbacks = self.core()
        epoch_size = self.preparator.get_epoch_size_online()
        model.fit(self.preparator.run_online(self.epoch_counter), verbose=1, callbacks=callbacks,
                  steps_per_epoch=ceil(epoch_size / self.preparator.output_chunk_size), epochs=self.epoch_counter)
        self.model = model
        return model

    def train_model_offline(self):
        epoch_size = self.preparator.get_epoch_size_offline()
        model, callbacks = self.core()
        model.fit(self.preparator.read_offline(epoch_size), verbose=1,
                  callbacks=callbacks, steps_per_epoch=ceil(epoch_size / self.preparator.output_chunk_size),
                  epochs=self.epoch_counter)
        self.model = model
        return model
