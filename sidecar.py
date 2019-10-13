#!/usr/bin/env python3

"""
Starts a demo HTTP server to capture and transform audio
as a live demonstration of the trained model.

Brandon Thomas 2019-07-29 <bt@brand.io> <echelon@gmail.com>
"""

import argparse
import falcon
import io
import librosa
import numpy as np
import os
import scipy
import soundfile
import subprocess
import tempfile
import tensorflow as tf
import zmq

from falcon_multipart.middleware import MultipartMiddleware
from model import CycleGAN
from preprocess import *
from wsgiref import simple_server

print("TensorFlow version: {}".format(tf.version.VERSION))


class Converter():
    def __init__(self, model_dir, model_name):
        self.num_features = 24
        self.sampling_rate = 16000
        self.frame_period = 5.0

        self.model = CycleGAN(num_features = self.num_features, mode = 'test')

        self.model.load(filepath = os.path.join(model_dir, model_name))

        self.mcep_normalization_params = np.load(os.path.join(model_dir, 'mcep_normalization.npz'))
        self.mcep_mean_A = self.mcep_normalization_params['mean_A']
        self.mcep_std_A = self.mcep_normalization_params['std_A']
        self.mcep_mean_B = self.mcep_normalization_params['mean_B']
        self.mcep_std_B = self.mcep_normalization_params['std_B']

        self.logf0s_normalization_params = np.load(os.path.join(model_dir, 'logf0s_normalization.npz'))
        self.logf0s_mean_A = self.logf0s_normalization_params['mean_A']
        self.logf0s_std_A = self.logf0s_normalization_params['std_A']
        self.logf0s_mean_B = self.logf0s_normalization_params['mean_B']
        self.logf0s_std_B = self.logf0s_normalization_params['std_B']

    def convert(self, wav, conversion_direction='A2B'):
        wav = wav_padding(wav = wav, sr = self.sampling_rate, frame_period = self.frame_period, multiple = 4)
        f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = self.sampling_rate, frame_period = self.frame_period)
        coded_sp = world_encode_spectral_envelop(sp = sp, fs = self.sampling_rate, dim = self.num_features)
        coded_sp_transposed = coded_sp.T

        if conversion_direction == 'A2B':
            f0_converted = pitch_conversion(f0 = f0, mean_log_src = self.logf0s_mean_A, std_log_src = self.logf0s_std_A, mean_log_target = self.logf0s_mean_B, std_log_target = self.logf0s_std_B)
            coded_sp_norm = (coded_sp_transposed - self.mcep_mean_A) / self.mcep_std_A
            coded_sp_converted_norm = self.model.test(inputs = np.array([coded_sp_norm]), direction = conversion_direction)[0]
            coded_sp_converted = coded_sp_converted_norm * self.mcep_std_B + self.mcep_mean_B
        else:
            f0_converted = pitch_conversion(f0 = f0, mean_log_src = self.logf0s_mean_B, std_log_src = self.logf0s_std_B, mean_log_target = self.logf0s_mean_A, std_log_target = self.logf0s_std_A)
            coded_sp_norm = (coded_sp_transposed - self.mcep_mean_B) / self.mcep_std_B
            coded_sp_converted_norm = self.model.test(inputs = np.array([coded_sp_norm]), direction = conversion_direction)[0]
            coded_sp_converted = coded_sp_converted_norm * self.mcep_std_A + self.mcep_mean_A

        coded_sp_converted = coded_sp_converted.T
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = self.sampling_rate)
        wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = self.sampling_rate, frame_period = self.frame_period)

        # For debugging model output, uncomment the following line:
        # librosa.output.write_wav('model_output.wav', wav_transformed, self.sampling_rate)

        # TODO: Perhaps ditch this. It's probably unnecessary work.
        upsampled = librosa.resample(wav_transformed, self.sampling_rate, 48000)
        pcm_data = upsampled.astype(np.float64)
        stereo_pcm_data = np.tile(pcm_data, (2,1)).T

        buf = io.BytesIO()
        scipy.io.wavfile.write(buf, 48000, stereo_pcm_data.astype(np.float32))
        return buf

# Set up model
# This should live long in memory, so we do it up front.
model_dir_default = './model/sf1_tm1'
model_name_default = 'sf1_tm1.ckpt'
converter = Converter(model_dir_default, model_name_default)

class IndexHandler():
    def on_get(self, request, response):
        response.content_type = 'text/html'
        response.body = INDEX_HTML

class ApiHandler():
    def on_post(self, request, response):
        # NB: uses middleware to pull out data.
        form_data = request.params['audio_data'].file
        data, samplerate = soundfile.read(form_data)

        # For debugging browser input, uncomment the following line:
        # scipy.io.wavfile.write('browser_input_audio.wav', samplerate, data)

        # NB: Convert the input stereo signal into mono.
        # In the future the frontend should be responsible for sampling details.
        mono = data[:, 0]

        # NB: We must downsample to the rate that the network is trained on.
        downsampled = librosa.resample(mono, samplerate, 16000)

        # Evaluate the model
        print(">>> Converting...")
        results = converter.convert(downsampled, conversion_direction = 'A2B')

        temp_dir = tempfile.TemporaryDirectory(prefix='tmp_ml_audio')
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav')

        temp_file.write(results.read())

        out_file = temp_dir.name + '/output.ogg'

        # NB: Browsers have a great deal of trouble decoding WAV files unless they are in the
        # narrow slice of the WAV spec expected. None of the {librosa, scipy, soundfile} python
        # tools do a good job of this, so here we shell out to ffmpeg and generate OGG.
        # It's lazy and messy, but it works for now.
        # See https://github.com/librosa/librosa/issues/361 for a survey of the library landscape
        # See https://bugzilla.mozilla.org/show_bug.cgi?id=523837 for one of dozens of browser codec bugs
        _stdout = subprocess.check_output(['ffmpeg', '-i', temp_file.name, '-acodec', 'libvorbis', out_file])

        response.content_type = 'audio/ogg'
        with open(out_file, mode='rb') as f:
            response.data = f.read()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5555)
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:{}".format(args.port))
    print('Running server...')

    while True:
        #  Wait for next request from client
        message = socket.recv()
        print("Received request: %s" % message)

        # DO WORK

        #  Send reply back to client
        socket.send(b"World")

if __name__ == '__main__':
    main()

