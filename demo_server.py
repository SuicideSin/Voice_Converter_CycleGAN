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

from falcon_multipart.middleware import MultipartMiddleware
from model import CycleGAN
from preprocess import *
from wsgiref import simple_server

INDEX_HTML = '''
<!doctype html>
<html>
  <body>
    <h1>Voice Demo</h1>
  </body>
  <script src="./script/recorder.js" type="application/javascript"></script>
  <script type="application/javascript">

      var audio_context;
      var recorder;

      function startUserMedia(stream) {
	var input = audio_context.createMediaStreamSource(stream);
	window.recorder = new Recorder(input);
      }

      function createDownloadLink() {
	window.recorder && window.recorder.exportWAV(function(blob) {

          console.log('preparing send');
  	  var xhr=new XMLHttpRequest();
          xhr.responseType = 'blob';
	  xhr.onload=function(e) {
            console.log('response received');
            /*if (this.readyState === 4) {
	      console.log("Server returned: ",e.target.responseText);
	    }*/
            //var blob = new Blob([xhr.response], {type: 'audio/ogg'});
            //var objectUrl = URL.createObjectURL(blob);
	    var objectUrl = window.URL.createObjectURL(this.response);
            console.log('audio URL', objectUrl);
            var audio = document.getElementById('audio');
            audio.src = objectUrl;
            audio.play();
 	  };
          var fd = new FormData();
	  fd.append("audio_data", blob, "audio_file.wav");
          console.log(blob);
          console.log('sending...');
	  xhr.open("POST","/upload",true);
	  xhr.send(fd);
          console.log('sent');

	});
      }

      function startRecording(button) {
	window.recorder && window.recorder.record();
        document.getElementById('start').disabled = true;
        document.getElementById('stop').disabled = false;
      }

      function stopRecording(button) {
	window.recorder && window.recorder.stop();
        document.getElementById('start').disabled = false;
        document.getElementById('stop').disabled = true;

	// create WAV download link using audio data blob
	createDownloadLink();

	window.recorder.clear();
      }

     window.onload = function init() {
        //var getUserMedia;
	try {
	  // webkit shim
	  //window.AudioContext = window.AudioContext || window.webkitAudioContext;
	  //getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia;
	  //window.URL = window.URL || window.webkitURL;

	  audio_context = new AudioContext();
	} catch (e) {
	  alert('No web audio support in this browser!');
	}

        var media = navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        //var media = getUserMedia({ audio: true, video: false });
        window.media = media;
        console.log('media', media);
        media.then(startUserMedia);
        //navigator.mediaDevices.getUserMedia({ audio: true, video: false }).then(startUserMedia);
        //navigator.mediaDevices.getUserMedia({ audio: true, video: false }).then(startUserMedia);
      };
  </script>
  <button id="start" onclick="startRecording(this);">record</button>
  <br>
  <br>
  <button id="stop" onclick="stopRecording(this);" disabled>stop</button>
  <br>
  <br>
    <audio controls id="audio">
        <source id="source" src="" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
</html>
'''


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
        # TODO: Load wav
        #filepath = '/home/bt/dev/audio-samples/brandon_trump/wavs/test1.wav'
        #wav, _ = librosa.load(filepath, sr = self.sampling_rate, mono = True)

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

        # TODO: Return Wav
        librosa.output.write_wav('output_server_test.wav', wav_transformed, self.sampling_rate)

        #scipy.io.wavfile.write(buf, self.sampling_rate, (upsampled*maxv).astype(np.int16))

        #upsampled = librosa.resample(data, self.sampling_rate, 48000)

        #scipy.io.wavfile.write(buf, self.sampling_rate, wav_transformed.astype(np.int16))

        """
Data from the browser
---------------------
Input File     : 'input_server_test.wav'
Channels       : 2
Sample Rate    : 48000
Precision      : 54-bit
Duration       : 00:00:08.96 = 430080 samples ~ 672 CDDA sectors
File Size      : 6.88M
Bit Rate       : 6.14M
Sample Encoding: 64-bit Floating Point PCM

However we're returning
-----------------------
Input File     : 'output_server_test.wav'
Channels       : 1
Sample Rate    : 16000
Precision      : 25-bit
Duration       : 00:00:09.00 = 144000 samples ~ 675 CDDA sectors
File Size      : 576k
Bit Rate       : 512k
Sample Encoding: 32-bit Floating Point PCM
        """

        # https://bugzilla.mozilla.org/show_bug.cgi?id=523837

        #maxv = np.iinfo(np.float64).max
        upsampled = librosa.resample(wav_transformed, self.sampling_rate, 48000)
        #pcm_data = (upsampled*maxv).astype(np.float64)
        pcm_data = upsampled.astype(np.float64)
        print(pcm_data.shape)

        stereo_pcm_data = np.tile(pcm_data, (2,1)).T
        print(stereo_pcm_data.shape)

        #scipy.io.wavfile.write(buf, self.sampling_rate, wav_transformed)
        print('--- scipy wavfile ---')
        buf = io.BytesIO()
        scipy.io.wavfile.write(buf, 48000, stereo_pcm_data.astype(np.float32))
        scipy.io.wavfile.write('output_server_test_2.wav', 48000, stereo_pcm_data.astype(np.float32))
        print(buf)
        return buf


        #print('--- sound file ---')
        #sf = soundfile.SoundFile(buf)
        #print(sf)
        #buf2 = io.BytesIO()
        #buf2 = bytearray(len(stereo_pcm_data))
        #sf.buffer_read_into(buf2, dtype='int16')
        #new_buf = sf.buffer_read(dtype='float64')
        #print(new_buf)

        #buf2 = sf.buffer_read(dtype='float64')
        #print(buf2)
        #print(dir(buf2))
        #return buf2


model_dir_default = './model/sf1_tm1'
model_name_default = 'sf1_tm1.ckpt'
converter = Converter(model_dir_default, model_name_default)

class IndexHandler():
    def on_get(self, request, response):
        response.content_type = 'text/html'
        response.body = INDEX_HTML

class ApiHandler():
    def on_post(self, request, response):
        print("\n\n====Request=====\n\n")
        print(request)
        print(request.content_type)
        print(request.content_length)
        #print(request.params)
        #print(request.stream)
        #data, samplerate = soundfile.read(io.BytesIO(urlopen(url).read()))
        #data, samplerate = soundfile.read(request.stream)
        # NB: uses middleware to pull out data.
        form_data = request.params['audio_data'].file
        #print(form_data)
        data, samplerate = soundfile.read(form_data)
        scipy.io.wavfile.write('input_server_test.wav', samplerate, data)
        print(data)
        print(data.shape)
        print(samplerate)

	# Scale audio to the range of 16 bit PCM
	# https://stackoverflow.com/a/52757235
	#data /= 1.414 # Scale to [-1.0, 1.0]
	#data *= 32767 # Scale to int16
	#data = data.astype(np.int16)
	#scipy.io.wavfile.write(output_filename, sample_rate, audio)

	#>>> y, sr = librosa.load(librosa.util.example_audio_file(), sr=22050)
	#>>> y_8k = librosa.resample(y, sr, 8000)
	#>>> y.shape, y_8k.shape
	#((1355168,), (491671,))

        # NB: Convert the input stereo signal into mono.
        # In the future the frontend should be responsible for sampling details.
        #mono = [x[0] for x in data]
        mono = data[:, 0]

        downsampled = librosa.resample(mono, samplerate, 16000)

        buf = converter.convert(downsampled, conversion_direction = 'A2B')

        temp_dir = tempfile.TemporaryDirectory(prefix='tmp_ml_audio')
        f_input = tempfile.NamedTemporaryFile(suffix='.wav')
        f_output = tempfile.NamedTemporaryFile(suffix='.wav')

        f_input.write(buf.read())

        print(temp_dir.name)
        print(f_input.name)
        print(f_output.name)
        out_file = temp_dir.name + '/output.ogg'
        #out_file = './temp_output.ogg'
        print(out_file)

        #out = subprocess.check_output(['ffmpeg', '-i', f_input.name, './ffmpeg_output.wav'])
        #out = subprocess.check_output(['ffmpeg', '-i', f_input.name, out_file])
        out = subprocess.check_output(['ffmpeg', '-i', f_input.name, '-acodec', 'libvorbis', out_file])
        print(out)

        #response.content_type = 'audio/wav'
        response.content_type = 'audio/ogg'
        with open(out_file, mode='rb') as f:
            response.data = f.read()

        #response.data = buf.getvalue()
        #response.data = buf
        #response.body = buf[:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    api = falcon.API(middleware=[MultipartMiddleware()])
    api.add_route('/', IndexHandler())
    api.add_route('/upload', ApiHandler())
    api.add_static_route('/script', os.path.abspath('./script'))
    api.add_static_route('/sound', os.path.abspath('./sound'))
    print('Serving on 0.0.0.0:%d' % args.port)
    simple_server.make_server('0.0.0.0', args.port, api).serve_forever()

if __name__ == '__main__':
    main()

