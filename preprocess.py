import yaml
import argparse
import warnings
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from joblib import Parallel, delayed

import torchaudio

warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')


class PreProcessor:
    def preprocess(self):
        pass


class AudioProcessor(PreProcessor):
    def __init__(self, dataset_path, output_path):
        self.fns = list(Path(dataset_path).glob('*.wav'))
        self.output_dir = Path(output_path) / 'wav'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.resample = torchaudio.transforms.Resample(48000, 24000)

    def process_file(self, fn):
        wav, sr = torchaudio.load(fn)
        wav = self.resample(wav)
        torchaudio.save(str(self.output_dir / fn.name), wav, self.resample.new_freq, encoding='PCM_S', bits_per_sample=16)

    def preprocess(self):
        Parallel(n_jobs=-1)(
            delayed(self.process_file)(fn) for fn in tqdm(self.fns, total=len(self.fns))
        )


# class TextProcessor(PreProcessor):
#     def __init__(self, data_path, output_dir):
#         with open(data_path, 'r', encoding='utf-8') as f:
#             self.labels = yaml.safe_load(f)
#         self.output_dir = Path(output_dir)
#
#     def preprocess(self):
#         data = OrderedDict()
#         for k, v in tqdm(self.labels.items(), total=len(self.labels)):
#             phoneme = v['phone_level3']
#             data[f'DUMMY/{k}.wav'] = phoneme
#         assert len(data) == 5000
#         keys = list(data.keys())
#         train = [f'{k}|{data[k]}\n' for k in keys[:4900]]
#         valid = [f'{k}|{data[k]}\n' for k in keys[4900:4990]]
#         test = [f'{k}|{data[k]}\n' for k in keys[4990:]]
#         with open(self.output_dir / 'train.txt', 'w', encoding='utf-8') as f:
#             f.writelines(train)
#         with open(self.output_dir / 'valid.txt', 'w', encoding='utf-8') as f:
#             f.writelines(valid)
#         with open(self.output_dir / 'test.txt', 'w', encoding='utf-8') as f:
#             f.writelines(test)


if __name__ == '__main__':
    try:
        torchaudio.set_audio_backend('sox_io')
    except RuntimeError:
        torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
        torchaudio.set_audio_backend('soundfile')
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', type=str, required=True)
    parser.add_argument('--wav_output_dir', type=str, required=True)
    # parser.add_argument('--text_file_path', type=str, default='./filelists/basic5000.yaml')
    # parser.add_argument('--text_output_dir', type=str, default='./filelists')
    args = parser.parse_args()

    ap = AudioProcessor(args.wav_dir, args.wav_output_dir)
    # tp = TextProcessor(args.text_file_path, args.text_output_dir)

    print('Start audio preprocessing')
    ap.preprocess()
    # print('Start text preprocessing')
    # tp.preprocess()
