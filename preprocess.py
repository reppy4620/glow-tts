import argparse
import warnings
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed

import torchaudio
torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False

warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')


def process_file(fn, resample, output_dir):
    wav, sr = torchaudio.load(fn)
    wav = resample(wav)
    torchaudio.save(str(output_dir / fn.name), wav, resample.new_freq)


def preprocess(dataset_dir, output_dir):
    out_dir = output_dir / 'wav'
    out_dir.mkdir(parents=True, exist_ok=True)

    fns = list(dataset_dir.glob('*.wav'))[:100]
    resample = torchaudio.transforms.Resample(48000, 22050)
    Parallel(n_jobs=-1)(
        delayed(process_file)(fn, resample, out_dir) for fn in tqdm(fns, total=len(fns))
    )


if __name__ == '__main__':
    # torchaudio.set_audio_backend('sox_io')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--wav_output_dir', type=str, required=True)
    # parser.add_argument('--data_text', type=str, required=True)
    # parser.add_argument('--text_output_dir', type=str, required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.wav_output_dir)

    preprocess(dataset_dir, output_dir)

    # output_dir = Path(args.text_output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)
    # with open(args.data_text, 'r') as f:
    #     lines = f.readlines()
    # train = lines[:4900]
    # valid = lines[4900:4990]
    # test = lines[4990:]
    # with open(output_dir / 'train.txt', 'w') as f:
    #     f.writelines(train)
    # with open(output_dir / 'valid.txt', 'w') as f:
    #     f.writelines(valid)
    # with open(output_dir / 'test.txt', 'w') as f:
    #     f.writelines(test)
