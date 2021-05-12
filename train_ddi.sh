# install for mecab
apt-get install -y swig mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file

# install modules for cleaning text
pip install unidecode mecab-python3==0.996.2 unidic jaconv zenhan pykakasi torchaudio

# install apex
git clone --recursive https://github.com/NVIDIA/apex.git
pip install apex

# preprocess wav files and text
if [ $# -ne 0 ]; then
  python preprocess.py --dataset_dir /content/drive/MyDrive/dataset/jsut/basic5000/wav \
                       --wav_output_dir /content/drive/MyDrive/dataset/jsut/processed \
                       --text_file_path ./filelists/basic5000.yaml \
                       --text_output_dir ./filelists
  ln -s /content/drive/MyDrive/dataset/jsut/processed DUMMY
fi

# build cython script
cd monotonic_align
python setup.py build_ext --inplace
cd ..

# run training script
python init.py -c ./configs/base.json -m base
python train.py -c ./configs/base.json -m base
