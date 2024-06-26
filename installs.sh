git clone https://github.com/facebookresearch/audiocraft.git
cd audiocraft
pip install -e .
pip install dora-search
pip install numba

sudo apt-get install build-essential libeigen3-dev libyaml-dev libfftw3-dev libtag1-dev libchromaprint-dev
pip install -U essentia-tensorflow

curl https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb --output genre_discogs400-discogs-effnet-1.pb
curl https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb --output discogs-effnet-bs64-1.pb
curl https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb --output mtg_jamendo_moodtheme-discogs-effnet-1.pb
curl https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb --output mtg_jamendo_instrument-discogs-effnet-1.pb