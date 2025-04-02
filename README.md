<h1>search</h1>
<hr>

<h3>steps</h3>
<hr>
<pre>
1. install required pakages 
pip3 install -r requirements.txt
2. prepare mp3 dir
3. convert mp3 file to wav file 
python3 cvtmp3.py --mp3_dir='mp3_dir' ---wav_dir='wav_dir to save' 
4. fiter invalid samples
python3 filter_samples --samples_dir='wavfile_dir'
5. inference
python3 inference.py --wav_dir='wav_dir' 
</pre>
<pre>

python3 initialize.py --path='directory path that contains all the wav files'
python3 inference.py --path='query wav file path'
</pre>


