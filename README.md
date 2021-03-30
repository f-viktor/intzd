# INTDZ aka. "It's Not That Deep, Zen"
Aka. I'm too cheap to pay $8/mo for good quality audiobooks.  
Using tacotron2 -> waveglow to convert .epub into audiobooks.

## Disclaimer
I used free e-books from the wonderful standardebooks.org, but any e-book should work as long as it's in .epub format.  
Converting the e-books into audiobooks takes "a while"™. With a GTX1060 it takes about as long as the resulting audiobook is.  
The voice, while legible, is inferior to an actual person reading the book.
I've hopefully uploaded some [sample audio](https://github.com/f-viktor/intzd/blob/master/sampleAudio.ogg?raw=true) so you can see what you're getting without having to wait 30 minutes to convert a short chapter.  
There are superior voice synthesizers out there, but none that you can run for free on your own machine.
If you find one, let me know. Maybe Glow-TTS would be better for this, but I haven't found a good writeup for it.  
I have no idea about the legality of uploading such audiobooks on the internet or even creating them.
Best if you don't do this with anything that is actively copyrighted. I take no responsibility for how you use this.  

## How to
1. We are using CUDA /w TensorFlow here, so be sure to have at least a GPU driver installed and a CUDA capable GPU.  
2. python=python3 If you are still running a peasant tier OS that aliases python to python2, then you are the architect of your own misery  
3. create a python venv so this filthy script language doesn't contaminate your pure golang distrou~
```
python -m venv tts
source tts/bin/activate
```
4. `pip install -r requirements.txt` to be honest i don't remember installing this much stuff, i guess they are mostly dependencies
5. Unzip your e-book (yes, .epub files are .zip archives), and locate the folder with the text in it.  
it will likely be called `bookname/epub/text` or something similar. hopefully you will find separate .xhtml files for each   chapter here, and nothing else.
6. Remove everything from the folder that you do not want to convert to audio (table of contents, afterword, etc.)
7. You can now start the voice synthesis as `python intdz.py -b bookname/epub/text -o bookname/audio`  
This will go file-by-file in the `/text` folder and when the audio file is done, it will save to `/audio`
8. The PyTorch module will cry about something, just copy whatever it suggests in red and it should work.

## Usage
```
usage: intdz.py [-h] [--book BOOK] [--output OUTPUT] [--format FORMAT] [--audio_speed AUDIO_SPEED] [--verbose] [N ...]

Convert .epub into audiobooks via AI Text-to-Speech

positional arguments:
  N                     text/html files to parse

optional arguments:
  -h, --help            show this help message and exit
  --book BOOK, -b BOOK  Folder of text files to process
  --output OUTPUT, -o OUTPUT
                        folder to put the resulting audio file
  --format FORMAT, -f FORMAT
                        Set the output audio format
  --audio_speed AUDIO_SPEED, -as AUDIO_SPEED
                        Set the speed of the resulting audio
  --verbose, -v

```

### Example usecases
(unzip your .epub book into a folder beforehand)  

Convert an entire book (results will be in ./audiobook)
```
python intdz.py -b unzippedbook/epub/text/
```
Convert only the first two chapters
```
python intdz.py unzippedbook/epub/text/chapter-1.xhtml  unzippedbook/epub/text/chapter-2.xhtml
```
Convert the entire book and format as mp3
```
python intdz.py -b unzippedbook/epub/text/ -f .mp3
```
Convert the entire book and output it to a custom folder
```
python intdz.py -b unzippedbook/epub/text/ -o custom/folder/path
```
Convert a book, with each file playing back at 0.5x speed
```
python intdz.py -b unzippedbook/epub/text/ -as 0.5
```


## How long does it take
Testing on [The Book of Tea](https://standardebooks.org/ebooks/okakura-kakuzo/the-book-of-tea), the resulting audio is 107 minutes (1:47) whereas the suggested reading time is 66 minutes. This means the audiobook is 60% longer than the suggested reading time displayed on standardebooks.org. This isn't too bad as audiobooks are generally slower than reading the actual book to begin with. This is fairly normal for any audiobook. Generating the audiofiles also took around 90 minutes on a 1060. That being said, listening at this speed is quite fatiguing, as the AI rarely makes pauses, and occasionally pronounces things weird. I found that setting playback speed to 0.9 makes for a better listening experinece, and this is now the default. You can control this value via the --audio_speed option.

## Improvement ideas
~~Save last transcribed sentence and corresponding audio to a file when script is interrupted.~~  
~~Option to only transcribe a file, or a specific subset of files.~~  
~~Recognize which chapters have already been transcribed, and continue from there.~~  
~~compress savefiles~~   
Convert to a different file format directly, without writing a wav.   
Re-try tacotron input if we run out of decoder steps, by splitting the sentence into parts  
Add an option to play the audio real-time while transcribing.  

## FAQ
What is `Warning! Reached max decoder steps`?  
- This is tachotron's error message meaning that it failed to create a spectogram from the subsentence.
This usually happens if the sentence is too long, or is weirdly punctuated.
Most commonly, it will result in the audio being cut off, or having some weird robot-voice anomaly.
If you want to avoid this, you can manually splice sentences with commas, but it should happen relatively rarely.
(like once or twice per chapter)

## Sources
https://pytorch.org/hub/nvidia_deeplearningexamples_tacotron2/
