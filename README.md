# INTDZ aka. "It's Not That Deep, Zen"
Aka. I'm too cheap to pay $8/mo for good quality audiobooks.  
Using tacotron2 -> waveglow to convert .epub into audiobooks.

## Disclaimer
I used free e-books from the wonderful standardebooks.org, but any e-book should work as long as it's in .epub format.  
Converting the e-books into audiobooks takes "a while"™. With a GTX1060 it takes about as long as the resulting audiobook is.  
The voice, while legible, is inferior to an actual person reading the book.
I've hopefully uploaded some sample audio so you can see what you're getting without having to wait 30 minutes to convert a short chapter.  
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
6. You can now start the voice synthesis as `python intdz.py bookname/epub/text bookname/audio`  
This will go file-by-file in the `/text` folder and when the audio file is done, it will save to `/audio`
7. The PyTorch module will cry about something, just copy whatever it suggests in red and it should work.


## Improvement ideas
~~Save last transcribed sentence and corresponding audio to a file when script is interrupted.~~  
~~Option to only transcribe a file, or a specific subset of files.~~  
~~Recognize which chapters have already been transcribed, and continue from there.~~  
compress savefiles https://medium.com/@busybus/zipjson-3ed15f8ea85d   
Add an option to play the audio real-time while transcribing.  
Convert to a different file format directly, without writing a wav.   
Re-try tacotron input if we run out of decoder steps, by splitting the sentence into parts  

## FAQ
What is `Warning! Reached max decoder steps`?  
- This is tachotron's error message meaning that it failed to create a spectogram from the subsentence.
This usually happens if the sentence is too long, or is weirdly punctuated.
Most commonly, it will result in the audio being cut off, or having some weird robot-voice anomaly.
If you want to avoid this, you can manually splice sentences with commas, but it should happen relatively rarely.
(like once or twice per chapter)

## Sources
https://pytorch.org/hub/nvidia_deeplearningexamples_tacotron2/
