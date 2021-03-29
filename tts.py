import sys
import numpy as np # this goes into the vocaliser file
import torch  # sane
from scipy.io.wavfile import write #same
#import soundfile as sf # ogg codec crashes on long files

# set up tachotron and waveglow from torchub
def initializeTTSEngine():
    tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
    # modified this to allow longer sentences, unsure if it did anything at all.
    tacotron2.max_decoder_steps = 3000
    tacotron2 = tacotron2.to('cuda')
    tacotron2.eval()

    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to('cuda')
    waveglow.eval()
    return tacotron2, waveglow

# just stuff i found here https://pytorch.org/hub/nvidia_deeplearningexamples_tacotron2/
# This does the actual text to speech
def vocalise(subsentences, outputFile, outputFormat, tacotron2, waveglow):
    print("[+] Starting TTS on "+outputFile.split("/")[-1])
    rate = 22050
    # load a saved state if any
    audio_numpy, currentSubsentenceIndex = loadVocaliseState(outputFile)
    try:# this is here so we can save before Ctrl+C

        # process each sentence as tacotron2 -> waveglow -> whatever
        from tqdm import tqdm
        for text in tqdm(subsentences[currentSubsentenceIndex:],
                            dynamic_ncols=True,
                            initial=currentSubsentenceIndex,
                            total=len(subsentences)):

            #print(text)
            # preprocessing
            sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
            sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)

            # run the models
            with torch.no_grad():
                _, mel, _, _ = tacotron2.infer(sequence)
                audio = waveglow.infer(mel)
            audio_numpy = np.concatenate((audio_numpy, audio[0].data.cpu().numpy()))
            # save current position in case TTS is interrupted
            currentSubsentenceIndex = currentSubsentenceIndex + 1

        # save resulting wav file
        write(outputFile+outputFormat, rate, audio_numpy)
        #sf.write(outputFile+".wav",audio_numpy, rate, format="ogg")
        #writeAudio(outputFile+".wav",audio_numpy, rate)

        cleanSaves(outputFile)
    except KeyboardInterrupt:
        saveVocaliseState(outputFile,outputFormat,audio_numpy,currentSubsentenceIndex,rate)
        sys.exit()



import io
import json, zlib, base64
import os
def saveVocaliseState(outputFile, outputFormat, audio_numpy, currentSubsentenceIndex,rate):


    # saving binary array
    memfile = io.BytesIO()
    np.save(memfile, audio_numpy)
    memfile.seek(0)
    saveDict = {
        "subsentenceIndex" : currentSubsentenceIndex,
        "binaryAudio_numpy" : base64.b64encode(zlib.compress(memfile.read())).decode('ascii')
        # latin-1 maps byte n to unicode code point n
    }

    with open(outputFile+'.sav', 'w') as f:
        json.dump(saveDict, f)

    # save partial audio file
    # sf.write(outputFile+".ogg.part",audio_numpy, rate, format="ogg")
    #writeAudio(outputFile+".mp3.part",audio_numpy, rate)
    write(outputFile+outputFormat+".part", rate, audio_numpy)

    # this means there was a partial TTS, so we also save where we left off
    print("\n[!] TTS interrupted, partial file written to "+ outputFile+outputFormat+".part progress saved to "+outputFile+".sav")
    print("[?] To continue from here, set the --output to the folder containing the .sav file")
    return

def loadVocaliseState(loadPath):
    loadPath=loadPath+".sav"
    if os.path.exists(loadPath):
        with open(loadPath, 'r') as f:
            saveDict = json.load(f)
            memfile = io.BytesIO()
            memfile.write(zlib.decompress(base64.b64decode(saveDict["binaryAudio_numpy"])))
            memfile.seek(0)
            audio_numpy = np.load(memfile)
            print("[+] Found saved state at "+loadPath+" continuing from subsentence " + str(saveDict["subsentenceIndex"]))
            return audio_numpy, saveDict["subsentenceIndex"]
    else:
        print("[+] Saved state not found at "+loadPath+" starting from the beginning")
        audio_numpy = np.ndarray(1)
        currentSubsentenceIndex = 0
        return audio_numpy, currentSubsentenceIndex

def cleanSaves(outputFile):
    if os.path.exists(outputFile+".sav"):
        os.remove(outputFile+".sav")
    if os.path.exists(outputFile+outputFormat+".part"):
        os.remove(outputFile+outputFormat+".part")
    print("[+] Partial files "+outputFile+".sav, "+outputFile+outputFormat+".part and "+outputFile+outputFormat+" removed")
