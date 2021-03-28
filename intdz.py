import os
import sys

# get the input agruments
def getArgs():
    if  len(sys.argv) < 3:
        print("python3 main.py chaptersfolder outputfolder")
        sys.exit(2)
    inputFolder = sys.argv[1]
    outputFolder = sys.argv[2]
    return inputFolder, outputFolder

# gets the list of files in the folder, expects them to be some sort of html
def getChapters(inputFolder):
    chapterList = os.listdir(inputFolder)
    print("[+] Found "+str(len(chapterList))+" files in "+inputFolder)
    return chapterList

from bs4 import BeautifulSoup
# converts .xhtml files to plain text
def parseChapter(chapter):
    with open(chapter) as html:
        soup = BeautifulSoup(html, features="html.parser")

        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()    # rip it out

        # get text
        plainChapter = soup.get_text()
        print("[+] Converted "+chapter+" into plaint text")
        return plainChapter
    print("[!] Failed to convert "+chapter+" into plaint text")
    return

#split text to small parts as tacotron2 doesn't like long sentences and will clip them
def splitSentences(fulltext):
    #for splitting text by sentence
    import nltk.data
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences =tokenizer.tokenize(fulltext)

    #addittionally splitting at every comma
    subsentences = []
    for i in sentences:
        subsentences.extend(i.split(", "))
    return subsentences

# just stuff i found here https://pytorch.org/hub/nvidia_deeplearningexamples_tacotron2/
def vocalise(subsentences, outputFile):
    print("[+] Starting TTS on "+outputFile.split("/")[-1])
    import torch
    tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')

    import numpy as np
    from scipy.io.wavfile import write

    tacotron2 = tacotron2.to('cuda')
    # modified this to allow longer sentences, unsure if it did anything at all.
    tacotron2.max_decoder_steps = 3000
    tacotron2.eval()

    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to('cuda')
    waveglow.eval()


    try:# this is here so we can save before Ctrl+C
        # process each sentence as tacotron2 -> waveglow -> whatever
        rate = 22050
        audio_numpy = np.ndarray(1)
        for text in subsentences:
            print(text)
            # preprocessing
            sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
            sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)

            # run the models
            with torch.no_grad():
                _, mel, _, _ = tacotron2.infer(sequence)
                audio = waveglow.infer(mel)
            #audio_numpy = audio[0].data.cpu().numpy()
            audio_numpy = np.concatenate((audio_numpy, audio[0].data.cpu().numpy()))


        # save resulting wav file
        write(outputFile+".wav", rate, audio_numpy)
    except KeyboardInterrupt:
        write(outputFile+".wav.part", rate, audio_numpy)
        # this means there was a partial TTS, so we also save where we left off
        print("\n[!] TTS interrupted, partial file written to "+ outputFile+".wav.part \nprogress saved to "+outputFile+".sav")
        print("[?] To continue from here run  python intdz.py something something")
        sys.exit()

# I hate this language
if __name__ == "__main__":
    inputFolder, outputFolder = getArgs()
    chapterList = getChapters(inputFolder)
    chapterList.sort()
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    i = 0
    for chapter in chapterList:
        plainChapter = parseChapter(inputFolder+"/"+chapter)
        subsentences = splitSentences(plainChapter)
        vocalise(subsentences,outputFolder+"/"+os.path.splitext(chapter)[0])
        i=i+1
        print("[+] TTS on "+chapter+" done, saved to "+outputFolder+"/"+os.path.splitext(chapter)[0]+".wav ["+str(i)+"/"+str(len(chapterList))+"]")
