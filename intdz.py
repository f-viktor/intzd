import os
import sys
import tts
import bookparser
import argparse

# argparse, and starts run coroutine
def getArgs():
    parser = argparse.ArgumentParser(description='AppRTC')
    parser.add_argument('chapters', metavar='N', type=str, nargs='*',
                    help='text/html files to parse')
    parser.add_argument('--book', '-b', help='Folder of text files to process', default="")
    parser.add_argument('--output', '-o', help='folder to put the resulting audio file', default="audiobook")
    parser.add_argument('--format', '-f' ,help='Set the output audio format', default=".wav"), #doesnt work lol
    parser.add_argument('--verbose', '-v', action='count') # doesnt work either lol

    args = parser.parse_args()

    if len(args.chapters) > 0 and args.book:
        print("Don't use file lists and --book at the same time")

    if len(args.chapters) == 0 and args.book == "":
        print("Set at least one filename or --book")

    return args


# I hate this language
if __name__ == "__main__":
    args = getArgs()
    if args.book != "":
        chapterList = bookparser.getChapters(args.book)
    else:
        chapterList = args.chapters
    chapterList.sort()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    tacotron2, waveglow = tts.initializeTTSEngine()
    i = 0
    for chapter in chapterList:
        i=i+1
        chapterTitle= os.path.basename(os.path.splitext(chapter)[0])
        # check if we finished transcribing this before
        if not os.path.exists(args.output+"/"+chapterTitle+args.format):
            plainChapter = bookparser.parseChapter(args.book+chapter)
            subsentences = bookparser.splitSentences(plainChapter)
            tts.vocalise(subsentences, args.output+"/"+chapterTitle, args.format, tacotron2, waveglow)
            print("[+] TTS on "+chapter+" done, saved to "+args.output+"/"+chapterTitle+args.format+" ["+str(i)+"/"+str(len(chapterList))+"]")
        else:
            print("[+] "+chapter+" already exists at "+args.output+"/"+chapterTitle+args.format+", skipping! ["+str(i)+"/"+str(len(chapterList))+"]")
