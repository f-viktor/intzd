import os

# gets the list of files in the folder, expects them to be some sort of html
def getChapters(inputFolder):
    chapterList = os.listdir(inputFolder)
    print("[+] Found "+str(len(chapterList))+" files in "+inputFolder)
    return chapterList

# converts .xhtml files to plain text
def parseChapter(chapter):
    with open(chapter) as html:
        from bs4 import BeautifulSoup
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
