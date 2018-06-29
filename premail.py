import os
import io
import re
import numpy as np
from stemming.porter2 import stem

Tags = ["Date:","From:","Message-ID:","Content-Transfer-Encoding:","X-Beenthere:","URL:"]
Headers = ["wrote:"]
Symbols = ["----","____"]
ContentRegexReplacer = {re.compile('</?\w+[^>]*>'):' __html_tag '}
ContentSymbols = ["\n", ",", ".", ";", "!", "?","(", ")", "{", "}",
                  "[", "]", "+", "'", '"', "-", "*", "=", "#", "/",
                  "\\", "$" , ">", "'s"]
WordReplacer = {"http:": "__http_address",
                   "https:": "__http_address",
                   "@": "__email_address"}
WordSymbols = [":"]
WordRegexReplacer = {re.compile("[0-9]+:[0-9]+(:[0-9]+)*"):"__time",
                     re.compile("[0-9]+/[0-9]+(/[0-9]+)*"):"__date",
                     re.compile("[0-9-+_.]+"): "__number",
                     re.compile("[0-9-+.]+%"): "__percentage"}

def convert_file(old_filepath, new_filepath):
        old_file = io.open(old_filepath, 'r', encoding='latin-1')
        try:
            lines = old_file.readlines()
        except:
            print(old_filepath)
            return
        new_file = open(new_filepath, 'w', encoding='utf-8')

        line_index = 0
        line = lines[line_index]
        line_index += 1
        # It seems the content will be separated by a blank line
        while line!='\n':
            # Still need to take care of the line, because sometimes it is not the content
            # And have to get rid of some symbols
            line = lines[line_index]
            line_index += 1
        line_index -= 1
        while line_index < len(lines):
            line = lines[line_index]
            line_index += 1
            is_meaningless_line = False
            for t in Tags:
                if line.__contains__(t):
                    is_meaningless_line = True
                    break
            for h in Headers:
                if line.__contains__(h):
                    is_meaningless_line = True
                    break
            for s in Symbols:
                if line.__contains__(s):
                    is_meaningless_line = True
                    break
            if is_meaningless_line:
                continue
            # get rid of html tags like <html> </script> ...
            for l in ContentRegexReplacer:
                line = l.sub(ContentRegexReplacer[l], line)
            for s in ContentSymbols:
                line = line.replace(s, " ")
            words = line.split()
            for word in words:
                word = word.lower()
                # processing word
                for wr in WordReplacer:
                    if word.__contains__(wr):
                        word = WordReplacer[wr]
                        break
                for wr in WordRegexReplacer:
                    if wr.fullmatch(word) != None:
                        word = WordRegexReplacer[wr]
                        break
                for ws in WordSymbols:
                    word = word.replace(ws, "")
                #ignore the word with only 1 characters except 'i'
                if (len(word) > 1 or word == 'i') and (word[0].isalpha() or word[0].isnumeric() or word[0:2]=="__"):
                    new_file.write(stem(word) + "\n")
        old_file.close()
        new_file.close()

def convert_sb_file(old_file_path, new_file_path):
    old_file = io.open(old_file_path, 'r', encoding='latin-1')
    try:
        lines = old_file.readlines()
    except:
        print(old_file_path)
        return
    new_file = open(new_file_path, 'w', encoding='utf-8')
    line_index = 0
    line = lines[line_index]
    line_index = line_index+1

    while line != "\n":
        line = lines[line_index]
        line_index = line_index+1
    line_index -=1
    while line_index<len(lines):
        line = lines[line_index]
        line_index = line_index+1
        for tag in Tags:
            if tag in line:
                continue
        for head in Headers:
            if head in line:
                continue
        for symbols in Symbols:
            if symbols in line:
                continue
        for l in ContentRegexReplacer:
            line = line.replace(str(l), ContentRegexReplacer[l])
        for s in ContentSymbols:
            line = line.replace(s, " ")
        words = line.split()
        for word in words:
            word = word.lower()
            for wr in WordReplacer:
                if wr in word:
                    word = WordReplacer[wr]
                    break
            for wr in WordReplacer:
                if wr in word:
                    word = WordReplacer[wr]
                    break
            for wr in WordRegexReplacer:
                if wr.fullmatch(word) != None:
                    word = WordRegexReplacer[wr]
                    break
            word  = word.replace(":", "")
            new_file.write(stem(word)+"\n")
    old_file.close()
    new_file.close()


def convert(train_dir, pre_dir):
    mail_files = os.listdir(train_dir)
    #file_count = 0
    for mailFile in mail_files:
        if os.path.isdir(mailFile):
            continue
        convert_file(os.path.join(train_dir, mailFile), os.path.join(pre_dir, str(file_count) + '.txt'))
        # convert_file(train_dir + "/" + mailFile, pre_dir + "/" + str(file_count) +".txt")
        #file_count += 1
       # print(str(file_count) + " files converted")

def debug_main():
    train_dir = "/home/linbuyu/Desktop/2018EXP2/spam"
    pre_dir = "/home/linbuyu/Desktop/2018EXP2/spam_pre"
    os.path.join(train_dir, '')
    os.path.join(pre_dir, '')

    if not os.path.exists(pre_dir):
        os.mkdir(pre_dir)

    for path, dirname, filenames in os.walk(train_dir):
        for filename in filenames:
            train_file = os.path.join(path, filename)
            pre_file = os.path.join(pre_dir, filename)
            print("old file:")
            convert_file(os.path.join(path, filename), os.path.join(pre_dir, filename))


    #convert_file(train_dir, pre_dir)

if __name__ == '__main__':
    debug_main()













