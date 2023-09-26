import sys
import re
import gzip


def tokenize(txtFile):
    content_string = ""
    token_list = []
    with gzip.open(txtFile, mode='rt', encoding='utf-8') as inFile:
        content_string = inFile.read()

    group1 = re.findall(r'([$\w\'-]+)|([^$\w\s\'-]+)', content_string)
    # flatten the list to remove the resultant tuple
    result_list = [match for group in group1 for match in group if match]
    return result_list


txtFile = sys.argv[1]

result = tokenize(txtFile)

for i in result:
    print(i)
