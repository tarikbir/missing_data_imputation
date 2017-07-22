import timeit
import numpy as np
import math
from collections import Counter
#################
# Init Settings #
#################
sep = ','  # Separator
fileName = "kddn"  # File name
fileNameLoss = fileName+".5loss"  # File name with lost data
#############
# Functions #
#############
def printLine():
    #Function that prints a line.
    print("--------------------------------------------")
def elapsed():
    #Function that calculates elapsed time. Needs init for global tT first.
    global tT
    t = abs(tT-timeit.default_timer())
    h = int((t) / 3600)
    m = int(((t) - 3600 * h) / 60)
    s = (((t) - 3600 * h) - 60 * m)
    tT = timeit.default_timer()
    return [h,m,s]
def findElement(ls,k):
    #Function to find an element in lists and return its index without generating ValueError.
    try:
        elm = ls.index(k)
        return elm
    except:
        return None
def isfloat(s):
    #Function to check if value is float. Returns true if castable.
    try:
        float(s)
        return True
    except:
        return False
def give_id(v):
    #Function to give ids to strings. Helps to make numerical calculations easier. Needs global id list and strings list.
    global id, strings
    if v in strings:
        return strings[v]
    else:
        print("NewID: {:>12s} replaced with id: {:<4d}".format(v, id))
        strings[v] = id
        id -= 1  # IDs are negative.
        return id+1
def get_id(v):
    #Function that returns the string of the given id.
    global strings
    return next((st for st, k in strings.items() if k == v), None)
def mse():
    #Function that calculates M S Error.
    total = 0.0
    global original, imported, missing, miss
    for idx, v in enumerate(missing):
        i, j = v
        x = imported[i][j]
        y = original[i][j]
        if isfloat(y):
            y = float(y)
        else:
            y = give_id(y)
        total += abs(x - y)
    return math.sqrt(total/miss)
###################
# File Operations #
###################
tT = timeit.default_timer()  # Initialization of elapsed() function.
with open(fileNameLoss, 'r', errors='replace') as inputFile, open(fileName, 'r', errors='replace') as inputFileOrg:
    original = []  # Imported original file without missing values
    imported = []  # Imported file as 2d array
    row = 0  # Imported file's rows
    missing = []  # Imported missing elements' indexes
    importedNM = []  # Imported non missing elements
    importedNM_index = []  # Imported non missing elements' indexes
    tagList = []  # Holds tags at the end of lines (to exclude them from imputation)
    tagListNM = []  # Holds non missing lines' tags at the end of lines (to use in LSE method)
    strings = {}  # Holds strings as ids to rewrite later
    style = []  # Holds input style to output similar to input
    id = -1  # Initial value of id.
    print("Info: Importing file [{}], please wait...".format(fileName))
    for idx, l in enumerate(inputFile):  #TODO: Make importing with numpy to get rid of redundant lists, many useless code and algorithms.
        l = l.replace('\n', '')  # Hardcoded to remove any unnecessary lines in a file.
        imported.append(l.split(sep))
        row += 1  # Cheap way to get row amount
    for idx_, l in enumerate(inputFileOrg):
        l = l.replace('\n', '')  # Hardcoded to remove any unnecessary lines in a file.
        original.append(l.split(sep))
    col = len(imported[0]) - 1  # Cheap way to get column amount
    print("Info: File has has {} rows and {} columns.".format(row,col))
    for i in imported[0]:  # Get value type (to rewrite later)
        if i.find('.') != -1:  # Cheap way to check if string is float or not.
            style.append('f')
        elif i.find('.') == -1:
            style.append('d')
    for i in range(row):
        tagList.append(imported[i][col])  # Get the tag of this row and then...
        del(imported[i][-1])  # ...remove it from the main list.
        missingFlag = False
        for j in range(col):  # Scan for missing elements.
            if imported[i][j] != '':  # If not missing, do conversions, give ids, etc.
                v = imported[i][j]
                if isfloat(v):  # Check to see if element can be converted to float...
                    imported[i][j] = float(v)  # ...if it can, cast it and add it to the list.
                else:
                    imported[i][j] = give_id(v)  # ...if it can't, give it an id and add the id to the list.
            else:  # If found a missing string:
                missing.append( [i,j] )  # Add the index to missing array.
                missingFlag = True  # Flag this row to make appropriate changes.
        if not missingFlag:
            importedNM_index.append(i) # Add indexes to the smaller NM array
            importedNM.append(imported[i])  # Add elements to the NM array
            tagListNM.append(tagList[i])  # Add the tag of that row to the NM tag array
    tags = Counter(tagList).most_common()
    miss = len(missing)
    print("Info: Data has {} missing elements".format(miss))
    dataSet = np.array(imported)  # Whole data set
    dataSetNM = np.array(importedNM)  # Non missing data set
    t = elapsed()
    print("Info: File import completed in [{:>2d}:{:>2d}:{:>4.1f}]".format(t[0],t[1],t[2]))
    printLine()
##############
# Imputation #
##############
    choice = int(input("Method?:\n•Least Squares Data Imputation (1)\n•Naive Bayes Imputation (2)\n•Hot Deck Imputation (3)\n•Imputation with Most Frequent Element (4)\nSelection:"))
    print("Info: Imputing process started... This may take a long time...")
    if choice == 1:  # -------------- Least Squares Data Imputation --------------
        nonZero = dataSetNM[:, ~np.all(dataSetNM == 0, axis=0)]  # Gets non-zero columns
        indexes = np.asarray(np.where(~dataSetNM.any(axis=0))).tolist()[0]  # Gets non-zero column indexes
        nonZeroT = nonZero.transpose()
        for i, t in enumerate(tagListNM):
            tagListNM[i] = give_id(t)  # Give ids to the tags (we need numerals)
        tagSet = np.array(tagListNM)
        B = np.dot(np.dot(np.linalg.pinv(np.dot(nonZeroT, nonZero)), nonZeroT), tagSet)  # LSE formula ((Bt*B)^-1)*Bt*y
        for idx, v in enumerate(missing):
            i,j = v
            sumB = sum([b*imported[i][idx] for idx, b in enumerate(B) if idx != j])  # Sum of all elements in B except the missing column's
            if findElement(indexes,j):
                dnm = B[findElement(indexes,j)]
                imported[i][j] = abs((tagSet[i] - sumB) / dnm)
            else:
                imported[i][j] = 0.0
            print("Info: Imputed", idx + 1, "out of", miss, ":", v, "-> (", imported[i][j], ")")
    elif choice == 2:  # -------------- Naive Bayes Imputation --------------
        totalTags = math.fsum([y[1] for y in tags])  # Tag Total
        for idx, v in enumerate(missing):
            i,j = v
            tagMiss = tagList[i]  # Missing data's tag
            currentColumn = [r[j] for k,r in enumerate(importedNM) if tagListNM[k] == tagMiss]
            imported[i][j] = Counter(currentColumn).most_common(1)[0][0]
            #TODO: Generate frequency tables beforehand to make imputing faster
            print("Info: Imputed", idx + 1, "out of", miss, ":", v, "-> (", imported[i][j], ")")
    elif choice == 3:  # -------------- Hot Deck Imputation --------------
        kHD = 20
        for idx,v in enumerate(missing):  # For each missing element in data set
            i,j = v
            euclidean = []
            euclideanTotal = 0
            for r in range(len(importedNM)):  # Loop all non-missing rows...
                for c in range(col):  # ...and all of its columns...
                    if c != j:  # ...except itself...
                        euclideanTotal += (imported[i][c] - importedNM[r][c])**2  # ...to calculate the euclidean distance of both...
                e = math.sqrt(euclideanTotal)
                euclidean.append( [e, importedNM_index[r]] )  # Append found euclidean and index of that in the original data set
            sorted(euclidean, key=lambda l: l[0], reverse=True)  # Sorts the euclidean list by their first value
            lst = [imported[euclidean[r][1]][j] for r in range(kHD)]  # Gets the list of first kHD elements of those values
            imported[i][j] = Counter(lst).most_common(1)[0][0]  # Imputes the most common element from above list.
            print("Info: Imputed", idx + 1, "out of", miss, ":", v, "-> (", imported[i][j], ")")
    elif choice == 4:  # -------------- Imputation with Most Frequent Element --------------
        for idx,v in enumerate(missing):
            i,j = v
            currentColumn = [r[j] for r in importedNM]
            imported[i][j] = Counter(currentColumn).most_common(1)[0][0]  # Simply imputes the most common element of that column regardless of any other information.
            print("Info: Imputed", idx + 1, "out of", miss, ":", v, "-> (", imported[i][j], ")")
    else:
        print("Error: Wrong input. No imputations done.")
    #####################################
    # End of imputations / File writing #
    #####################################
    t = elapsed()
    print("Info: Imputed list generated in [{:>2d}:{:>2d}:{:>4.1f}]".format(t[0],t[1],t[2]))
    printLine()
    print("Info: MSE: {:.3f}%".format(mse()))
    printLine()
    with open(fileName + ".imputed", 'w', errors='replace') as outputFile:
        print("Info: Generating output file...")
        for i in range(row):
            for j in range(col):
                x = imported[i][j]
                if x < 0:  # Hardcoded because data set doesn't have any negative floats (IDs are all negative)
                    imported[i][j] = get_id(x)
                else:
                    if style[j] == 'f':
                        x = "{:.2f}".format(x)
                    elif style[j] == 'd':
                        x = "{:d}".format(int(x))
                    imported[i][j] = str(x)
            line = sep.join(imported[i]) + "," + tagList[i] + '\n'
            outputFile.write(line)
        outputFile.truncate()
        t = elapsed()
        print("Info: Output file written in [{:>2d}:{:>2d}:{:>4.1f}]".format(t[0],t[1],t[2]))
        printLine()
    