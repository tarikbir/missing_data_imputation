# Imputation Methods for Missing Data

This is a basic python code to read a dataset, find missing data and apply imputation methods to recover data, with as less error as possible.

## About

This code is mainly written for a specific data set. Taken a specific route to write it as simple and shorter as possible. Since the debug data set was not very suitable for this kind of code, there are some necessary hard-codings.

Initialization has only the file name, and the separator used in the file type. Since the debug file was not readable with csv-reader functions, it reads the file as string and seperates it with given separator.
```python
sep = ','  # Separator
fileName = "kddn"  # File name
fileNameLoss = fileName+".5loss"  # File name with lost data (Used 5loss because my data was missing 5%)
createOutputFile = True
```
File import was done with [with open](https://docs.python.org/3.6/library/functions.html#open) method of python. It reads the file, line by line, then import them properly into a list. If data has strings or anything that can't be converted to float, the program should give it a numerical id to keep things easy to calculate. Then it converts the list into [numpy array](https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.ndarray.html#numpy.ndarray) to make calculations faster. Also, while importing, the program also finds and appends the missing values as indexes, while also generating a non-missing version of the imported file (if the row has a missing data, skip it) which makes calculations easier.
```python
original = []  # Imported original file without any missing values (this is for calculating mse, this debug file only)
imported = []  # Imported file as 2d array
missing = []  # Imported missing data indexes
importedNM = []  # Imported non-missing array
importedNM_index = []  # Imported non missing indexes (holds indexes of NM to rewrite later)
tagList = []  # Holds tags at the end of lines (to exclude them from imputation)
tagListNM = []  # Holds tags of non-missing lines
strings = {}  # Holds strings as ids to rewrite later
style = []  # Holds input style to output similar to input (int/float)
strID = 0  # Initial value of id.
stringColumns = []  # Holds if columns are strings or not
...
tags = Counter(tagList).most_common()  # Holds all tags and counts them
miss = len(missing)  # How many missing data are there
```
After importing, there are 4 imputation methods available to use in this code:
* [Least Squares Data Imputation](#least-squares-data-imputation)
* [Naive Bayes Imputation](#naive-bayes-imputation)
* [Hot Deck Imputation](#hot-deck-imputation)
* [Imputation with Most Frequent Element](#imputation-with-most-frequent-element)


The program loops every element of `missing` with;
```python
for idx,v in enumerate(missing):
    i,j = v  # Gets the index of missing element
```
And imputes each element with the methods below. After every missing data gets imputed, it calculates the [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error) and prints it out. Then starts writing the file.
```python
line = sep.join(imported[i]) + sep + tagList[i] + '\n'  # Reads the list as a row, adds the tag at the end, ends the line.
outputFile.write(line)
```
## Functions Used
* elapsedStr(): Function that calculates elapsed time and returns it as a string. Needs init for global tT first.
```python
t = abs(tT-timeit.default_timer())
h = int(t / 3600)
m = int((t - 3600 * h) / 60)
s = round((t - 3600 * h) - 60 * m)
tT = timeit.default_timer()
if h+m+s < 0.1:
    strT = "[really quick]."
else:
    strT = "in [{:>02d}:{:>02d}:{:>02d}].".format(h,m,s)
return strT
```
* isfloat(s): Function to check if value is `float`. Returns true if castable.
```python
try:
    float(s)
    return True
except:
    return False
```
* give_id(v): Function to give ids to strings. Helps to make numerical calculations easier. Needs global `strID` (id list) and `strings` list.
```python
if v in strings:
    return strings[v]
else:
    strings[v] = strID
    strID += 1
    return strID-1
```
* get_id(v): Function that returns the string of the given id. Needs global `strings` list.
```python
v=round(v)
return next((st for st, k in strings.items() if k == v), None)
```
* mse(): Function that calculates mean squared error.
```python
total = 0.0
for _, v in enumerate(missing):
    i, j = v
    x = imported[i][j]
    y = original[i][j]
    ...
    total += abs(x - y) #Adds everything to the grand total
return math.sqrt(total/miss) #Returns the root of the average
```
## Imputation Methods
For each code examples below; `imported` is the data set and `i,j` is the found missing data's index.
### Least Squares Data Imputation
This method imputes the missing data with least squares formula and rewrites the data.
```python
B = np.dot(np.dot(np.linalg.pinv(np.dot(nonZeroT, nonZero)), nonZeroT), tagSet)  # ß'=(Xᵀ.X)⁺.Xᵀ.y
...
sumB = sum([b*imported[i][idx] for idx, b in enumerate(B) if idx != j])  # Does dot product of B and row, except i, sums all.
imported[i][j] = (tagSet[i] - sumB) / B[index]  # Then solves x for ß'[j].x + sum_of_ß' = y[i]
```
### Naive Bayes Imputation
This method uses the Naive Bayes method to impute with frequency, in tandem with tags. Imputes the most frequent element on the column of the missing data with relation to same row's tag.
```python
tagMiss = tagList[i]  # Missing data's tag
currentColumn = [r[j] for k,r in enumerate(importedNM) if tagListNM[k] == tagMiss]  # Gets the whole column with matching tags.
imported[i][j] = Counter(currentColumn).most_common(1)[0][0]  # Imputes most common one.
```
### Hot Deck Imputation
This most common method gets the geometric distance of each row to the missing data's row and uses a kHD (default:20) value to determine how many of the most close rows' element should be picked as the most common one. In other words, imputes the geometrically closest rows' most common data.
```python
sorted(euclidean, key=lambda l: l[0], reverse=True)  # Sorts the euclidean distance list by their distance value [distance,index]
lst = [imported[euclidean[r][1]][j] for r in range(kHD)]  # Gets the list of first kHD elements of those values
imported[i][j] = Counter(lst).most_common(1)[0][0]  # Imputes the most common element from above list.
```
### Imputation with Most Frequent Element
This impractical method is just there to add some spice and allows comparison for other methods' results. It imputes the most common element of that column, regardless of anything else. Fast, but highly unreliable.
```python
currentColumn = [r[j] for r in importedNM]
imported[i][j] = Counter(currentColumn).most_common(1)[0][0]
```
## Bugs
[Bug reports](https://github.com/Tharky/missing_data_imputation/issues) and code recommendations are always appreciated.

There is also lots of TODO in the code, I'll get to fixing them later.
