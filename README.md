# Imputation Methods for Missing Data

This is a basic python code to read a dataset, find missing data and apply imputation methods to try to recover some data with some error.

## About

There are 4 imputation methods used in this code (available to select within the runtime of the code):
* [Least Squares Data Imputation](#least-squares-data-imputation)
* [Naive Bayes Imputation](#naive-bayes-imputation)
* [Hot Deck Imputation](#hot-deck-imputation)
* [Imputation with Most Frequent Element](#imputation-with-most-frequent-element)

## Methods
For each code examples below; `imported` is the data set and `i,j` is the found missing data's index.
### Least Squares Data Imputation
This method imputes the missing data with least squares formula and rewrites the data.
```python
B = np.dot(np.dot(np.linalg.pinv(np.dot(nonZeroT, nonZero)), nonZeroT), tagSet)  # ß'=(Xᵀ.X)⁺.Xᵀ.y
...
sumB = sum([b*imported[i][idx] for idx, b in enumerate(B) if idx != j])  # Does dot product of B and row, except i, sums all.
imported[i][j] = abs((tagSet[i] - sumB) / B[index])  # Then solves x for ß'[j].x + sum_of_ß' = y[i]
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
