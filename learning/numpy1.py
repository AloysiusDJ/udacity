import numpy as np

a = np.array([1,2,3])
a = np.array([1, 2, 3,4,5], ndmin = 2) 
a = np.array([1, 2, 3], dtype = complex) 
student = np.dtype([('name','S20'), ('age', 'i1'), ('marks', 'f4')]) 
a = np.array([('abc', 21, 50),('xyz', 18, 75)], dtype = student) 

a = np.array([[1,2],[3,4]])
print(a.shape)
a = np.array([[1,2,3],[4,5,6]]) 
a.shape = (3,2) #reshape

a = np.arange(24) # 1 dimension range
b = a.reshape(2,4,3)
x = np.empty([3,2], dtype = int) 
x = np.zeros((5,), dtype = np.int) 
x = np.ones([2,2], dtype = int) 

x = (1,2,3) 
a = np.asarray(x) 
s = 'Hello World' 
a = np.frombuffer(s, dtype = 'S1') 

list = range(5) 
it = iter(list)  
x = np.fromiter(it, dtype = float) 

x = np.arange(10,20,2) 
x = np.linspace(10,20,5) #even space
a = np.logspace(1,10,num = 10, base = 2) 

a[2:5]
a[...,1] #items from column
a[1,...]  # items from row
a[...,1:] #items from column onwards


x = np.array([[1, 2], [3, 4], [5, 6]]) 
y = x[[0,1,2], [0,1,0]] #elements at (0,0), (1,1) and (2,0) -> [1  4  5]

x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]]) 
z = x[1:4,1:3] 
y = x[1:4,[1,2]] 

x[x > 5] #list of items > 5
a = np.array([np.nan, 1,2,np.nan,3,4,5]) 
a[~np.isnan(a)]

c = a * b  #same dim

a = np.array([[1,2],[3,4]]) 
b = np.array([1,2]) #broadcasting ***
c=a+b

b = a.T #transpose
for x in np.nditer(b): #iterate

c = b.copy(order='C') #sort row 
c = b.copy(order='F') #sort columns

for x in np.nditer(a, op_flags=['readwrite']): #modifyiung array
   x[...]=2*x
   
for x,y in np.nditer([a,b]): #where a and b are arrays, simultaneously iterate

Changing Shape
1	reshape: Gives a new shape to an array without changing its data
2	flat : A 1-D iterator over the array
3	flatten : Returns a copy of the array collapsed into one dimension
4	ravel : Returns a contiguous flattened array

Transpose Operations
1	transpose : Permutes the dimensions of an array
2	ndarray.T : Same as self.transpose()
3	rollaxis : Rolls the specified axis backwards
4	swapaxes : Interchanges the two axes of an array

Changing Dimensions
1	broadcast : Produces an object that mimics broadcasting
2	broadcast_to : Broadcasts an array to a new shape
3	expand_dims : Expands the shape of an array
4	squeeze : Removes single-dimensional entries from the shape of an array

Joining Arrays
1	concatenate : Joins a sequence of arrays along an existing axis
2	stack : Joins a sequence of arrays along a new axis
3	hstack : Stacks arrays in sequence horizontally (column wise)
4	vstack : Stacks arrays in sequence vertically (row wise)

Splitting Arrays
1	split : Splits an array into multiple sub-arrays
2	hsplit : Splits an array into multiple sub-arrays horizontally (column-wise)
3	vsplit : Splits an array into multiple sub-arrays vertically (row-wise)

Adding / Removing Elements
1	resize : Returns a new array with the specified shape
2	append : Appends the values to the end of an array
3	insert : Inserts the values along the given axis before the given indices
4	delete : Returns a new array with sub-arrays along an axis deleted
5	unique : Finds the unique elements of an array

Vector string operations (numpy.char)
1	add() : Returns element-wise string concatenation for two arrays of str or Unicode
2	multiply() : Returns the string with multiple concatenation, element-wise
3	center() : Returns a copy of the given string with elements centered in a string of specified length
4	capitalize() : Returns a copy of the string with only the first character capitalized
5	title() : Returns the element-wise title cased version of the string or unicode
6	lower() : Returns an array with the elements converted to lowercase
7	upper() : Returns an array with the elements converted to uppercase
8	split() : Returns a list of the words in the string, using separatordelimiter
9	splitlines() : Returns a list of the lines in the element, breaking at the line boundaries
10	strip() : Returns a copy with the leading and trailing characters removed
11	join() : Returns a string which is the concatenation of the strings in the sequence
12	replace() : Returns a copy of the string with all occurrences of substring replaced by the new string
13	decode() : Calls str.decode element-wise
14	encode() : Calls str.encode element-wise

Trig:
sin = np.sin(a*np.pi/180) #convert to radians by multiplying pi/180
inv = np.arcsin(sin) 
np.degrees(inv) 


np.around(a) 
np.around(a, decimals = 1) 
np.around(a, decimals = -1)
np.floor(a)
np.ceil(a)
numpy.reciprocal() #elementwise, if <0 it is 0

np.power(a,2) 
np.power(a,b) # array a power of array b
np.mod(a,b) 

np.amin(a,1) #row
np.amin(a,0) #column
np.amax(a) 
np.amax(a, axis = 0) #column
np.ptp(a) #range(max-min)
np.ptp(a, axis = 1) 
np.ptp(a, axis = 0) 
np.percentile(a,50) #value below which a given percentage of observations in a group of observations fall
np.percentile(a,50, axis = 1)
np.percentile(a,50, axis = 0)
np.median(a) 
np.mean(a) 
np.average(a) #same as mean when no weight
np.average(a,weights = wts) #weighter average i.e sum of array*weights divided by sum of weights

#std = sqrt(mean(abs(x - x.mean())**2))
np.std([1,2,3,4])
np.var([1,2,3,4]) #square root of sd
 

np.sort(a)
np.sort(a, axis = 0)
np.sort(a, order = 'name')
np.argsort(x) #index of sort
np.argmax(a) 
np.argmin(a) 
np.nonzero (a) #indice of nonzero

y = np.where(x > 3) #indice with condition
x[y] #elements with condition as single list

condition = np.mod(x,2) == 0 
np.extract(condition, x) #'Extract elements using condition'

b = a.view() #shallow copy
s = a[:, :2] #slice creates shallow copy
b = a.copy() #deep copy

numpy.matlib.empty()
numpy.matlib.zeros()
numpy.matlib.ones()
numpy.matlib.eye()
numpy.matlib.identity()
numpy.matlib.rand()
numpy.linalg.dot()
numpy.linalg.vdot()
numpy.linalg.inner()
numpy.linalg.matmul()
numpy.linalg.determinent()
numpy.linalg.solve()
numpy.linalg.inv()
    


