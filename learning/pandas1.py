Series	1	1D labeled homogeneous array, sizeimmutable.
Data Frames	2	General 2D labeled, size-mutable tabular structure with potentially heterogeneously typed columns.

index(row, axis=1), column(axis=0)

import pandas as pd
import numpy as np

data = np.array(['a','b','c','d'])
s = pd.Series(data)
s = pd.Series(data,index=[100,101,102,103])

data = {'a' : 0., 'b' : 1., 'c' : 2.}
s = pd.Series(data)
s = pd.Series(data,index=['b','c','d','a']) #index order maintained, no value NaN

s = pd.Series(5, index=[0, 1, 2, 3])

s[0]
s[:3]
s[-3:] #last 3
s['a'] #retrive using index
s[['a','c','d']]


data = [1,2,3,4,5]
df = pd.DataFrame(data)
data = [['Alex',10],['Bob',12],['Clarke',13]]
df = pd.DataFrame(data,columns=['Name','Age'],dtype=float)
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data)
df = pd.DataFrame(data, index=['rank1','rank2','rank3','rank4'])
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)
df = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b'])
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
      'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
print(df)

df ['one'] #select column
df['three']=pd.Series([10,20,30],index=['a','b','c']) #add column
df['four']=df['one']+df['three']

del df['one']/df.pop('two')

df.loc['b'] #select row
df = df.append(df2) #add rows
df.iloc[2]
df[2:4] # selectmultiple rows

df = df.drop(0) #delete row with label 0

s = pd.Series(np.random.randn(4))
s.axes #row axes labels
s.empty
s.ndim
s.values

df.T #Transpose
df.axes #row and col axes labels
df.dtypes
df.emppty
df.ndim
df.shape
df.size
df.values

df.sum() #default axis is 0 column
df.sum(1)
df.mean()
df.std()
df.count() #non null
df.median()
df.mode()
df.min()
df.max()
df.prod()
df.cumsum()
df.cumprod(), df.abs() (only numeric)
df.describe() #only number
df.describe(include=['object']) #string
df.describe(include=['all'])

Table wise Function Application: pipe()
Row or Column Wise Function Application: apply()
Element wise Function Application: applymap()

def adder(ele1,ele2):
   return ele1+ele2

df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df.pipe(adder,2)
df.apply(np.mean)

df.apply(np.mean) #default col 0 axis
df.apply(np.mean,axis=1)
df.apply(lambda x: x.max() - x.min())

df['col1'].map(lambda x:x*100)
df.applymap(lambda x:x*100)


N=20
df = pd.DataFrame({
   'A': pd.date_range(start='2016-01-01',periods=N,freq='D'),
   'x': np.linspace(0,stop=N-1,num=N),
   'y': np.random.rand(N),
   'C': np.random.choice(['Low','Medium','High'],N).tolist(),
   'D': np.random.normal(100, 10, size=(N)).tolist()
})
#reindex the DataFrame
df_reindexed = df.reindex(index=[0,2,5], columns=['A', 'C', 'B'])
df1 = df1.reindex_like(df2)
#("Data Frame with Forward Fill:")
df2.reindex_like(df1,method='ffill') #bfill, nearest

df1.rename(columns={'col1' : 'c1', 'col2' : 'c2'}, index = {0 : 'apple', 1 : 'banana', 2 : 'durian'}) # inplace=True def is False

for key in df: 
for key,value in df.iteritems():
for row_index,row in df.iterrows(): #row is Series
for row in df.iteritems():     #row first value is index

sorted_df=unsorted_df.sort_index() #def sorted on row labels in ascending
sorted_df = unsorted_df.sort_index(ascending=False) # axis=1 

sorted_df = unsorted_df.sort_values(by='col1')
sorted_df = unsorted_df.sort_values(by=['col1','col2'])
sorted_df = unsorted_df.sort_values(by='col1' ,kind='mergesort')


Series String functions:
    s = pd.Series(['Tom', 'William Rick', 'John', 'Alber@t', np.nan, '1234','SteveSmith'])
    s.str.lower()
    
1	lower()	Converts strings in the Series/Index to lower case.
2	upper()	Converts strings in the Series/Index to upper case.
3	len()	Computes String length().
4	strip()	Helps strip whitespace(including newline) from each string in the Series/index from both the sides.
5	split(' ')	Splits each string with the given pattern.
6	cat(sep=' ')	Concatenates the series/index elements with given separator.
7	get_dummies()	Returns the DataFrame with One-Hot Encoded values.
8	contains(pattern)	Returns a Boolean value True for each element if the substring contains in the element, else False.
9	replace(a,b)	Replaces the value a with the value b.
10	repeat(value)	Repeats each element with specified number of times.
11	count(pattern)	Returns count of appearance of pattern in each element.
12	startswith(pattern)	Returns true if the element in the Series/Index starts with the pattern.
13	endswith(pattern)	Returns true if the element in the Series/Index ends with the pattern.
14	find(pattern)	Returns the first position of the first occurrence of the pattern.
15	findall(pattern)	Returns a list of all occurrence of the pattern.
16	swapcase	Swaps the case lower/upper.
17	islower()	Checks whether all characters in each string in the Series/Index in lower case or not. Returns Boolean
18	isupper()	Checks whether all characters in each string in the Series/Index in upper case or not. Returns Boolean.
19	isnumeric()	Checks whether all characters in each string in the Series/Index are numeric. Returns Boolean.


    
pd.get_option("display.max_rows")
pd.get_option("display.max_columns")
pd.set_option("display.max_rows",80)
pd.reset_option("display.max_rows")
pd.describe_option("display.max_rows")

.loc()	Label based #scalar label, list oi labels, slice, boolean array
df.loc[:,'A'] #row, col
df.loc['a']>0

.iloc()	Integer based #int, list, range
df.iloc[1:5, 2:4]

.ix()	Both Label and Integer based
df.ix[:4]#rows
print df.ix[:,'A']

Series	s.loc[indexer]	Scalar value
DataFrame	df.loc[row_index,col_index]	Series object

df['A']
df.A
df[['A','B']]
df[2:2]

df.pct_change()

#covariance
s1 = pd.Series(np.random.randn(10))
s2 = pd.Series(np.random.randn(10))
s1.cov(s2)
NOTE: Covariance method when applied on a DataFrame, computes cov between all the columns.
frame = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
frame['a'].cov(frame['b'])
frame['a'].corr(frame['b']) #correlation

s = pd.Series(np.random.np.random.randn(5), index=list('abcde'))
s.rank() #average , min, max, first

Function on series of data
df.rolling(window=3).mean()
df.expanding(min_periods=3).mean()
df.ewm(com=0.5).mean()

r = df.rolling(window=3,min_periods=1)
r.aggregate(np.sum)
r['A'].aggregate(np.sum)
r[['A','B']].aggregate(np.sum)
r['A'].aggregate([np.sum,np.mean])
r.aggregate({'A' : np.sum,'B' : np.mean})

df['one'].isnull()
df['one'].notnull()
df.fillna(0)
df.fillna(method='pad') #fill
df.fillna(method='bfill')
df.dropna()
df.dropna(axis=1)
df.replace({1000:10,2000:60})


ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)
df.groupby(['Team','Year']).groups
grouped=df.groupby('Team')
for name,group in grouped:
    print name
    print group
grouped.get_group(2014)

grouped['Points'].agg(np.mean)
grouped.agg(np.size)
grouped['Points'].agg([np.sum, np.mean, np.std])

score = lambda x: (x - x.mean()) / x.std()*10
grouped.transform(score)

df.groupby('Team').filter(lambda x: len(x) >= 3)

JOINS
pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,left_index=False, right_index=False, sort=True)
pd.merge(left,right,on='id') # left and right dataframes
pd.merge(left,right,on=['id','subject_id'])
pd.merge(left, right, on='subject_id', how='left') #left join, right jin, outer join, inner join

pd.concat([one,two])
pd.concat([one,two],keys=['x','y'])
pd.concat([one,two],keys=['x','y'],ignore_index=True) #for unique indexes
pd.concat([one,two],axis=1) #new cols added

one.append(two) #same as concat axis=0 
one.append([two,one,two])

pd.datetime.now()
pd.Timestamp('2017-03-01')
pd.date_range("11:00", "13:30", freq="30min").time
pd.to_datetime(pd.Series(['Jul 31, 2009','2010-01-10', None]))
pd.date_range('1/1/2011', periods=5) #def D day
pd.date_range('1/1/2011', periods=5,freq='M')
pd.bdate_range('1/1/2011', periods=5)

pd.Timedelta('2 days 2 hours 15 minutes 30 seconds')
pd.Timedelta(6,unit='h')

s = pd.Series(pd.date_range('2012-1-1', periods=3, freq='D'))
td = pd.Series([ pd.Timedelta(days=i) for i in range(3) ])
df = pd.DataFrame(dict(A = s, B = td))
df['C']=df['A']+df['B']

s = pd.Series(["a","b","c","a"], dtype="category")
cat = pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'])

df = pd.DataFrame(np.random.randn(10,4),index=pd.date_range('1/1/2000',periods=10), columns=list('ABCD'))
df.plot()
df = pd.DataFrame(np.random.rand(10,4),columns=['a','b','c','d')
df.plot.bar()
df = pd.DataFrame(np.random.rand(10,4),columns=['a','b','c','d')
df.plot.bar(stacked=True)
df = pd.DataFrame(np.random.rand(10,4),columns=['a','b','c','d')
df.plot.barh(stacked=True)
df = pd.DataFrame({'a':np.random.randn(1000)+1,'b':np.random.randn(1000),'c':np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])
df.plot.hist(bins=20)
df=pd.DataFrame({'a':np.random.randn(1000)+1,'b':np.random.randn(1000),'c':np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])
df.diff.hist(bins=20)
df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
df.plot.box()
df = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
df.plot.area()
df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
df.plot.scatter(x='a', y='b')
df = pd.DataFrame(3 * np.random.rand(4), index=['a', 'b', 'c', 'd'], columns=['x'])
df.plot.pie(subplots=True)

pandas.read_csv(filepath_or_buffer, sep=',', delimiter=None, header='infer',names=None, index_col=None, usecols=None
df=pd.read_csv("temp.csv",index_col=['S.No'])
df = pd.read_csv("temp.csv", dtype={'Salary': np.float64}) #convert
df=pd.read_csv("temp.csv", names=['a', 'b', 'c','d','e'])
df=pd.read_csv("temp.csv", skiprows=2)

s = pd.Series(list('abc'))
s = s.isin(['a', 'c', 'e']) #return series of booleans


SELECT total_bill, tip, smoker, time FROM tips LIMIT 5;
tips[['total_bill', 'tip', 'smoker', 'time']].head(5)
SELECT * FROM tips WHERE time = 'Dinner' LIMIT 5;
tips[tips['time'] == 'Dinner'].head(5)
SELECT sex, count(*) FROM tips GROUP BY sex;
tips.groupby('sex').size()


