print("Python for Data Science")

# to know the python version
import sys
print(sys.version)

# some tools in writing
# to end line and begin new line
print("Wael Chmaisani\n Lebanon")
# to do space tab
print("Wael Chmaisani\t PhD in Computational Physics")
# to write a backslash
print("wael\\chmaisani")

# Expressions and variables

x=2+4*5-6/5+116//3
print(x)
print(type(x))
y=int(x)
print(y)
print(type(y))

# String
A="Wael Chmaisani"
print(len(A))
print(A[0])
print(A[-1])
print(A[4])
print(A[5])
print(A[0:4])
print(A[-10:-1])
print(A[::2])
print(A[0:4:2])
B=A+" PhD in computationl Physics"
print(B)
C=2*A
print(C)
print(A.upper())
print(B.lower())
print(A.replace("Wael","Nader"))
print(A.find("Chmaisani"))
print(A.find("R"))

# Tuple
X=(("Wael","Chmaisani"),1,"Nader",(1.76,82))
print(type(X))
print(len(X))
print(X[0])
print(X[0:3])
print(X[-2])
print(X[0][1])
print(X[0][1][5])
F=("PhD",2)
W=X+F
print(W)

Z=(2,6,4,-1,10,3,-2)
G=sorted(Z)
print(G)

# List
Q=[["Wael","chmaisani"],(1988,33),1.76,82,["Nader",1]]
print(Q)
print(len(Q))
print(type(Q))
print(Q[0][0])
print(Q[3])
print(Q[0:3])
print(Q[0:4:2])
del(Q[4])
Q[1]=1988
Q[0][0]="Nader"
print(Q)
P=Q
L=Q[:]
P.extend(["PhD","Physics"])
L.append(["PhD",2019])
print(Q)
print(P)
print(L)

# Transfer a string to a list
H=A.split()
print(H)

# Set
F={1,2,3,4,5,6,7,8,9,10}
K={3,8,14,22}
print(type(F))
print(len(F))
print(1 in F)
print(22 in F)
F.add(11)
F.remove(10)
print(F)
print(F.intersection(K))
print(F.union(K))
print(F.issuperset(K))
print(K.issubset(F))

# Transfer a list to a set
D=set(H)
print(D)

# Dictionary
V={"Name":"Wael","Family":"Chmaisani","Birthday":1988,"Major":"Physics"}
print(V)
print(type(V))
print(len(V))
print(V["Name"])
print(V["Birthday"])
print(V.keys())
print(V.values())
print("Name" in V)
del(V["Major"])
V["Graduation"]=2019
print(V)

# Branching (if,else,elseif)
GPA=3.8
if(GPA>=3):
    print("your application is approved")
    if(GPA>=3.7)and (GPA<=4):
        print("you can get a scholarship")
elif(GPA==2.9):
    print("need an interview")
else:
    print("your application is not approved")
print("Thank you")

x=2
y=7
if(x+y>=10):
    z=x+y
    print("z=",z)
elif(x==y):
    z=2*x+2*y
    print("z=",z)
else:
    z=x*y
    print("z=",z)
    
# "range function" and "for loop"
A=[1,1,2,3,4]

for a in A:
    a
    print("the element of A is",a)
    
for i,a in enumerate(A):
    a
    i
    print("the index",i,"corresponds to",a)
    
for i in range(0,5):
    A[i]=0
    print(A)

B=[1,4,8,12,31,45,66,22,18]
for i in range(0,9):
    if(B[i]<=60):
        B[i]=0
        print(B)
    else:
        print(B)

# "While loop"

x=3
y=1
while(y!=x):
    print("the value of y is",y)
    y=y+1


A=[0,0,3,0]
B=[]
i=0
while(A[i]==0):
    B.append(A[i])
    print(B)
    i=i+1

C=[1,4,8,12,31,45,66,22,18]
print(C)
i=0
while(C[i]<=60):
    C[i]=0
    print(C)
    i=i+1
# Building functions

def add1(a):
    '''
    add 1 to a
    '''

    b=a+1
    print("the value of a is added 1 and gives b =",b)
    return(b)

z=add1(6)
print(z)

help(add1)

def Mult(a,b):
    '''
    Multiplication between two values
    '''
    c=a*b
    print("the multiplication of",a,"and",b,"is",c)
    return(c)

y=Mult(4,8)
print(y)
v=Mult(4,"Wael")
print(v)

help(Mult)

def NoWork():
    pass
    
print(NoWork())


def printlist(A):
    for i,a in enumerate(A):
        a
        i
        print("the index",i,"corresponds to the value",a)

A=[10,20,30,40]
print(printlist(A))

def createtuple(*a):
    print(a)
    return(a)

print(createtuple("Wael","Chmaisani"))
print(createtuple("Wael","Ali","Chmaisani"))

def addChmaisani(x):
    x=x+"Chmaisani"
    print("My name is",x)
    return(x)
x="Nader"

print(addChmaisani(x))

def graduationyear():
    date=2019
    return(date)
    
date=2012

print(graduationyear())
print(date)


def graduationyear():
     global date
     date=2019
     return(date)
    
date=2012

print(graduationyear())
print(date)

def addx(y):
    y=y+x
    print(x)
    return y
x=5
print(addx(4))

# Class


class Circle():
    def __init__(self,radius,color):
        self.radius=radius;
        self.color=color;
    def add_radius(self,r):
        self.radius=self.radius+r;
    def perimeter(self):
        p=2*3.13*self.radius
        return p
    def area(self):
        a=3.14*self.radius**2
        return a
 
    
C=Circle(10,"red")
print(C.radius)
print(C.color)
C.radius=5
C.color="blue"
print(C.radius)
print(C.color)
C.add_radius(8)
print(C.radius)
print(C.perimeter())
print(C.area())
print(dir(Circle))

class Rectangle():
    def __init__(self,height,width,color):
        self.height=height;
        self.width=width;
        self.color=color;
    def add(self,h,w):
        self.height=self.height+h;
        self.width=self.width+w;
    def perimeter(self):
        p=2*(self.height+self.width)
        return p
    def area(self):
        a=self.height*self.width
        return a

R=Rectangle(4,3,"red")
print(R.height)
print(R.width)
print(R.color)
R.add(2,2)
print(R.height)
print(R.width)
print(R.perimeter())
print(R.area())
R.width=10
R.height=5
R.color="blue"
print(R.height)
print(R.width)
print(R.color)
print(dir(Rectangle))

# Working with files
PY102=open("/Users/Toshiba/Desktop/PY102.txt","a")
print(PY102.name)
print(PY102.mode)

PY103=open("/Users/Toshiba/Desktop/PY103.txt","w")
with open("PY103.txt","w") as PY103:
    PY103.write("Wael")
    
PY102=open("/Users/Toshiba/Desktop/PY102.txt","r")
PY103=open("/Users/Toshiba/Desktop/PY103.txt","w")
with open("PY102.txt","r") as PY102:
    with open("PY103.txt","w") as PY103:
        for line in PY102:
            PY103.write(line)

# Pandas
import pandas as pd
A = {"Name":["Wael", "Rola", "Nader","Angy","Nelly","Berlinti"],"Birthday":[1988,1990,2019,2019,2011,2011],
"Hoby":["Basketball","Party","Painting","Watch TV","Playing","Phone"]}
print(A)
df=pd.DataFrame(A)
print(df)
print(df.head())
print(df.iloc[0,0])
print(df.iloc[0:2,0:3])
print(df.iloc[0:6,0])
print(df.iloc[0,0:3])
print(df["Birthday"])
print(df["Birthday"].unique())
print(df["Birthday"]>=2000)
df1=df[["Birthday"]]
df2=df[["Name","Birthday"]]
print(df1)
print(df2)
df.to_csv("Parent.csv")

# Numpy and matplotlip
# 1D array+
import numpy as np
a=np.array([0,1,2,3,4,5])
print(a)
print(a[0])
print(a[0:4])
print(type(a))
print(a.dtype)
print(a.size)
print(a.ndim)
print(a.shape)
a[0]=1
print(a)
print(a.mean())
print(a.max())

u=np.array([1,0])
v=np.array([0,1])
z=u+v
print(z)
x=u-v
print(x)
w=u*2
print(w)
k=u+3
print(k)
p=u*v
print(p)
d=np.dot(u,v)
print(d)

x=np.array([0,np.pi/2,np.pi])
y=np.sin(x)
print(y)

l=np.linspace(-2,2,num=5)
print(l)

import matplotlib.pyplot as plt
x=np.arange(0,2*np.pi,0.1)
y=np.sin(x)
plt.plot(x,y)

# 2D array
import numpy as np
b=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(b[0][0])
print(b[2][3])
print(type(b))
print(b.dtype)
print(b.size)
print(b.ndim)
print(b.shape)

X=np.array([[1,0],[0,1]])
Y=np.array([[2,1],[1,2]])
Z=X+Y
print(Z)
S=X-Y
print(S)
P=2*X
print(P)
U=1+X
print(U)
W=X*Y
print(W)
T=np.dot(X,Y)
print(T)

#**************************************************************************************************************************************************

print("Data Analysis with Python")

# Importing a csv file, exporting a Pandas dataframe, starting statistical analysis

import pandas as pd
import numpy as np
url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
df=pd.read_csv(url, header=None)
print(df)

print("the first 10 rows of the data frame")
print(df.head(10))
print("the last 10 rows of the data frame")
print(df.tail(10))

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", headers)
df.columns=headers
print("the data frame with inserting headers\n")
print(df.head(10))
print("the types of data")
print(df.dtypes)
print("Find the name of the columns of the dataframe")
print(df.columns)
print(df.info())

print("save df as excel and csv file")
df.to_csv(r"C:\Users\Toshiba\Desktop\automobile.csv")
df.to_excel(r"C:\Users\Toshiba\Desktop\automobile.xlsx")

print("replace ? by not a number NaN")
df1=df.replace("?",np.NaN)
print(df1.head(10))
print("We can drop missing values along the column price")
df=df1.dropna(subset=["price"], axis=0)
print(df.head(10))

print(df.describe())
print(df.describe(include="all"))

df2=df[["symboling","normalized-losses","make"]]
print(df2)
print(df2.describe(include="all"))

# Data wrangling

import pandas as pd
import numpy as np
url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
df=pd.read_csv(url, header=None)

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns=headers
print(df)

print("replace the ? in df by nan")
df1=df.replace("?",np.nan)
df=df1[:]
print(df)

print("drop the rows in the normalized-losses that have nan")
df.dropna(subset=["normalized-losses"],axis=0,inplace=True)
print(df)

print("replace nan in price column by its mean")
df["price"]=df["price"].astype("float")
m=df["price"].mean()
print(m)
df["price"]=df["price"].replace(np.nan,m)

print("replace nan in normalized-losses column by its mean")
df["normalized-losses"]=df["normalized-losses"].astype("float")
m=df["normalized-losses"].mean()
print(m)
df["normalized-losses"]=df["normalized-losses"].replace(np.nan,m)
print(df.head(25))


print("formating the city-mpg and change its name to city-L/100km")
df["city-mpg"]=235/df["city-mpg"]
df.rename(columns={"city-mpg":"city-L/100km"},inplace=True)
print(df.head(5))

print("Normalization")
df["length"]=df["length"]/df["length"].max()
print(df[["length"]])
df["length"]=(df["length"]-df["length"].min())/(df["length"].max()-df["length"].min())
print(df[["length"]])
df["length"]=(df["length"]-df["length"].mean())/df["length"].std()
print(df[["length"]])

print("Binning the price column")
bins=np.linspace(min(df["price"]),max(df["price"]),4)
bins_names=["Low","Medium","High"]
df["price-binned"]=pd.cut(df["price"],bins,labels=bins_names,include_lowest=True)
print(df[["price-binned"]])
print(df)

print("Turning the categorial variables of fuel-type to quantitative variable 0 and 1")
print(df[["fuel-type"]])
pd.get_dummies(df["fuel-type"])
print(df.iloc[0:50,15:20])


#Explotory Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway


url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
df=pd.read_csv(url, header=None)

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns=headers
print(df)

print("replace the ? in df by nan")
df1=df.replace("?",np.nan)
df=df1[:]
print(df)

print(df.describe())

print(df.describe(include="all"))

df["price"]=df["price"].astype("float")
df.dropna(subset=["price"],axis=0,inplace=True)
drive_wheels_count=df["drive-wheels"].value_counts().to_frame()
drive_wheels_count.rename(columns={"drive-wheels":"count"},inplace=True)
drive_wheels_count.index.name="drive-wheel"
print(drive_wheels_count)

sns.boxplot(x="drive-wheels",y='price',data=df)

x=df["engine-size"]
y=df["price"]
plt.scatter(x,y)
plt.title("engine size vs price")
plt.xlabel("engine-size")
plt.ylabel("price")

dfgroup_mean=df[["drive-wheels","body-style","price"]].groupby(["drive-wheels","body-style"],as_index=False).mean()
print(dfgroup_mean)
df_pivot=dfgroup_mean.pivot(index="drive-wheels",columns="body-style")
df_pivot=df_pivot.fillna(0) #fill missing values with 0
print(df_pivot)

fig, ax = plt.subplots()
im = ax.pcolor(df_pivot, cmap='RdBu')
#label names
row_labels = df_pivot.columns.levels[1]
col_labels = df_pivot.index
#move ticks and labels to the center
ax.set_xticks(np.arange(df_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(df_pivot.shape[0]) + 0.5, minor=False)
#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)
#rotate label if too long
plt.xticks(rotation=90)
#show colobar and the heatmap
fig.colorbar(im)
plt.show()



df_anova=df[["make","price"]]
make=df_anova.groupby(["make"])
print(make.get_group("honda")["price"])
print(make.get_group("subaru")["price"])
results_F_p=stats.f_oneway(make.get_group("honda")["price"],make.get_group("subaru")["price"])
print(results_F_p)

sns.regplot(x="engine-size",y="price",data=df)


df.dropna(subset=["horsepower"],axis=0,inplace=True)
df["horsepower"]=df["horsepower"].astype("float")
pearson_coef,p_value=stats.pearsonr(df["horsepower"],df["price"])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
print(df.corr())
print(df[["horsepower","price"]].corr())

# Model Development

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
df=pd.read_csv(url, header=None)

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns=headers
print(df)

print("replace the ? in df by nan")
df1=df.replace("?",np.nan)
df=df1[:]
print(df)
df["price"]=df["price"].astype("float")
m=df["price"].mean()
print(m)
df["price"]=df["price"].replace(np.nan,m)
df["horsepower"]=df["horsepower"].astype("float")
m1=df["horsepower"].mean()
print(m1)
df["horsepower"]=df["horsepower"].replace(np.nan,m1)

print("Simple Linear Regression Model SLR")
lm=LinearRegression()
X=df[["highway-mpg"]]
Y=df[["price"]]
lm.fit(X,Y)
print(lm.intercept_)
print(lm.coef_)
Yhat=lm.predict(X)
print(Yhat)

print("Multiple Linear Regression Model MLR")
lm=LinearRegression()
Z=df[["horsepower","curb-weight","engine-size","highway-mpg"]]
Y=df["price"]
lm.fit(Z,Y)
print(lm.intercept_)
print(lm.coef_)
Yhat=lm.predict(Z)
print(Yhat)

print("Regression Plot for SLR")
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg",y="price",data=df)
plt.ylim(0,)

print("Residual Plot for SLR")
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df["highway-mpg"],df["price"])


print("Distribution Plot for MLR")
lm=LinearRegression()
Z=df[["horsepower","curb-weight","engine-size","highway-mpg"]]
Y=df["price"]
lm.fit(Z,Y)
print(lm.intercept_)
print(lm.coef_)
Yhat=lm.predict(Z)
print(Yhat)

width = 12
height = 10
plt.figure(figsize=(width, height))
ax1=sns.distplot(df["price"],hist=False,color="r",label="Actual Value")
sns.distplot(Yhat,hist=False,color="b",label="Fitted Value",ax=ax1)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.show()
plt.close()


print("Third order polynomial regression model base")
x = df['highway-mpg']
y = df['price']
f=np.polyfit(x,y,3)
p=np.poly1d(f)  # to display the polynomial function
print(p)

print("Define PlotPolly function to draw the one dimensional polynomial functions")
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)
    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ highway-mpg')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')
    plt.show()
    plt.close()
PlotPolly(p, x, y,'highway-mpg')

print("Multivariate Polynomial function of degree 2)
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)
Z=df[["horsepower","curb-weight","engine-size","highway-mpg"]]
Y=df["price"]
Z_pr=pr.fit_transform(Z)
print(Z.shape)
print(Z_pr.shape)

print("but pipline method is more simple")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
Z=df[["horsepower","curb-weight","engine-size","highway-mpg"]]
Y=df["price"]
input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(degree=2,include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(input)
Z = Z.astype(float)
pipe.fit(Z,Y)
ypipe=pipe.predict(Z)
print(ypipe)
print(ypipe[0:4])

print("Mean Square Error MSE and R-Square for SLR")
lm=LinearRegression()
X=df[["highway-mpg"]]
Y=df[["price"]]
lm.fit(X,Y)
Yhat=lm.predict(X)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Yhat)
print('The mean square error of price and predicted value is: ', mse)
R2=lm.score(X, Y)
print('The R-square is: ',R2)

print("Mean Square Error MSE and R-Square for MLR")
lm=LinearRegression()
Z=df[["horsepower","curb-weight","engine-size","highway-mpg"]]
Y=df["price"]
lm.fit(Z,Y)
Yhat=lm.predict(Z)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Yhat)
print('The mean square error of price and predicted value is: ', mse)
R2=lm.score(Z, Y)
print('The R-square is: ',R2)

print("Mean Square Error MSE and R-Square for polynomial fitting")
x = df['highway-mpg']
y = df['price']
f=np.polyfit(x,y,3)
p=np.poly1d(f)  # to display the polynomial function
print(p)


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y,p(x))
print('The mean square error of price and predicted value is: ', mse)
from sklearn.metrics import r2_score
R2=r2_score(y, p(x))
print('The R-square is: ',R2)

# Model Evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import clean data 
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'
df = pd.read_csv(path)
df.to_csv(r"C:\Users\Toshiba\Desktop\module_5_auto.csv")

df=df._get_numeric_data()
print(df.head())

    
#Training and Testing
#We will place the target data price in a separate dataframe y_data and Drop price data in dataframe x_data
y_data=df["price"]
x_data=df.drop("price",axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

from sklearn.linear_model import LinearRegression
lre=LinearRegression()
#We fit the model using the feature "horsepower"
lre.fit(x_train[['horsepower']], y_train)
#Let's calculate the R^2 on the test data
test_R=lre.score(x_test[['horsepower']], y_test)
print(test_R)
#We can see the R^2 is much smaller using the test data compared to the training data
train_R=lre.score(x_train[['horsepower']], y_train)
print(train_R)

#Sometimes you do not have sufficient testing data; as a result, you may want to perform cross-validation. Let's go over several methods that you can use for cross-validation
from sklearn.model_selection import cross_val_score
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
#The default scoring is R^2. Each element in the array has the average R^2 value for the fold:
print(Rcross)   
#We can calculate the average and standard deviation of our estimate
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())
#you can also use the function 'cross_val_predict' to predict the output. The function splits up the data into the specified number of folds, with one fold for testing and the other folds are used for training
from sklearn.model_selection import cross_val_predict
yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
print(yhat[0:5])


#overfitting,underfitting and Model selection

#Let's create Multiple Linear Regression objects and train the model using 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg' as features.

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print(yhat_train[0:5])
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print(yhat_test[0:5])


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()

#Let's examine the distribution of the predicted values of the training data.
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
#So far, the model seems to be doing well in learning from the training dataset. But what happens when the model encounters new data from the testing dataset? When the model generates new values from the test data, we see the distribution of the predicted values is much different from the actual target values.
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)

#Let's see if polynomial regression also exhibits a drop in the prediction accuracy when analysing the test dataset.
from sklearn.preprocessing import PolynomialFeatures
#overfitting
#Let's create a degree 5 polynomial model
y_data=df["price"]
x_data=df.drop("price",axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
#Now, let's create a Linear Regression model "lrpoly" and train it
lrpoly = LinearRegression()
lrpoly.fit(x_train_pr, y_train)


#R^2 of the training data
R_train=lrpoly.score(x_train_pr, y_train)
print(R_train)
#R^2 of the test data-A negative R^2 is a sign of overfitting.
R_test=lrpoly.score(x_test_pr, y_test)
print(R_test)


def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)
    
    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
    
PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, lrpoly,pr)

#The following function will be used in the next section
#The following interface allows you to experiment with different polynomial orders and different amounts of data.

def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train,y_test, poly, pr)
interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))

#Let's see how the R^2 changes on the test data for different order polynomials and then plot the results:

y_data=df["price"]
x_data=df.drop("price",axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)

Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:

    from sklearn.preprocessing import PolynomialFeatures
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])    
    
    lr.fit(x_train_pr, y_train)
    
    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')


#Redige Regression
y_data=df["price"]
x_data=df.drop("price",axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)
print("number of test samples :", x_test.shape[0])

from sklearn.preprocessing import PolynomialFeatures    
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])

from sklearn.linear_model import Ridge
rm=Ridge(alpha=1)
rm.fit(x_train_pr, y_train)
yhat = rm.predict(x_test_pr)
print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)

#We select the value of alpha that minimizes the test error. To do so, we can use a for loop. We have also created a progress bar to see how many iterations we have completed so far
from tqdm import tqdm
Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

#We can plot out the value of R^2 for different alphas
width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
#Here the model is built and tested on the same data, so the training and test data are the same.

#Grid Search

y_data=df["price"]
x_data=df.drop("price",axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)
print("number of test samples :", x_test.shape[0])


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

parameters= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]
Rm=Ridge()
Grid = GridSearchCV(Rm, parameters,cv=4, iid=None)
Grid.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
BestRm=Grid.best_estimator_
print(BestRm)
test_score=BestRm.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)
print(test_score)

################################################################################################################
# Summary of PY102: Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Starting point of data analysis 
#import the url of data and transform it to a data frame df
url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
df=pd.read_csv(url,header=None)
#make headears for df
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns=headers
print(df)
#get information about df
print(df.info()) #count, type of each column and the memory usage
print(df.head(5)) # show the first n rows
print(df.tail(5)) # show the last n rows
print(df.dtypes) # show the type of each column (int64,float64 or object)
print(df.columns) # show the names of columns
print(df.describe()) # get a statistical information of each column (count,mean,standard deviation,quartiles,min,max)
print(df.describe(include="all")) # get additional statistical information concerning the columns that have object (number of unique elements,top element,frequency of top element)

# export the df data as csv file
#df.to_csv(r"C:\Users\Toshiba\Desktop\CarPrice.csv")

# Data Cleaning

# replace the "?" to not a number "nan"
df=df.replace("?",np.nan)
print(df)

# drop the nan of the price column
df.dropna(subset=["price"],axis=0, inplace=True) #axis=0 (drop row that has nan), axis=1 (drop column that has nan), inplace for do not need create new df
print(df[["price"]])
print(df[["price"]].head(30))

# change the type of a column
df["price"]=df["price"].astype("float")
print(df.dtypes)

# replace the nan of normalized losses by the mean of that column
df["normalized-losses"]=df["normalized-losses"].astype("float")
mean=df["normalized-losses"].mean()
print(mean)
df=df.replace(np.nan,mean)
print(df[["normalized-losses"]].head(40))

# normalization
print(df[["length"]].head(40))
# simple normalization xnew=xold/xmax  0<x<1
df["length"]=df["length"]/df["length"].max()
print(df[["length"]].head(40))
#min-max normalization xnew=(xold-xmin)/(xmax-xmin)  0<x<1
df["length"]=(df["length"]-df["length"].min())/(df["length"].max()-df["length"].min())
#Z-score normalization xnew=(xold-mean)/std     -3<z<+3 
df["length"]=(df["length"]-df["length"].mean())/(df["length"].std())
print(df[["length"]].head(40))

# rename a column
df.rename(columns={"length":"normalized-length"},inplace=True)
print(df.columns)

# bins
bins=np.linspace(min(df["price"]),max(df["price"]),4) # create line space with three categories
bins_name=["low","medium","high"] # name the categories
df["price-bins"]=pd.cut(df["price"],bins,labels=bins_name,include_lowest=True)
print(df[["price","price-bins"]])

#transform a categorial variable to dummy variable (one-hot encoding method)
print(pd.get_dummies(df["fuel-type"]))

# Exploratory Data Analysis EDA (Statistical Calculation)

# count each kind in a categorial column (object)-value_counts() method
drive_wheels_counts=df["drive-wheels"].value_counts().to_frame()
print(drive_wheels_counts)
drive_wheels_counts.rename(columns={"drive-wheels":"counts"},inplace=True) # change the new column (drive_wheels_counts)
drive_wheels_counts.index.name="drive-wheels" # make a name for the rows (index)
print(drive_wheels_counts)

# get statistical information of each kind in a categorial column as a box plot (low and high quartiles(interquartiles IQ)-median-low and high extremes (1.5 of IQ)-outliers)
import seaborn as sns
sns.boxplot(x="drive-wheels",y="price",data=df)

# draw a scatter plot 
plt.scatter(df["engine-size"],df["price"])
plt.title("scatter plot price vs engine-size")
plt.xlabel("engine-size")
plt.ylabel("price")

# do some calculations as example the mean of the price of a categorial variables (groupby method,pivot,heatmap)
mean_categories=df[["drive-wheels","body-style","price"]].groupby(["drive-wheels","body-style"],as_index=False).mean()
print(mean_categories)

df_pivot=mean_categories.pivot(index="drive-wheels",columns="body-style") # create a table as pivot
print(pivot)


fig, ax = plt.subplots()
im = ax.pcolor(df_pivot, cmap='RdBu') # draw a heatmap with color map Red and Blue
#label names
row_labels = df_pivot.columns.levels[1]
col_labels = df_pivot.index
#move ticks and labels to the center
ax.set_xticks(np.arange(df_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(df_pivot.shape[0]) + 0.5, minor=False)
#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)
#rotate label if too long
plt.xticks(rotation=90)
#show colobar and the heatmap
fig.colorbar(im)
plt.show()


# Correlation:F-test and p-value of two kinds of a categorial column and their effect on the target variable (F large: strong correlation,p small less than 0.001: very good certainty)
from scipy import stats
from scipy.stats import f_oneway
group=df[["make","price"]].groupby(["make"])
F_test,p_value=stats.f_oneway(group.get_group("honda")["price"],group.get_group("subaru")["price"])
print(F_test,p_value)

# Correlation: regression plot
sns.regplot(x="engine-size",y="price",data=df)
plt.ylim(0,)

# Correlation: Pearson coefficient (person_coef=-1 strong negative corr, person_coef=+1 strong positive corr,person_coef=0 no relationship)
print(df.corr()) # print the pearson coefficient between all variable
print(df[["engine-size","price"]].corr()) # print the pearson coefficient between specific columns
pearson_coef,p_value=stats.pearsonr(df["engine-size"],df["price"]) # print the pearson coefficient and p-value for specific columns (p-value<0.001:strong certainty,p<0.05:moderate,p<0.1:weak,p>0.1:no certainty)
print(pearson_coef,p_value)

# Model Development (Regression Functions)

# Single Linear Regression SLR (yhat=b0+b1.x, b0:intercept, b1:coefficient)
x=df[["highway-mpg"]] # independent variable
y=df[["price"]] # dependent variable
from sklearn.linear_model import LinearRegression
lr=LinearRegression() # define the constructor of model SLR
lr.fit(x,y) # do the fitting
yhat=lr.predict(x) # create the yhat=b0+b1.x
print(yhat[0:5]) # print the first 5 values of yhat
print(lr.intercept_) # print b0
print(lr.coef_) # print b1

# Multiple Linear Regression MLR (yhat=b0+b1.x1+b2.x2+.., b0:intercept, b1:coefficient of x1, b2:coefficient of x2,...)
df.dropna(subset=["horsepower"],axis=0,inplace=True)
df.dropna(subset=["curb-weight"],axis=0,inplace=True)
df.dropna(subset=["highway-mpg"],axis=0,inplace=True)
df.dropna(subset=["engine-size"],axis=0,inplace=True)
x=df[["horsepower","curb-weight","engine-size","highway-mpg"]] # independent variable
x=x.astype("float")
y=df[["price"]] # dependent variable
from sklearn.linear_model import LinearRegression
lr=LinearRegression() # define the constructor of model SLR
lr.fit(x,y) # do the fitting
yhat=lr.predict(x) # create the yhat=b0+b1.x
print(yhat[0:5]) # print the first 5 values of yhat
print(lr.intercept_) # print b0
print(lr.coef_) # print b1,b2...

# Model visualisation
# Regression plot for knowing if the SLR is suitable model
import seaborn as sns
width=12
height=10
plt.figure(figsize=(width,height))
sns.regplot(x="highway-mpg",y="price",data=df)
plt.close()
# Residual plot for knowing if the SLR is suitable model (Residual plot has equidistant point between the mean: SLR suitable model,has parabola: not suitable non linear more suitable, has non uniform distribution:not corret)
import seaborn as sns
width=12
height=10
plt.figure(figsize=(width,height))
sns.residplot(df["highway-mpg"],df["price"])
plt.close()
# Distribution plot for knowing if the choosed MLR model suitable or not
# create the MLR model
df.dropna(subset=["horsepower"],axis=0,inplace=True)
df.dropna(subset=["curb-weight"],axis=0,inplace=True)
df.dropna(subset=["highway-mpg"],axis=0,inplace=True)
df.dropna(subset=["engine-size"],axis=0,inplace=True)
x=df[["horsepower","curb-weight","engine-size","highway-mpg"]] # independent variable
x=x.astype("float")
y=df[["price"]] # dependent variable
from sklearn.linear_model import LinearRegression
lr=LinearRegression() # define the constructor of model SLR
lr.fit(x,y) # do the fitting
yhat=lr.predict(x) # create the yhat=b0+b1.x
print(yhat[0:5]) # print the first 5 values of yhat
print(lr.intercept_) # print b0
print(lr.coef_) # print b1,b2...
# Plot the distribution plot
ax1=sns.distplot(y,hist=False,color="r",label="Actual value") # plot the distribution (not the histogram) for the actual price
sns.distplot(yhat,hist=False,color="b",label="predicted value",ax=ax1) # plot the distribution (not the histogram) for the predicted price
plt.title("Actual vs Fitted price")
plt.xlabel("price")
plt.ylabel("number of casrs that have the price x")
plt.show()
plt.close()

# Polynomial Regression
# One-Dimensional Polynomial Regression (yhat=b0+b1x+b2x^2....)
import numpy as np
x=df["highway-mpg"]
y=df["price"]
f=np.polyfit(x,y,3) # fitting as a polynom of degree 3
p=np.poly1d(f) # define the polynom function p(x)
print(p)
#Define PlotPolly function to draw the one dimensional polynomial functions
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)
    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ highway-mpg')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')
    plt.show()
    plt.close()
PlotPolly(p, x, y,'highway-mpg')

# Multi-dimensional Polynomial Linear Regression PLR (yhat=b0+b1x1+b2x2+b3x1x2+b4x1^2+b5x2^2+....)
# Transform the predictor x into polynom of degree n
df.dropna(subset=["horsepower"],axis=0,inplace=True)
df.dropna(subset=["curb-weight"],axis=0,inplace=True)
df.dropna(subset=["highway-mpg"],axis=0,inplace=True)
df.dropna(subset=["engine-size"],axis=0,inplace=True)
x=df[["horsepower","curb-weight","engine-size","highway-mpg"]] # independent variable
x=x.astype("float")
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2,include_bias=False) # define the polynomial constructor
x_pr=pr.fit_transform(x) # transform the x values into polynomial fitting
print(x_pr)
# Scaling the polynomial x values for high dimension n
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x)
x_sc=sc.transform(x)
print(x_sc)
# Pipeline method (Scaling+Polynomial transform+Linear regression) to construct the yhat of PLR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
x=df[["horsepower","curb-weight","engine-size","highway-mpg"]] # independent variable-predictor
y=df[["price"]] # dependent variable-target
x=x.astype("float")
input=[("scale",StandardScaler()),("polynomial",PolynomialFeatures(degree=2)),("model",LinearRegression())]
pipe=Pipeline(input) # create the object pipe as model
pipe.fit(x,y)
yhat=pipe.predict(x)
print(yhat[0:5])

# Measures for In-Sample Evaluation
# Mean-Squared-Error MSE = (squared area of y-yhat)/number of samples (small MSE: good fitting,large MSE: bad fitting)
#for SLR
x=df[["highway-mpg"]] 
y=df[["price"]] 
from sklearn.linear_model import LinearRegression
lr=LinearRegression() 
lr.fit(x,y) 
yhat=lr.predict(x) 
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y,yhat)
print(mse)
#For Polynom
x = df['highway-mpg']
y = df['price']
f=np.polyfit(x,y,3)
p=np.poly1d(f)
mse=mean_squared_error(y,p(x))
print(mse)
# R-squared=1-((MSE of regression line between y and yhat)/(MSE of the average ybar of data between y and ybar))-R close to zero:bad fitting, R close to 1: good fitting
#for SLR,MLR,PLR
x=df[["highway-mpg"]] 
y=df[["price"]] 
from sklearn.linear_model import LinearRegression
lr=LinearRegression() 
lr.fit(x,y) 
yhat=lr.predict(x) 
Rsq=lr.score(x,y)
print(Rsq)
#for polynom
x = df['highway-mpg']
y = df['price']
f=np.polyfit(x,y,3)
p=np.poly1d(f)
from sklearn.metrics import r2_score
Rsq=r2_score(y,p(x))
print(Rsq)

# Model Evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import clean data 
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'
df = pd.read_csv(path)
#df.to_csv(r"C:\Users\Toshiba\Desktop\module_5_auto.csv")

df=df._get_numeric_data()
print(df.head())


# Splitting the data df into train data (for construct the model) and test data (for testing the model)
x_data=df.drop("price",axis=1) # drop the target variable from the predictor variables
y_data=df["price"] # define the target variable
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.1,random_state=0) # split the data into 90% training data and 10% (test_size) testing data
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train[["horsepower"]],y_train) # fitting the training data to construct the llinear regression model
yhat=lr.predict(x_train[["horsepower"]]) # calculate the predicted value yhat
Rsq=lr.score(x_test[["horsepower"]],y_test) # testing the R^2 using the testing data
print(Rsq)

# Distribution plot for the training data (y_train,yhat_train) and distibution plot for testing data (y_test,yhat_test)
# Define DistributionPlot function to use for plotting the data quickly
def DistributionPlot(y,yhat,NameActual,NamePredicted,title):
    width=12
    height=10
    plt.figure(figsize=(width,height))
    ax1=sns.distplot(y,hist=False,color="r",label=NameActual)
    ax2=sns.distplot(yhat,hist=False,color="b",label=NamePredicted,ax=ax1)
    plt.xlabel("price")
    plt.ylabel("number of cars that have x price")
    plt.title(title)
    plt.legend()
    plt.show()
    plt.close()  
# Split the data and define the LR for using the MLR model 
x_data=df.drop("price",axis=1) 
y_data=df["price"] 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.1,random_state=0) from sklearn.linear_model import LinearRegression
lr=LinearRegression()
# Fitting the training data and plot it
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_train)
yhat_train=lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
DistributionPlot(y_train,yhat_train,"Actual value(train)","Predicted value (train)","Distribution plot for training data")
# Fitting the testing data and plot it
lr.fit(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_test)
yhat_test=lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
DistributionPlot(y_test,yhat_test,"Actual value(test)","Predicted value (test)","Distribution plot for testing data")

# Cross validation method uses when we don't have enough testing data 
# This method split the data into folds (one fold for test and the others for train)
# We permute the test fold and calculate the R-sqaured for each permutation then we calculate their mean
x_data=df.drop("price",axis=1)
y_data=df["price"]
from sklearn.linear_model import LinearRegression
lre=LinearRegression()
from sklearn.model_selection import cross_val_score
Rsq=cross_val_score(lr,x_data[["horsepower"]],y_data,cv=4) # split the data into 4-folds (cv)
Rsq_mean=Rsq.mean()
print(Rsq_mean)
# we can show the yhat values for each permutation
from sklearn.model_selection import cross_val_predict
yhat=cross_val_predict(lr,x_data[["horsepower"]],y_data,cv=4)
print(yhat[0:5])

# Using training data to construct a PLR model and calculate the yhat using the testing data then plot the function
x_data=df.drop("price",axis=1) # drop the target variable from the predictor variables
y_data=df["price"] # define the target variable
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.45,random_state=0) # split the data into 90% training data and 10% (test_size) testing data
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=5)
x_train_pr=pr.fit_transform(x_train[["horsepower"]])
x_test_pr=pr.fit_transform(x_test[["horsepower"]])
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train_pr,y_train)
yhat=lr.predict(x_test_pr)
print(yhat[0:4])
# Plot the PLR function
def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
    xmax=max([xtrain.values.max(), xtest.values.max()])
    xmin=min([xtrain.values.min(), xtest.values.min()])
    x=np.arange(xmin, xmax, 0.1)
    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, lr,pr)

# Plot R^2 of the testing data as function of order of polynomial (n=1,2,3,4) 
x_data=df.drop("price",axis=1) # drop the target variable from the predictor variables
y_data=df["price"] # define the target variable
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.45,random_state=0) # split the data into 90% training data and 10% (test_size) testing data
Rsq=[]
order=[1,2,3,4]
for n in order:
    from sklearn.preprocessing import PolynomialFeatures
    pr=PolynomialFeatures(degree=n)
    x_train_pr=pr.fit_transform(x_train[["horsepower"]])
    x_test_pr=pr.fit_transform(x_test[["horsepower"]])
    from sklearn.linear_model import LinearRegression
    lr=LinearRegression()
    lr.fit(x_train_pr,y_train)
    Rsq.append(lr.score(x_test_pr,y_test))
    print(Rsq)
plt.plot(order,Rsq)
plt.xlabel("order of polynom")
plt.ylabel("R^2")
plt.title("R^2 vs order of polynom")
plt.text(3,0.75,"Maximum R^2")
# Overfitting: the model is too flexible and fits the noise not the actual function
# Underfitting: the model is too simple to fit the data

# Ridge Regression: introduce alpha parameter (alpha=0.1,0.001,1,10...) for PLR model of high order - when increase alpha the PLR coefficient b1,b2,... decrease 
x_data=df.drop("price",axis=1) 
y_data=df["price"] 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.1,random_state=0) # split the data into 90% training data and 10% (test_size) testing data
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
from sklearn.linear_model import Ridge
rm=Ridge(alpha=1)
rm.fit(x_train_pr,y_train)
yhat=rm.predict(x_test_pr)
print(yhat[0:4])   
# Plot the R^2 as function of alpha to estimate the best alpha value for PLR of training data and testing data
x_data=df.drop("price",axis=1) 
y_data=df["price"] 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.1,random_state=0) # split the data into 90% training data and 10% (test_size) testing data
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
Rsq_train=[]
Rsq_test=[]
Alpha=10*np.array(range(0,1000))
for alpha in Alpha:
    from sklearn.linear_model import Ridge
    rm=Ridge(alpha=alpha)
    rm.fit(x_train_pr,y_train)
    Rsq_train.append(rm.score(x_train_pr,y_train))
    Rsq_test.append(rm.score(x_test_pr,y_test))
width=12
height=10
plt.figure(figsize=(width,height))
plt.plot(Alpha,Rsq_train,"r",label="R^2 training data")
plt.plot(Alpha,Rsq_test,"b",label="R^2 testing data")
plt.xlabel("alpha")
plt.ylabel("R^2")
plt.title("R^2 for different alpha")
plt.legend()

#Grid Search: For estimate the best alpha value for a Ridge Regression 
y_data=df["price"]
x_data=df.drop("price",axis=1)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
parameters= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]
Rm=Ridge()
Grid = GridSearchCV(Rm, parameters,cv=4)
Grid.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
BestRm=Grid.best_estimator_  # best alpha value
print(BestRm)
test_score=BestRm.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)
print(test_score)
score=Grid.cv_results_ # The resulting scores of the different free parameters are stored in this dictionary: cv+results
print(score)

#*************************************************************************************************************************************

print("PY103: Data Visualization with Python")

## import the data of Canadian Immigration dataset from the net and cleanning it

import numpy as np 
import pandas as pd 

df = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)   # This data contains information about 20 rows before the dataset and 2 footnote to skip it we used skiprow and skipfooter

print(df)
print(df.head())
print(df.tail())
print(df.columns)
df.drop(["Type","Coverage","AREA","REG","DEV",],axis=1,inplace=True)
df.rename(columns={"OdName":"Country","AreaName":"Continent","RegName":"Region"},inplace=True)
print(df.columns)
print(df.columns.tolist())
print(df.index)
print(df.index.tolist())
print(df.shape)
print(df.dtypes)
print(df.info())
df["Total"]=df.sum(axis=1) #add column of name "Total" summation of each row
print(df[["Total"]])
#df.to_excel(r"C:\Users\Toshiba\Desktop\Immigration.xlsx")
print(df.describe(include="all"))
# select columns
print(df.Country)
print(df[1980])
print(df[["Country",1980]])
#select rows and columns
print(df.iloc[0:5,0:5])  # index location [rows,columns]
print(df.iloc[87]) # index location [row_index]
# to select only the Japan in the Country Columns we should use the set_index method when we finish we use the reset_index method to remove the Country selection
df.set_index("Country",inplace=True)
print(df.loc["Japan"])
print(df.loc["Japan",2013])
print(df.loc["Japan",[1980,1981,1982,1983,1984]])
df.reset_index("Country",inplace=True)
# Column names that are integers (such as the years) might introduce some confusion For example, when we are referencing the year 2013, one might confuse that when the 2013th positional index To avoid this ambuigity, let's convert the column names into strings: '1980' to '2013'
df.columns=list(map(str,df.columns))
print(df.columns)
years=list(map(str,range(1980,2014)))
print(years)
#Filtering based on a criteria: To filter the dataframe based on a condition, we simply pass the condition as a boolean vector. For example, Let's filter the dataframe to show the data on Asian countries (Continent = Asia) and (Region = Southern Asia)
condition1 = df['Continent'] == 'Asia'
print(condition1)
condition2=df["Region"]=="Southern Asia"
print(condition2)
condition3=df[(df["Continent"]=="Asia") & (df["Region"]=="Southern Asia")]
print(condition3)


## Matplotlib: Line Plot

import numpy as np 
import pandas as pd 

df = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)   


df.drop(["Type","Coverage","AREA","REG","DEV",],axis=1,inplace=True)
df.rename(columns={"OdName":"Country","AreaName":"Continent","RegName":"Region"},inplace=True)
df["Total"]=df.sum(axis=1) 
print(df)


import matplotlib as mpl
import matplotlib.pyplot as plt

# optional: check if Matplotlib is loaded
print('Matplotlib version: ', mpl.__version__)
# optional: apply a style to Matplotlib
print(plt.style.available)
mpl.style.use(['ggplot']) # optional: for ggplot-like style

# Plotting in pandas:Plotting in pandas is as simple as appending a df.plot() method to a series or dataframe
# Line Plot For Haiti as functions of years
df.set_index("Country",inplace=True)
df.columns=list(map(str,df.columns))
years=list(map(str,range(1980,2014)))
haiti=df.loc["Haiti",years]
haiti.index = haiti.index.map(int) # let's change the index values (years) of haiti object to type integer for plotting

haiti.plot(kind="line")
plt.title("Immigration of Haiti")
plt.xlabel("Years")
plt.ylabel("Number of Immigrants")
plt.text(2010,6000,"Earthquake 2010")
plt.show()

# Line Plot For India and china as functions of years
df.set_index("Country",inplace=True)
years=list(map(int,range(1980,2014)))
India_china=df.loc[["India","China"],years]
# Recall that pandas plots the indices on the x-axis and the columns as individual lines on the y-axis. Since India_China is a dataframe with the country as the index and years as the columns, we must first transpose the dataframe using transpose() method to swap the row and columns.
India_china=India_china.transpose()
India_china.plot(kind="line")
plt.title("Immigration of India and China")
plt.xlabel("Years")
plt.ylabel("Number of Immigrants")
plt.show()

#Note: How come we didn't need to transpose Haiti's dataframe before plotting (like we did for India_China)? That's because haiti is a series as opposed to a dataframe, and has the years as its indices print(haiti.head())

## Matplotlib: Area Plot

import numpy as np 
import pandas as pd 

df = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)   


df.drop(["Type","Coverage","AREA","REG","DEV",],axis=1,inplace=True)
df.rename(columns={"OdName":"Country","AreaName":"Continent","RegName":"Region"},inplace=True)
df["Total"]=df.sum(axis=1) 
print(df)


import matplotlib as mpl
import matplotlib.pyplot as plt

df.sort_values("Total",ascending=False,axis=0,inplace=True) # sort the data in decending order relative to the cumulative total
print(df.head())
top5=df.head(5)
print(top5)
years=list(map(int,range(1980,2014)))
top5=top5[years].transpose()

top5.plot(kind="area",alpha=0.7,  # 0 - 1, default value alpha = 0.5
             stacked=False,
             figsize=(20, 10))      #Area plots are stacked by default. And to produce a stacked area plot, each column must be either all positive or all negative values (any NaN, i.e. not a number, values will default to 0). To produce an unstacked plot, set parameter stacked to value False. The unstacked plot has a default transparency (alpha value) at 0.5. We can modify this value by passing in the alpha parameter.

plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')
plt.show()

# Note: in the case of multipleplots we prefer to use the Artist layer instead of scripting layer of matplotlib artifieciels so we use the Axes instance of your current plot "ax"

ax = top5.plot(kind='area', alpha=0.35, figsize=(20, 10))

ax.set_title('Immigration Trend of Top 5 Countries')
ax.set_ylabel('Number of Immigrants')
ax.set_xlabel('Years') 

## Matplotlib: Histograms
import numpy as np 
import pandas as pd 

df = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)   


df.drop(["Type","Coverage","AREA","REG","DEV",],axis=1,inplace=True)
df.rename(columns={"OdName":"Country","AreaName":"Continent","RegName":"Region"},inplace=True)
df["Total"]=df.sum(axis=1) 
print(df)


import matplotlib as mpl
import matplotlib.pyplot as plt

df.columns=list(map(str,df.columns))
count,bin_edges=np.histogram(df["2013"])
print(count) # frequency count
print(bin_edges) # bin ranges, default = 10 bins

df["2013"].plot(kind="hist",figsize=(8, 5),xticks=bin_edges)

plt.title('Histogram of Immigration from 195 Countries in 2013')
plt.ylabel('Number of Countries')
plt.xlabel('Number of Immigrants')
plt.show()

#Exerise:What is the immigration distribution for Denmark, Norway, and Sweden for years 1980 - 2013?
years=list(map(int,range(1980,2013)))
df.set_index("Country",inplace=True)
df1=df.loc[["Denmark","Norway","Sweden"],years].transpose()
print(df1)

count,bin_edges=np.histogram(df1,15)  # split the bins into 15 instead of 10
xmin = bin_edges[0] - 10   #  first bin value is 31.0, adding buffer of 10 for aesthetic purposes 
xmax = bin_edges[-1] + 10  #  last bin value is 308.0, adding buffer of 10 for aesthetic purposes

df1.plot(kind ='hist', 
          figsize=(10, 6),
          bins=15,
          alpha=0.6,
          xticks=bin_edges,
          color=['coral', 'darkslateblue', 'mediumseagreen'],
          stacked=True,xlim=(xmin, xmax)) #If we do not want the plots to overlap each other, we can stack them using the stacked parameter
plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')
plt.show()

##Matplotlib: Bar chart

import numpy as np 
import pandas as pd 

df = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)   


df.drop(["Type","Coverage","AREA","REG","DEV",],axis=1,inplace=True)
df.rename(columns={"OdName":"Country","AreaName":"Continent","RegName":"Region"},inplace=True)
df["Total"]=df.sum(axis=1) 
print(df)


import matplotlib as mpl
import matplotlib.pyplot as plt

df.set_index("Country",inplace=True)
years=list(map(int,range(1980,2013)))
iceland=df.loc["Iceland",years]
print(iceland)

iceland.plot(kind="bar",figsize=(10, 6), rot=90)  # rotate the xticks(labelled points on x-axis) by 90 degrees
plt.title("Bar chart of iceland")
plt.xlabel("Years")
plt.ylabel("Immigrants")
# Annotate arrow
plt.annotate('',  # s: str. Will leave it blank for no text
             xy=(5, 30),  # place head of the arrow at point (year 2012 , pop 70)
             xytext=(28, 20),  # place base of the arrow at point (year 2008 , pop 20)
             xycoords='data',  # will use the coordinate system of the object being annotated
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', lw=5))
# Annotate Text
plt.annotate('2008 - 2011 Financial Crisis',  # text to display
             xy=(28, 30),  # start the text at at point (year 2008 , pop 30)
             rotation=72.5,  # based on trial and error to match the arrow
             va='bottom',  # want the text to be vertically 'bottom' aligned
             ha='left',  # want the text to be horizontally 'left' algned)
plt.show()

#Horizontal bar chart
df.sort_values("Total",ascending=False,axis=0,inplace=True)
top15=df["Total"].head(15)
print(top15)

top15.plot(kind="barh",figsize=(10, 6), rot=90,color='steelblue')  
plt.title("Bar chart of Immigrants from top 15")
plt.xlabel("Immigrants")
plt.ylabel("Country")
# annotate value labels to each country
for index, value in enumerate(top15): 
    label = format(int(value), ',') # format int with commas
# place text at the end of bar (subtracting 47000 from x, and 0.1 from y to make it fit within the bar)
    plt.annotate(label, xy=(value - 47000, index - 0.10), color='white')
plt.show()

##Matplotlin: Pie chart

import numpy as np 
import pandas as pd 

df = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)   


df.drop(["Type","Coverage","AREA","REG","DEV",],axis=1,inplace=True)
df.rename(columns={"OdName":"Country","AreaName":"Continent","RegName":"Region"},inplace=True)
df["Total"]=df.sum(axis=1) 
print(df)


import matplotlib as mpl
import matplotlib.pyplot as plt

continent=df.groupby("Continent",axis=0).sum() # same the number of immigrants in each continent
print(continent)

colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
explode_list = [0.1, 0, 0, 0, 0.1, 0.1] # ratio for each continent with which to offset each wedge.

continent['Total'].plot(kind='pie',
                            figsize=(15, 6),
                            autopct='%1.1f%%',   # add in percentages
                            startangle=90,       # start angle 90° (Africa)
                            shadow=True,         # add shadow     
                            labels=None,         # turn off labels on pie chart
                            pctdistance=1.12,    # the ratio between the center of each pie slice and the start of the text generated by autopct 
                            colors=colors_list,  # add custom colors
                            explode=explode_list # 'explode' lowest 3 continents
                            )

# scale the title up by 12% to match pctdistance
plt.title('Immigration to Canada by Continent [1980 - 2013]', y=1.12) 
plt.axis('equal') # Sets the pie chart to look like a circle.
# add legend
plt.legend(labels=continent.index, loc='upper left') 
plt.show()

##Matplotlib: Box Plot

import numpy as np 
import pandas as pd 

df = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)   


df.drop(["Type","Coverage","AREA","REG","DEV",],axis=1,inplace=True)
df.rename(columns={"OdName":"Country","AreaName":"Continent","RegName":"Region"},inplace=True)
df["Total"]=df.sum(axis=1) 
print(df)


import matplotlib as mpl
import matplotlib.pyplot as plt

df.set_index("Country",inplace=True)
years=list(map(int,range(1980,2013)))
japan=df.loc["Japan",years]
print(japan)

japan.plot(kind="box", figsize=(8, 6))
plt.title('Box plot of Japanese Immigrants from 1980 - 2013')
plt.ylabel('Number of Immigrants')
plt.show()

#Exercise: Compare the distribution of the number of new immigrants from India and China for the period 1980 - 2013
df.set_index("Country",inplace=True)
years=list(map(int,range(1980,2013)))
India_China=df.loc[["India","China"],years].transpose()
print(India_China)

India_China.plot(kind="box", figsize=(8, 6))
plt.title('Box plot of India_China Immigrants from 1980 - 2013')
plt.ylabel('Number of Immigrants')
plt.show()

#if you prefer to create horizontal box plots, you can pass the vert parameter in the plot function and assign it to False. You can also specify a different color in case you are not a big fan of the default red color.
df.set_index("Country",inplace=True)
years=list(map(int,range(1980,2013)))
India_China=df.loc[["India","China"],years].transpose()
print(India_China)

India_China.plot(kind="box", figsize=(8, 6),color='blue', vert=False)
plt.title('Box plot of India_China Immigrants from 1980 - 2013')
plt.ylabel('Number of Immigrants')
plt.show()

#Exercise:  Create a box plot to visualize the distribution of the top 15 countries (based on total immigration) grouped by the decades 1980s, 1990s, and 2000s

df.columns=list(map(str,df.columns))
df.sort_values("Total",ascending=False,axis=0,inplace=True)
top15=df.head(15)
years_80s=list(map(str,range(1980,1990)))
df_80s=top15.loc[:,years_80s].sum(axis=1)
years_90s=list(map(str,range(1990,2000)))
df_90s=top15.loc[:,years_90s].sum(axis=1)
years_00s=list(map(str,range(2000,2010)))
df_00s=top15.loc[:,years_00s].sum(axis=1)
print(df_80s,df_90s,df_00s)
df_centuries=pd.DataFrame({"1980s":df_80s,"1990s":df_90s,"2000s":df_00s})
print(df_centuries)
print(df_centuries.describe())

df_centuries.plot(kind="box",figsize=(10, 6))
plt.title('Immigration from top 15 countries for decades 80s, 90s and 2000s')
plt.show()

##Matplotlin: Subplots

import numpy as np 
import pandas as pd 

df = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)   


df.drop(["Type","Coverage","AREA","REG","DEV",],axis=1,inplace=True)
df.rename(columns={"OdName":"Country","AreaName":"Continent","RegName":"Region"},inplace=True)
df["Total"]=df.sum(axis=1) 
print(df)


import matplotlib as mpl
import matplotlib.pyplot as plt

df.set_index("Country",inplace=True)
years=list(map(int,range(1980,2013)))
India_China=df.loc[["India","China"],years].transpose()
print(India_China)

fig = plt.figure() # create figure

ax0 = fig.add_subplot(1, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)
ax1 = fig.add_subplot(1, 2, 2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**

# Subplot 1: Box plot
India_China.plot(kind='box', color='blue', vert=False, figsize=(20, 6), ax=ax0) # add to subplot 1
ax0.set_title('Box Plots of Immigrants from China and India (1980 - 2013)')
ax0.set_xlabel('Number of Immigrants')
ax0.set_ylabel('Countries')

# Subplot 2: Line plot
India_China.plot(kind='line', figsize=(20, 6), ax=ax1) # add to subplot 2
ax1.set_title ('Line Plots of Immigrants from China and India (1980 - 2013)')
ax1.set_ylabel('Number of Immigrants')
ax1.set_xlabel('Years')

plt.show()

##Matplotlib: Scatter Plot

import numpy as np 
import pandas as pd 

df = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)   


df.drop(["Type","Coverage","AREA","REG","DEV",],axis=1,inplace=True)
df.rename(columns={"OdName":"Country","AreaName":"Continent","RegName":"Region"},inplace=True)
df["Total"]=df.sum(axis=1) 
print(df)


import matplotlib as mpl
import matplotlib.pyplot as plt

df.columns=list(map(str,df.columns))
years=list(map(str,range(1980,2014)))
total=pd.DataFrame(df[years].sum(axis=0)) # create a df of name total with two columns (years,total)
print(total)
total.index=map(int,total.index) # change the years to type int
total.reset_index(inplace=True) # add the index column to total df
total.columns=["year","total"] # rename the two coulmns
print(total)

total.plot(kind='scatter', x='year', y='total', figsize=(15, 6), color='darkblue')
plt.title('Total Immigration to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.show()

x=total["year"]
y=total["total"]
print(total.dtypes)
f=np.polyfit(x,y,deg=1)
print(f[0],f[1])
p=np.poly1d(f)
print(p)

plt.plot(x,p(x),color="red")
plt.annotate('y={0:.0f} x + {1:.0f}'.format(f[0], f[1]), xy=(2000, 150000))
plt.show()

print("Estimated No. of Immigrants in 2015:",f[1]+2015*f[0])

#Exercise: Create a scatter plot of the total immigration from Denmark, Norway, and Sweden to Canada from 1980 to 2013?
df.columns=list(map(str,df.columns))
years=list(map(str,range(1980,2014)))
df.set_index("Country",inplace=True)
DNS=pd.DataFrame(df.loc[["Denmark","Norway","Sweden"],years].sum(axis=0))
DNS.index=map(int,DNS.index)
DNS.reset_index(inplace=True)
DNS.columns=["year","total"]
print(DNS)
print(DNS.shape)

DNS.plot(kind="scatter",x="year",y="total",figsize=(10,6),color='darkblue')
plt.title('Immigration from Denmark, Norway, and Sweden to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.show()


##Matplotlib: Bubble Plots

import numpy as np 
import pandas as pd 

df = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)   


df.drop(["Type","Coverage","AREA","REG","DEV",],axis=1,inplace=True)
df.rename(columns={"OdName":"Country","AreaName":"Continent","RegName":"Region"},inplace=True)
df["Total"]=df.sum(axis=1) 
print(df)


import matplotlib as mpl
import matplotlib.pyplot as plt

df.columns=list(map(str,df.columns))
years=list(map(str,range(1980,2014)))
df.set_index("Country",inplace=True)

# transposed dataframe
dft = df[years].transpose()
# cast the Years (the index) to type int
dft.index = map(int, dft.index)
# let's label the index. This will automatically be the column name when we reset the index
dft.index.name = 'Year'
# reset index to bring the Year in as a column
dft.reset_index(inplace=True)
# view the changes
print(dft.head())

# normalize Brazil data
norm_brazil = (dft['Brazil'] - dft['Brazil'].min()) / (dft['Brazil'].max() - dft['Brazil'].min())
# normalize Argentina data
norm_argentina = (dft['Argentina'] - dft['Argentina'].min()) / (dft['Argentina'].max() - dft['Argentina'].min())

# Brazil
ax0 = dft.plot(kind='scatter',
                    x='Year',
                    y='Brazil',
                    figsize=(14, 8),
                    alpha=0.5,  # transparency
                    color='green',
                    s=norm_brazil * 2000 + 10,  # pass in weights 
                    xlim=(1975, 2015)
                    )

# Argentina
ax1 = dft.plot(kind='scatter',
                    x='Year',
                    y='Argentina',
                    alpha=0.5,
                    color="blue",
                    s=norm_argentina * 2000 + 10,
                    ax=ax0
                    )

ax0.set_ylabel('Number of Immigrants')
ax0.set_title('Immigration from Brazil and Argentina from 1980 to 2013')
ax0.legend(['Brazil', 'Argentina'], loc='upper left', fontsize='x-large')

## Wafle Charts

import numpy as np 
import pandas as pd 

df = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)   


df.drop(["Type","Coverage","AREA","REG","DEV",],axis=1,inplace=True)
df.rename(columns={"OdName":"Country","AreaName":"Continent","RegName":"Region"},inplace=True)
df["Total"]=df.sum(axis=1) 
print(df)


import matplotlib as mpl
import matplotlib.pyplot as plt


def create_waffle_chart(categories, values, height, width, colormap, value_sign=''):

    # compute the proportion of each category with respect to the total
    total_values = sum(values)
    category_proportions = [(float(value) / total_values) for value in values]

    # compute the total number of tiles
    total_num_tiles = width * height # total number of tiles
    print ('Total number of tiles is', total_num_tiles)
    
    # compute the number of tiles for each catagory
    tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]

    # print out number of tiles per category
    for i, tiles in enumerate(tiles_per_category):
        print (dsn.index.values[i] + ': ' + str(tiles))
    
    # initialize the waffle chart as an empty matrix
    waffle_chart = np.zeros((height, width))

    # define indices to loop through waffle chart
    category_index = 0
    tile_index = 0

    # populate the waffle chart
    for col in range(width):
        for row in range(height):
            tile_index += 1

            # if the number of tiles populated for the current category 
            # is equal to its corresponding allocated tiles...
            if tile_index > sum(tiles_per_category[0:category_index]):
                # ...proceed to the next category
                category_index += 1       
            
            # set the class value to an integer, which increases with class
            waffle_chart[row, col] = category_index
    
    # instantiate a new figure object
    fig = plt.figure()

    # use matshow to display the waffle chart
    colormap = plt.cm.coolwarm
    plt.matshow(waffle_chart, cmap=colormap)
    plt.colorbar()

    # get the axis
    ax = plt.gca()

    # set minor ticks
    ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
    ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
    # add dridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    plt.xticks([])
    plt.yticks([])

    # compute cumulative sum of individual categories to match color schemes between chart and legend
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]

    # create legend
    import matplotlib.patches as mpatches
    legend_handles = []
    for i, category in enumerate(categories):
        if value_sign == '%':
            label_str = category + ' (' + str(values[i]) + value_sign + ')'
        else:
            label_str = category + ' (' + value_sign + str(values[i]) + ')'
            
        color_val = colormap(float(values_cumsum[i])/total_values)
        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

    # add legend to chart
    plt.legend(
        handles=legend_handles,
        loc='lower center', 
        ncol=len(categories),
        bbox_to_anchor=(0., -0.2, 0.95, .1)
    )
    plt.show()

#For example, say immigration from Scandinavia to Canada is comprised only of immigration from Denmark, Norway, and Sweden, and we're interested in visualizing the contribution of each of these countries to the Scandinavian immigration to Canada. Note that only we change the dataframe dsn in the above definition when you will use the waffle chart for a new data frame

df.set_index("Country",inplace=True)
# let's create a new dataframe for these three countries 
dsn = df.loc[['Denmark', 'Norway', 'Sweden'], :]
# let's take a look at our dataframe
print(dsn)

width = 40 # width of chart
height = 10 # height of chart
categories = dsn.index.values # categories
values = dsn['Total'] # correponding values of categories
colormap = plt.cm.coolwarm # color map class

create_waffle_chart(categories, values, height, width, colormap)

## Word Clouds

#Application: Alice-text

# import package and its set of stopwords
from wordcloud import WordCloud, STOPWORDS

# open the file and read it into a variable alice_novel
import urllib
alice_novel = urllib.request.urlopen('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/alice_novel.txt').read().decode("utf-8")

# We use the function set to remove any redundant stopwords
stopwords = set(STOPWORDS)
# Much better! However, said isn't really an informative word. So let's add it to our stopwords and re-generate the cloud
stopwords.add('said') # add the words said to stopwords

# instantiate a word cloud object
alice_wc = WordCloud(
    background_color='white',
    max_words=2000, # using only the first 2000 words in the novel
    stopwords=stopwords)

# generate the word cloud
alice_wc.generate(alice_novel)

# display the word cloud
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(14, 18))
plt.imshow(alice_wc, interpolation='bilinear') # image show
plt.axis('off') # remove the axis
plt.show()

# Let's use a mask of Alice

# save mask to alice_mask
import numpy as np
alice_mask = np.array(Image.open(urllib.request.urlopen("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/labs/Module%204/images/alice_mask.png")))
# Let's take a look at how the mask looks like
fig = plt.figure(figsize=(14, 18))
plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis('off')
plt.show()
# instantiate a word cloud object
alice_wc = WordCloud(background_color='white', max_words=2000, mask=alice_mask, stopwords=stopwords)

# generate the word cloud
alice_wc.generate(alice_novel)

# display the word cloud
fig = plt.figure(figsize=(14, 18))

plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()


## Seaborn: Regression Plot

import numpy as np 
import pandas as pd 

df = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)   


df.drop(["Type","Coverage","AREA","REG","DEV",],axis=1,inplace=True)
df.rename(columns={"OdName":"Country","AreaName":"Continent","RegName":"Region"},inplace=True)
print(df)


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

years=list(map(int,range(1980,2014)))
total=pd.DataFrame(df[years].sum(axis=0))
total.reset_index(inplace=True)
total.columns=["year","total"]
print(total)

plt.figure(figsize=(15, 10))

sns.set(font_scale=1.5) # increase the font size of the tickmark labels, the title, and the x- and y-labels so they don't feel left out!
sns.set_style('ticks')  # change background to white background or sns.set_style('whitegrid')
ax = sns.regplot(x='year', y='total', data=total, color='green', marker='+', scatter_kws={'s': 200}) #  marker:customize the marker shape,scatter_kws:increase the size of markers
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.show()

#Exercise: Use seaborn to create a scatter plot with a regression line to visualize the total immigration from Denmark, Sweden, and Norway to Canada from 1980 to 2013.

years=list(map(int,range(1980,2014)))
df.set_index("Country",inplace=True)
dsn=pd.DataFrame(df.loc[["Denmark","Sweden","Norway"],years].transpose().sum(axis=1))
dsn.reset_index(inplace=True)
dsn.columns=["year","total"]
print(dsn)

plt.figure(figsize=(15, 10))
sns.set(font_scale=1.5) 
sns.set_style('ticks')  
ax = sns.regplot(x='year', y='total', data=dsn, color='green', marker='+', scatter_kws={'s': 200}) #  marker:customize the marker shape,scatter_kws:increase the size of markers
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration from Denmark, Sweden, and Norway to Canada from 1980 - 2013')
plt.show()

## Folium (display map in Jupyter Notebook not Spyder)

## Create Maps

import folium
import numpy as np  
import pandas as pd 

#create the world map
world_map=folium.Map()
world_map

# create a Stamen Toner map of the world centered around Canada
world_map_toner = folium.Map(location=[56.130, -106.35], zoom_start=4, tiles='Stamen Toner') # locatiom(latitude,longitude) of canada,Zoom_start: increasing for more zooming, tiles: Stamen toner for river and coastal zones in black color map
world_map_toner

# create a Stamen Toner map of the world centered around Canada
world_map_terrain = folium.Map(location=[56.130, -106.35], zoom_start=4, tiles='Stamen Terrain') # tiles: Stamen terrain for hill shading and natural vegetation colors
world_map_terrain

## Maps with Marker

import folium
import pandas as pd

df_incidents = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Police_Department_Incidents_-_Previous_Year__2016_.csv')
df_incidents.shape

# get the first 100 crimes in the df_incidents dataframe
df_incidents = df_incidents.iloc[0:100, :]
df_incidents

# San Francisco latitude and longitude values
latitude = 37.77
longitude = -122.42

# create map and display it
sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)

# display the map of San Francisco
sanfran_map

# instantiate a feature group for the incidents in the dataframe
incidents = folium.map.FeatureGroup()

# loop through the 100 crimes and add each to the incidents feature group
for lat, lng, in zip(df_incidents.Y, df_incidents.X):
    incidents.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            radius=5, # define how big you want the circle markers to be
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )

# add pop-up text to each marker on the map
latitudes = list(df_incidents.Y)
longitudes = list(df_incidents.X)
labels = list(df_incidents.Category)

for lat, lng, label in zip(latitudes, longitudes, labels):
    folium.Marker([lat, lng], popup=label).add_to(sanfran_map)    
    
# add incidents to map
sanfran_map.add_child(incidents)

## Choropleth Maps

import pandas as pd 
import numpy as np
import folium

df_can = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)

print(df_can.head())
# print the dimensions of the dataframe
print(df_can.shape)
# clean up the dataset to remove unnecessary columns (eg. REG) 
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)

# let's rename the columns so that they make sense
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace=True)

# for sake of consistency, let's also make all column labels of type string
df_can.columns = list(map(str, df_can.columns))

# add total column
df_can['Total'] = df_can.sum(axis=1)

# years that we will be using in this lesson - useful for plotting later on
years = list(map(str, range(1980, 2014)))
print ('data dimensions:', df_can.shape)
df_can.head()

# download countries geojson file
#! wget --quiet https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/world_countries.json
    
#print('GeoJSON file downloaded!')

world_geo = r'world_countries.json' # geojson file


# create a numpy array of length 6 and has linear spacing from the minimum total immigration to the maximum total immigration
threshold_scale = np.linspace(df_can['Total'].min(),
                              df_can['Total'].max(),
                              6, dtype=int)
threshold_scale = threshold_scale.tolist() # change the numpy array to a list
threshold_scale[-1] = threshold_scale[-1] + 1 # make sure that the last value of the list is greater than the maximum immigration

# let Folium determine the scale.
world_map = folium.Map(location=[0, 0], zoom_start=2)
world_map.choropleth(
    geo_data=world_geo,
    data=df_can,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    threshold_scale=threshold_scale,
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Immigration to Canada',
    reset=True
)
world_map

#*************************************************************************************************************************************************8
